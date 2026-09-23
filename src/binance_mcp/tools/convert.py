"""Binance Convert (`/sapi/v1/convert`) — quotes, conversions, history and limit orders.

Convert is Binance's quote-and-accept swap rail: you ask for a quote (`getQuote`), which
reserves a `quoteId` and a ratio for 10 s–2 min, and then you either accept it
(`acceptQuote`) or let it expire. Nine tools:

- `binance_get_convert_pairs`             GET  .../exchangeInfo          (public, IP 3000)
- `binance_get_convert_asset_info`        GET  .../assetInfo             (signed)
- `binance_get_convert_quote`             POST .../getQuote              (signed, POST-read)
- `binance_accept_convert_quote`          POST .../acceptQuote           (**real money**)
- `binance_get_convert_order_status`      GET  .../orderStatus           (signed)
- `binance_get_convert_history`           GET  .../tradeFlow             (signed, UID 3000)
- `binance_place_convert_limit_order`     POST .../limit/placeOrder      (**real money**)
- `binance_cancel_convert_limit_order`    POST .../limit/cancelOrder     (**real money**)
- `binance_get_convert_open_limit_orders` GET  .../limit/queryOpenOrders (signed, UID 3000)

The three fund-moving tools pass through the **kill-switch that lives in
`binance_mcp.client`**, not here: a signed non-GET request is refused with
`TradingDisabledError` (surfaced as `Error: … trading is disabled …`) unless
`BINANCE_ALLOW_TRADING=1`. `POST /sapi/v1/convert/getQuote` is on the client's
`POST_READ_ALLOWLIST`, so quoting works with the kill-switch off — which is why every
gated docstring says to price the conversion there first.

Amounts and prices are **strings** end to end: validated as positive decimals and
forwarded verbatim, because a float round-trip (0.1 + 0.2) silently breaks precision.

Windows: `tradeFlow` is the only history endpoint here and Binance requires BOTH
`startTime` and `endTime`, at most 30 days apart, with no cursor parameter of any kind.
The only continuation it offers is the `moreData` flag plus re-asking the same window
with a narrower `endTime` — that is what `binance_get_convert_history` automates.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Literal, Self

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from binance_mcp.client import BinanceClient, get_client
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import ResponseFormat, clip_response, epoch_to_human, fmt_num, to_json
from binance_mcp.server import mcp

CONVERT_EXCHANGE_INFO_PATH = "/sapi/v1/convert/exchangeInfo"
CONVERT_ASSET_INFO_PATH = "/sapi/v1/convert/assetInfo"
CONVERT_GET_QUOTE_PATH = "/sapi/v1/convert/getQuote"
CONVERT_ACCEPT_QUOTE_PATH = "/sapi/v1/convert/acceptQuote"
CONVERT_ORDER_STATUS_PATH = "/sapi/v1/convert/orderStatus"
CONVERT_TRADE_FLOW_PATH = "/sapi/v1/convert/tradeFlow"
CONVERT_LIMIT_PLACE_PATH = "/sapi/v1/convert/limit/placeOrder"
CONVERT_LIMIT_CANCEL_PATH = "/sapi/v1/convert/limit/cancelOrder"
CONVERT_LIMIT_OPEN_ORDERS_PATH = "/sapi/v1/convert/limit/queryOpenOrders"

# "The max interval between startTime and endTime is 30 days" (tradeFlow, inventory H).
MAX_CONVERT_WINDOW_MS = 30 * 24 * 60 * 60 * 1000
# Walk windows stay a day under the documented cap, the way pay.py stays under 90 days:
# the endpoint rejects the whole call when the span is judged too wide, and losing a walk
# to an off-by-one at the boundary costs far more than one extra window.
WALK_WINDOW_MS = 29 * 24 * 60 * 60 * 1000
# tradeFlow's own `limit`: default 100, max 1000 (inventory H).
MAX_PAGE_LIMIT = 1000
DEFAULT_PAGE_LIMIT = 100
# UID weight 3000 per tradeFlow call -> 72,000 UID of the 180,000/min budget at the default.
DEFAULT_MAX_CALLS = 24

# Bare asset codes are ASCII-only, 1-20 chars (the real one-letter assets W and S exist).
_ASSET_PATTERN = re.compile(r"[A-Z0-9]{1,20}")
# Amounts travel verbatim: a plain unsigned decimal, no exponent, no sign, no leading dot.
_DECIMAL_PATTERN = re.compile(r"\d+(\.\d+)?")

# Context-window guards (rule 8) on top of the API's own `limit`. These bound MARKDOWN
# display only; `response_format="json"` always carries every row that was fetched.
MAX_DISPLAY_ROWS = 50
MAX_PAIR_ROWS = 100
MAX_ASSET_ROWS = 100

# `acceptQuote` / `orderStatus` statuses, verbatim from Binance. Only SUCCESS means the
# funds actually moved; nothing here is ever paraphrased as "converted" on its own.
_ORDER_STATUS_NOTES: dict[str, str] = {
    "PROCESS": (
        "Binance accepted the order and is still processing it — the conversion is **not** confirmed. "
        "Poll `binance_get_convert_order_status` with this orderId."
    ),
    "ACCEPT_SUCCESS": (
        "The quote was accepted; settlement is still in flight — the conversion is **not** confirmed complete. "
        "Poll `binance_get_convert_order_status` with this orderId."
    ),
    "SUCCESS": "The conversion completed: the assets have been exchanged.",
    "FAIL": "The conversion **failed** — no assets were exchanged. Request a fresh quote and try again.",
}


# -- helpers -------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _to_ms(value: Any) -> int | None:
    """Accept an epoch-ms int, a >=12-digit epoch-ms numeric string, or an ISO-8601 string.

    Raises ValueError on anything else — including a short digit string, which is
    ambiguous between an epoch and a bare year/id rather than silently misread.
    """
    if value is None or isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    digits = text[1:] if text.startswith("-") else text
    if digits.isdigit() and len(digits) >= 12:
        return int(text)
    iso = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        raise ValueError(f"{value!r} is not an epoch-ms integer (>=12 digits) or an ISO-8601 string") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def _as_ms(value: int | str | None) -> int | None:
    """Narrow an already-validated timestamp field back to `int` for typing purposes.

    The model's `mode="before"` validator has already normalized any raw input to
    `int | None` by construction time; the `str` arm only exists in the *declared* field
    type so the tool's JSON schema advertises ISO-8601 as an accepted input shape.
    """
    if value is None or isinstance(value, int):
        return value
    return _to_ms(value)


def _normalize_asset(value: str) -> str:
    """Uppercase and validate a bare asset code (`BTC`, and the real one-letter assets W/S).

    ASCII only: `str.isalnum()` accepts Unicode digits and letters (`\u00b2`, `\u0431`), which
    Binance does not.
    """
    asset = value.strip().upper()
    if _ASSET_PATTERN.fullmatch(asset) is None:
        raise ValueError(f"asset must be 1-20 characters of A-Z / 0-9 (e.g. 'BTC', 'USDT'); got {value!r}.")
    return asset


def _check_positive_decimal(value: str | None, field_name: str) -> str | None:
    """Validate as a positive decimal but return the ORIGINAL string, unrounded."""
    if value is None:
        return None
    if _DECIMAL_PATTERN.fullmatch(value) is None:
        # Decimal() happily parses '1E+5', '+5', '.5', 'Infinity' and 'NaN'; Binance's
        # filters do not. Reject the shape before parsing rather than forwarding it.
        raise ValueError(
            f"{field_name} must be a plain decimal string like '0.01' or '500' — no exponent, sign or "
            f"leading dot; got {value!r}."
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(
            f"{field_name} must be a decimal number sent as a string (e.g. '0.01'); got {value!r}."
        ) from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{field_name} must be a positive, finite decimal; got {value!r}.")
    return value


def _envelope_error(body: Any) -> str | None:
    """Return an `Error: …` string when a /sapi body carries a non-success `code`.

    The client already raises `BinanceEnvelopeError` for the `{"success": false}` shape;
    several /sapi endpoints instead answer HTTP 200 with a bare `{"code": …, "msg": …}`
    (accountSnapshot does, and any convert endpoint may), which has to be caught here.
    """
    if not isinstance(body, dict):
        return None
    code = body.get("code")
    if code in (None, 200, 0, "200", "0", "000000"):
        return None
    message = body.get("msg") or body.get("message") or "unknown error"
    return f"Error: {message} (code {code})"


def _ratio_line(data: dict[str, Any], from_asset: Any, to_asset: Any) -> str:
    ratio = data.get("ratio")
    inverse = data.get("inverseRatio")
    return (
        f"- **ratio**: 1 {from_asset} = {fmt_num(ratio)} {to_asset} "
        f"(inverseRatio: 1 {to_asset} = {fmt_num(inverse)} {from_asset})"
    )


def _format_convert_trade(item: dict[str, Any]) -> str:
    """One tradeFlow / orderStatus / open-limit-order row."""
    when = epoch_to_human(item.get("createTime"))
    from_amount = fmt_num(item.get("fromAmount"))
    to_amount = fmt_num(item.get("toAmount"))
    from_asset = item.get("fromAsset", "?")
    to_asset = item.get("toAsset", "?")
    status = item.get("orderStatus", "N/A")
    order_id = item.get("orderId", "N/A")
    ratio = fmt_num(item.get("ratio"))
    line = (
        f"- **{when}** — {from_amount} {from_asset} → {to_amount} {to_asset} "
        f"@ {ratio} — status **{status}** — orderId `{order_id}`"
    )
    expired = item.get("expiredTimestamp")
    if expired:
        line = f"{line} — expires {epoch_to_human(expired)}"
    return line


def _dedupe_by_order_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """De-duplicate by orderId, scoped to this one call's collected rows.

    An overlapping resume (or a window boundary re-read) is therefore safe: the same
    conversion never appears twice in one response.
    """
    seen: set[Any] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        order_id = item.get("orderId")
        if order_id is not None:
            if order_id in seen:
                continue
            seen.add(order_id)
        result.append(item)
    return result


def _create_time_ms(row: dict[str, Any]) -> int | None:
    """One row's `createTime` as an int, or None when it carries nothing usable."""
    value = row.get("createTime")
    if isinstance(value, int):
        return value
    text = str(value if value is not None else "").strip()
    digits = text[1:] if text.startswith("-") else text
    return int(text) if digits.isdigit() else None


def _sort_key_create_time(row: dict[str, Any]) -> int:
    """Sort key for the final ordering: a row with no usable `createTime` sorts last
    instead of raising and turning a successful walk into an unexpected-failure error."""
    ms = _create_time_ms(row)
    return ms if ms is not None else 0


def _oldest_create_time(rows: list[dict[str, Any]]) -> int | None:
    """Smallest `createTime` in a page, or None when no row carries a usable one."""
    times = [ms for ms in (_create_time_ms(row) for row in rows) if ms is not None]
    return min(times) if times else None


# -- enums ---------------------------------------------------------------------------


class ConvertWalletType(StrEnum):
    """Which wallet the conversion draws from / credits (quote + market-order side)."""

    SPOT = "SPOT"
    FUNDING = "FUNDING"


class LimitOrderWalletType(StrEnum):
    """Wallet selection for a convert LIMIT order — it may span both wallets."""

    SPOT = "SPOT"
    FUNDING = "FUNDING"
    SPOT_FUNDING = "SPOT_FUNDING"


class ConvertValidTime(StrEnum):
    """How long the quote stays acceptable before it expires."""

    TEN_SECONDS = "10s"
    THIRTY_SECONDS = "30s"
    ONE_MINUTE = "1m"
    TWO_MINUTES = "2m"


class ConvertSide(StrEnum):
    """Direction of a convert limit order, expressed against the base asset."""

    BUY = "BUY"
    SELL = "SELL"


class ConvertExpiredType(StrEnum):
    """How long a convert limit order rests before Binance expires it."""

    ONE_DAY = "1_D"
    THREE_DAYS = "3_D"
    SEVEN_DAYS = "7_D"
    THIRTY_DAYS = "30_D"


# -- input models --------------------------------------------------------------------


class _BaseInput(BaseModel):
    """Shared pydantic config for every input model in this module."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class ConvertPairsInput(_BaseInput):
    """Input for `binance_get_convert_pairs` (`GET /sapi/v1/convert/exchangeInfo`)."""

    from_asset: str | None = Field(
        default=None,
        min_length=1,
        description="Asset you would convert FROM, e.g. `BTC` (uppercased automatically). Binance: fromAsset. "
        "Pass at least one of from_asset / to_asset — this endpoint costs IP weight 3000 unfiltered.",
    )
    to_asset: str | None = Field(
        default=None,
        min_length=1,
        description="Asset you would convert TO, e.g. `USDT` (uppercased automatically). Binance: toAsset.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="markdown or json output.")

    @field_validator("from_asset", "to_asset")
    @classmethod
    def _check_asset(cls, value: str | None) -> str | None:
        return None if value is None else _normalize_asset(value)


class ConvertAssetInfoInput(_BaseInput):
    """Input for `binance_get_convert_asset_info` (`GET /sapi/v1/convert/assetInfo`)."""

    asset: str | None = Field(
        default=None,
        min_length=1,
        description="Optional client-side filter: show only this asset's precision row (e.g. `BTC`). "
        "The endpoint itself takes no filter — everything is returned and narrowed here.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="markdown or json output.")

    @field_validator("asset")
    @classmethod
    def _check_asset(cls, value: str | None) -> str | None:
        return None if value is None else _normalize_asset(value)


class ConvertQuoteInput(_BaseInput):
    """Input for `binance_get_convert_quote` (`POST /sapi/v1/convert/getQuote`)."""

    from_asset: str = Field(description="Asset to convert FROM, e.g. `BTC` (uppercased). Binance: fromAsset.")
    to_asset: str = Field(description="Asset to convert TO, e.g. `USDT` (uppercased). Binance: toAsset.")
    from_amount: str | None = Field(
        default=None,
        description="How much of from_asset to spend, as a decimal STRING (e.g. '0.01'). Exactly one of "
        "from_amount / to_amount. Binance: fromAmount.",
    )
    to_amount: str | None = Field(
        default=None,
        description="How much of to_asset you want to receive, as a decimal STRING (e.g. '500'). Exactly one "
        "of from_amount / to_amount. Binance: toAmount.",
    )
    wallet_type: ConvertWalletType | None = Field(
        default=None,
        description="Which wallet funds the conversion: SPOT (Binance's default) or FUNDING. Binance: walletType.",
    )
    valid_time: ConvertValidTime | None = Field(
        default=None,
        description="How long the quote stays acceptable: 10s (Binance's default), 30s, 1m or 2m. Binance: validTime.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="markdown or json output.")

    @field_validator("from_asset", "to_asset")
    @classmethod
    def _check_asset(cls, value: str) -> str:
        return _normalize_asset(value)

    @field_validator("from_amount", "to_amount")
    @classmethod
    def _check_amount(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _check_positive_decimal(value, info.field_name or "amount")

    @model_validator(mode="after")
    def _check_exactly_one_amount(self) -> Self:
        if (self.from_amount is None) == (self.to_amount is None):
            raise ValueError(
                "give exactly one of from_amount (spend this much of from_asset) or to_amount "
                "(receive this much of to_asset) — not both, not neither."
            )
        if self.from_asset == self.to_asset:
            raise ValueError("from_asset and to_asset must differ.")
        return self


class AcceptConvertQuoteInput(_BaseInput):
    """Input for `binance_accept_convert_quote` (`POST /sapi/v1/convert/acceptQuote`)."""

    quote_id: str = Field(
        min_length=1,
        description="The `quoteId` returned by `binance_get_convert_quote`. Accepting it EXECUTES the "
        "conversion at the quoted ratio. Binance: quoteId.",
    )


class ConvertOrderStatusInput(_BaseInput):
    """Input for `binance_get_convert_order_status` (`GET /sapi/v1/convert/orderStatus`)."""

    order_id: str | None = Field(
        default=None,
        min_length=1,
        description="Convert order id (from `binance_accept_convert_quote`). Exactly one of order_id / "
        "quote_id. Binance: orderId.",
    )
    quote_id: str | None = Field(
        default=None,
        min_length=1,
        description="Quote id (from `binance_get_convert_quote`). Exactly one of order_id / quote_id. "
        "Binance: quoteId.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="markdown or json output.")

    @model_validator(mode="after")
    def _check_exactly_one_id(self) -> Self:
        if (self.order_id is None) == (self.quote_id is None):
            raise ValueError("give exactly one of order_id or quote_id — not both, not neither.")
        return self


class ConvertHistoryInput(_BaseInput):
    """Input for `binance_get_convert_history` (`GET /sapi/v1/convert/tradeFlow`).

    Two mutually exclusive shapes over the same endpoint:

    - **single window** — `start_time` / `end_time` (or neither, for the last 30 days);
    - **walk** — `since` / `until` / `resume_before`, which slices the range into
      <=29-day windows, newest-first, within a `max_calls` budget.
    """

    start_time: int | str | None = Field(
        default=None,
        description="Single-window mode: window start — epoch ms, a >=12-digit epoch-ms string, or ISO-8601. "
        "Binance REQUIRES both bounds; give one and the other is filled locally with the 30-day rule. "
        "Binance: startTime.",
    )
    end_time: int | str | None = Field(
        default=None,
        description="Single-window mode: window end — epoch ms, a >=12-digit epoch-ms string, or ISO-8601. "
        "Span must be <= 30 days (validated here, not by a -1127 from Binance). Binance: endTime.",
    )
    since: int | str | None = Field(
        default=None,
        description="Walk mode: oldest instant to reach — epoch ms, epoch-ms string, or ISO-8601. Setting it "
        "switches this tool to the windowed walk and cannot be combined with start_time/end_time.",
    )
    until: int | str | None = Field(
        default=None,
        description="Walk mode: newest instant to start from (default: now).",
    )
    resume_before: int | str | None = Field(
        default=None,
        description="Walk mode: cursor from a previous truncated call — only fetch conversions from before "
        "this instant. Overrides `until`. Pass it alongside the same `since` used before.",
    )
    limit: int = Field(
        default=DEFAULT_PAGE_LIMIT,
        ge=1,
        le=MAX_PAGE_LIMIT,
        description=f"Rows per API call, 1-{MAX_PAGE_LIMIT} (this endpoint's own max; default "
        f"{DEFAULT_PAGE_LIMIT}). A full page means more rows exist in that window.",
    )
    max_calls: int = Field(
        default=DEFAULT_MAX_CALLS,
        ge=1,
        le=100,
        description=f"Budget of GET tradeFlow calls (UID weight 3000 each) to spend before stopping early "
        f"and returning a `resume_before` cursor. Also bounds the `moreData` re-asks inside a single window, "
        f"so one dense window can consume several calls. Default {DEFAULT_MAX_CALLS} = 72,000 UID weight.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="markdown or json output.")

    @field_validator("start_time", "end_time", "since", "until", "resume_before", mode="before")
    @classmethod
    def _normalize_time(cls, value: Any, info: ValidationInfo) -> int | None:
        try:
            return _to_ms(value)
        except ValueError:
            raise ValueError(f"{info.field_name} must be epoch ms or ISO-8601") from None

    @model_validator(mode="after")
    def _check_modes_not_mixed(self) -> Self:
        window_fields = [name for name in ("start_time", "end_time") if getattr(self, name) is not None]
        walk_fields = [name for name in ("since", "until", "resume_before") if getattr(self, name) is not None]
        if window_fields and walk_fields:
            raise ValueError(
                f"do not mix the single-window parameters ({', '.join(window_fields)}) with the walk "
                f"parameters ({', '.join(walk_fields)}): use start_time/end_time for one <=30-day window, "
                "or since/until/resume_before to walk a wider range."
            )
        return self


class PlaceConvertLimitOrderInput(_BaseInput):
    """Input for `binance_place_convert_limit_order` (`POST /sapi/v1/convert/limit/placeOrder`)."""

    base_asset: str = Field(description="Base asset of the pair, e.g. `BTC` (uppercased). Binance: baseAsset.")
    quote_asset: str = Field(description="Quote asset of the pair, e.g. `USDT` (uppercased). Binance: quoteAsset.")
    limit_price: str = Field(
        description="Trigger price expressed as baseAsset/quoteAsset, as a decimal STRING (e.g. '65000'). "
        "Binance: limitPrice.",
    )
    side: ConvertSide = Field(description="BUY or SELL the base asset. Binance: side.")
    expired_type: ConvertExpiredType = Field(
        description="How long the order rests before Binance expires it: 1_D, 3_D, 7_D or 30_D. Binance: expiredType.",
    )
    base_amount: str | None = Field(
        default=None,
        description="Amount of base asset, as a decimal STRING. Exactly one of base_amount / quote_amount. "
        "Binance: baseAmount.",
    )
    quote_amount: str | None = Field(
        default=None,
        description="Amount of quote asset, as a decimal STRING. Exactly one of base_amount / quote_amount. "
        "Binance: quoteAmount.",
    )
    wallet_type: LimitOrderWalletType | None = Field(
        default=None,
        description="Which wallet funds the order: SPOT (Binance's default), FUNDING or SPOT_FUNDING. "
        "Binance: walletType.",
    )

    @field_validator("base_asset", "quote_asset")
    @classmethod
    def _check_asset(cls, value: str) -> str:
        return _normalize_asset(value)

    @field_validator("limit_price", "base_amount", "quote_amount")
    @classmethod
    def _check_amount(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _check_positive_decimal(value, info.field_name or "amount")

    @model_validator(mode="after")
    def _check_exactly_one_amount(self) -> Self:
        if (self.base_amount is None) == (self.quote_amount is None):
            raise ValueError("give exactly one of base_amount or quote_amount — not both, not neither.")
        if self.base_asset == self.quote_asset:
            raise ValueError("base_asset and quote_asset must differ.")
        return self


class CancelConvertLimitOrderInput(_BaseInput):
    """Input for `binance_cancel_convert_limit_order` (`POST /sapi/v1/convert/limit/cancelOrder`)."""

    order_id: str = Field(
        min_length=1,
        description="The convert limit order to cancel, as returned by `binance_place_convert_limit_order` or "
        "`binance_get_convert_open_limit_orders`. Binance: orderId.",
    )


class ConvertOpenLimitOrdersInput(_BaseInput):
    """Input for `binance_get_convert_open_limit_orders` (`GET .../limit/queryOpenOrders`)."""

    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="markdown or json output.")


# -- the tradeFlow walk ---------------------------------------------------------------


@dataclass
class ConvertWalkResult:
    """Outcome of one `_walk_trade_flow` call.

    `resume_before` is always the `endTime` the NEXT request would have used — the loop's
    own boundary variable at exit, never inferred afterwards from which rows happened to
    come back. It is emitted whenever the walk stopped before reaching `since`, whatever
    the reason (budget or error), so a caller can always continue.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    calls_used: int = 0
    resume_before: int | None = None
    stop_reason: Literal["budget", "error"] | None = None
    stop_error: Exception | None = None
    possibly_incomplete: bool = False
    no_progress: bool = False


async def _fetch_trade_flow(
    client: BinanceClient, *, start_ms: int, end_ms: int, limit: int
) -> tuple[list[dict[str, Any]], bool]:
    """One `GET /sapi/v1/convert/tradeFlow` page → (rows, moreData)."""
    resp = await client.request(
        "GET",
        CONVERT_TRADE_FLOW_PATH,
        params={"startTime": start_ms, "endTime": end_ms, "limit": limit},
        auth="signed",
    )
    body = resp.json()
    envelope = _envelope_error(body)
    if envelope is not None:
        raise RuntimeError(envelope.removeprefix("Error: "))
    rows = body.get("list") or []
    return list(rows), bool(body.get("moreData"))


async def _walk_trade_flow(
    client: BinanceClient,
    *,
    since_ms: int,
    until_ms: int,
    limit: int,
    max_calls: int,
    window_ms: int = WALK_WINDOW_MS,
) -> ConvertWalkResult:
    """Walk [since_ms, until_ms] newest-first in <=`window_ms` windows, narrowing dense ones.

    `window_ms` is 29 days for the walk and the full 30-day cap for single-window mode, so
    a caller's own <=30-day window is sent to Binance exactly as asked (one call, one UID
    charge of 3000) instead of being split in two.

    Binance offers **no cursor parameter** on tradeFlow: the only continuation it gives is
    the `moreData` flag, so a window that still has rows is re-asked with
    `endTime = min(createTime)` — INCLUSIVE, because an exclusive `- 1` would drop every row
    tied on that instant but cut off by `limit` — until `moreData` is false or the window is
    exhausted. Tied rows re-read that way are deduped by orderId. When the oldest row is
    already at the window's end instant, `limit` rows share one millisecond and no narrower
    `endTime` exists: that is `possibly_incomplete`, not a silent drop.

    The budget is checked BEFORE every request, and a request that fails still counts
    against it, so `calls_used` is always the real number of requests made. On any
    failure the rows already collected are returned alongside the cursor — never just the
    error. `no_progress` is set when the cursor did not advance past the original upper
    bound (the very first request failed), because resuming from it would repeat the
    identical call.
    """
    result = ConvertWalkResult()
    next_end = until_ms

    while next_end >= since_ms:
        window_start = max(next_end - window_ms, since_ms)
        page_end = next_end
        while True:
            if result.calls_used >= max_calls:
                result.stop_reason = "budget"
                result.no_progress = page_end >= until_ms
                # A cursor that does not advance past the original upper bound would just
                # reproduce the identical UID-3000 call: emit none at all.
                result.resume_before = None if result.no_progress else page_end
                return result
            try:
                rows, more_data = await _fetch_trade_flow(client, start_ms=window_start, end_ms=page_end, limit=limit)
            except Exception as exc:  # rows collected so far must survive any failure (walk rule d)
                result.calls_used += 1  # the failed request still spent one call of the budget
                result.stop_reason = "error"
                result.stop_error = exc
                result.no_progress = page_end >= until_ms
                result.resume_before = None if result.no_progress else page_end
                return result
            result.calls_used += 1
            result.rows.extend(rows)
            if not more_data:
                break
            oldest = _oldest_create_time(rows)
            if oldest is None:
                # `moreData` with nothing to narrow against: Binance says there is more in
                # this window but gave no usable createTime, so those rows cannot be reached.
                result.possibly_incomplete = True
                break
            if oldest >= page_end:
                # The page's oldest row already sits at the window's end instant, so >= `limit`
                # conversions share that single millisecond: narrowing cannot advance past it
                # and the remainder of that instant is unreachable through this endpoint.
                result.possibly_incomplete = True
                break
            if oldest < window_start:
                break
            # INCLUSIVE narrowing. `oldest - 1` would silently drop every row that shares the
            # page's oldest createTime but fell past the page cut (a tie straddling the limit);
            # re-reading the tied rows instead is free, because they are deduped by orderId.
            page_end = oldest
        next_end = window_start - 1

    return result


# -- tools ---------------------------------------------------------------------------


@mcp.tool(
    name="binance_get_convert_pairs",
    annotations=ToolAnnotations(
        title="Binance Convert Pairs",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_convert_pairs(params: ConvertPairsInput) -> str:
    """List the convertible asset pairs and their per-pair minimum/maximum amounts.

    Calls `GET /sapi/v1/convert/exchangeInfo` — **unauthenticated** (no key needed) but
    **IP weight 3000** of a 12,000/min budget, so four unfiltered calls exhaust a minute.
    The pair list barely changes: cache the answer and pass `from_asset`/`to_asset` to
    keep the response small.

    When to Use:
    - Before quoting, to check a pair is convertible at all and that the amount you plan
      to convert sits between `fromAssetMinAmount` and `fromAssetMaxAmount`.
    - To find what a given asset can be converted into (`from_asset="BTC"`).

    When NOT to Use:
    - To get a price — that is `binance_get_convert_quote` (a quote reserves a ratio).
    - To read spot symbol filters — that is `binance_get_exchange_info`; convert pairs and
      spot trading pairs are different lists with different rules.

    Returns:
    One row per pair: from → to plus the min/max amount on both legs. Markdown display is
    capped at 100 pairs with a note to narrow the filter; `response_format="json"` carries
    every row returned.

    Examples:
    params = {"from_asset": "BTC"}
    params = {"from_asset": "BTC", "to_asset": "USDT"}

    Error Handling:
    429 means the IP weight budget is gone — wait out `Retry-After` rather than retrying;
    this single endpoint is 3000 weight per call.
    """
    try:
        query = {
            key: value
            for key, value in {"fromAsset": params.from_asset, "toAsset": params.to_asset}.items()
            if value is not None
        }
        client = get_client()
        resp = await client.request("GET", CONVERT_EXCHANGE_INFO_PATH, params=query, auth="none")
        body = resp.json()
        if (envelope := _envelope_error(body)) is not None:
            return envelope
        pairs: list[dict[str, Any]] = list(body) if isinstance(body, list) else []

        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json({"title": "Binance Convert Pairs", "count": len(pairs), "items": pairs}))

        display = pairs[:MAX_PAIR_ROWS]
        lines = [
            "# Binance Convert Pairs",
            "",
            f"Showing **{len(display):,}** of **{len(pairs):,}** pair(s). "
            "This endpoint costs IP weight 3000 — cache the result.",
        ]
        if len(pairs) > MAX_PAIR_ROWS:
            lines.append(f"_More than {MAX_PAIR_ROWS} pairs — pass `from_asset` and/or `to_asset` to narrow._")
        lines.append("")
        if display:
            for pair in display:
                lines.append(
                    f"- **{pair.get('fromAsset', '?')} → {pair.get('toAsset', '?')}** — from "
                    f"{fmt_num(pair.get('fromAssetMinAmount'))}–{fmt_num(pair.get('fromAssetMaxAmount'))} "
                    f"{pair.get('fromAsset', '')}, to {fmt_num(pair.get('toAssetMinAmount'))}–"
                    f"{fmt_num(pair.get('toAssetMaxAmount'))} {pair.get('toAsset', '')}"
                )
        else:
            lines.append("_No convertible pairs for that filter._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_convert_asset_info",
    annotations=ToolAnnotations(
        title="Binance Convert Asset Precision",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_convert_asset_info(params: ConvertAssetInfoInput) -> str:
    """Show the decimal precision (`fraction`) Convert accepts for each asset.

    Calls `GET /sapi/v1/convert/assetInfo` (SIGNED, IP weight 100). `fraction` is the
    number of decimal places an amount may carry for that asset on the convert rail —
    sending more precision than this is what `-1111` rejects.

    When to Use:
    - Before `binance_get_convert_quote`, to round `from_amount`/`to_amount` correctly.
    - When a quote was rejected for precision.

    When NOT to Use:
    - To find which pairs exist or their min/max sizes — that is
      `binance_get_convert_pairs`.

    Returns:
    One row per asset: asset and its accepted decimal places, filtered client-side when
    `asset` is given. Markdown display is capped at 100 rows.

    Examples:
    params = {}
    params = {"asset": "BTC"}

    Error Handling:
    -2015 means the key lacks permission or this machine's IP is not on the key's
    allowlist; the spot testnet has no `/sapi` endpoints at all (404).
    """
    try:
        client = get_client()
        resp = await client.request("GET", CONVERT_ASSET_INFO_PATH, auth="signed")
        body = resp.json()
        if (envelope := _envelope_error(body)) is not None:
            return envelope
        assets: list[dict[str, Any]] = list(body) if isinstance(body, list) else []
        if params.asset is not None:
            assets = [row for row in assets if str(row.get("asset", "")).upper() == params.asset]

        if params.response_format is ResponseFormat.JSON:
            return clip_response(
                to_json({"title": "Binance Convert Asset Info", "count": len(assets), "items": assets})
            )

        display = assets[:MAX_ASSET_ROWS]
        lines = [
            "# Binance Convert Asset Precision",
            "",
            f"Showing **{len(display):,}** of **{len(assets):,}** asset(s).",
            "",
        ]
        if display:
            lines.extend(
                f"- **{row.get('asset', '?')}** — {row.get('fraction', 'N/A')} decimal place(s)" for row in display
            )
        else:
            lines.append("_No assets for that filter._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_convert_quote",
    annotations=ToolAnnotations(
        title="Binance Get Convert Quote",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def binance_get_convert_quote(params: ConvertQuoteInput) -> str:
    """Request a convert quote: a reserved ratio, valid for 10 s to 2 minutes.

    Calls `POST /sapi/v1/convert/getQuote` (SIGNED, **UID weight 200**). This moves no
    funds — it is on the client's POST-read allowlist, so it works with the trading
    kill-switch off — but it is not free either: a quote is a short-lived reservation, so
    do not poll it in a loop.

    Nothing is converted until the quote is accepted with `binance_accept_convert_quote`
    before `validTimestamp`. After that instant the quoteId is void and a new quote is
    needed.

    When to Use:
    - To price a conversion (what you would receive, and at what ratio) before deciding.
    - As the first half of every conversion: quote → human approves → accept.

    When NOT to Use:
    - For an indicative market price — `binance_get_ticker_price` is free and does not
      reserve anything.
    - To convert at a price that is not currently available — place a convert limit order
      with `binance_place_convert_limit_order` instead.

    Returns:
    The quoteId, the ratio and inverseRatio, both amounts, and the expiry instant
    (`validTimestamp`) rendered as UTC, plus the instruction to accept it before it
    expires.

    Examples:
    params = {"from_asset": "BTC", "to_asset": "USDT", "from_amount": "0.01"}
    params = {"from_asset": "USDT", "to_asset": "BTC", "to_amount": "0.5", "valid_time": "1m"}

    Error Handling:
    -1111 means the amount carries more decimals than the asset's `fraction` (see
    `binance_get_convert_asset_info`); a rejection about limits means the amount is
    outside the pair's min/max (see `binance_get_convert_pairs`); -2015 means the key
    lacks permission or the IP is not allowlisted.
    """
    try:
        query = {
            key: value
            for key, value in {
                "fromAsset": params.from_asset,
                "toAsset": params.to_asset,
                "fromAmount": params.from_amount,
                "toAmount": params.to_amount,
                "walletType": params.wallet_type.value if params.wallet_type else None,
                "validTime": params.valid_time.value if params.valid_time else None,
            }.items()
            if value is not None
        }
        client = get_client()
        resp = await client.request("POST", CONVERT_GET_QUOTE_PATH, params=query, auth="signed")
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope

        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))

        quote_id = data.get("quoteId", "N/A")
        lines = [
            f"# Convert quote — {params.from_asset} → {params.to_asset}",
            "",
            f"- **quoteId**: `{quote_id}`",
            _ratio_line(data, params.from_asset, params.to_asset),
            f"- **fromAmount**: {fmt_num(data.get('fromAmount'))} {params.from_asset}"
            + (f" (from the {params.wallet_type.value} wallet)" if params.wallet_type else ""),
            f"- **toAmount**: {fmt_num(data.get('toAmount'))} {params.to_asset}",
            f"- **valid until**: {epoch_to_human(data.get('validTimestamp'))} "
            f"(validTimestamp {data.get('validTimestamp', 'N/A')})",
            "",
            f'_Nothing has been converted. Accept with `binance_accept_convert_quote` (quote_id="{quote_id}") '
            "before it expires — after `validTimestamp` the quote is void and you must request a new one._",
        ]
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_accept_convert_quote",
    annotations=ToolAnnotations(
        title="Binance Accept Convert Quote",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def binance_accept_convert_quote(params: AcceptConvertQuoteInput) -> str:
    """Accept a convert quote and EXECUTE the conversion. This moves real funds.

    Calls `POST /sapi/v1/convert/acceptQuote` (SIGNED, **UID weight 500**). The
    conversion is **irreversible**: converting back needs a new quote at whatever ratio
    the market offers then, so the round trip costs the spread twice.

    **Kill-switch.** Refused with `Error: … trading is disabled …` unless the server runs
    with `BINANCE_ALLOW_TRADING=1`. The gate lives in the HTTP client, so no tool can
    bypass it. If you see that error, the operator has deliberately put the server in
    read-only mode — report it, do not try to work around it.

    Always price the conversion with `binance_get_convert_quote` first (it works even
    with the kill-switch off) and have the human approve that exact quoteId and amount.

    When to Use:
    - Immediately after a human approved the ratio in a fresh quote, before it expires.

    When NOT to Use:
    - To "see what would happen" — that is `binance_get_convert_quote`.
    - To convert at a price that is not on offer now — use
      `binance_place_convert_limit_order`.

    Returns:
    A confirmation echoing exactly what Binance returned — orderId, createTime and
    `orderStatus` verbatim (PROCESS / ACCEPT_SUCCESS / SUCCESS / FAIL). Only **SUCCESS**
    means the assets were exchanged; the other statuses are reported as-is with the next
    step, never paraphrased as "converted".

    Examples:
    params = {"quote_id": "12415572564"}

    Error Handling:
    An expired or already-used quoteId is rejected by Binance — request a new quote
    rather than retrying this one. A 5xx or a timeout means the execution status is
    **UNKNOWN**: check with `binance_get_convert_order_status` (by quote_id) before
    accepting anything again — never blind-retry a conversion.
    """
    try:
        client = get_client()
        resp = await client.request(
            "POST", CONVERT_ACCEPT_QUOTE_PATH, params={"quoteId": params.quote_id}, auth="signed"
        )
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope

        status = data.get("orderStatus")
        order_id = data.get("orderId", "N/A")
        lines = [
            "# Convert quote accepted",
            "",
            f"- **quoteId**: `{params.quote_id}`",
            f"- **orderId**: `{order_id}`",
            f"- **createTime**: {epoch_to_human(data.get('createTime'))} ({data.get('createTime', 'N/A')})",
            f"- **orderStatus**: **{status if status is not None else 'not reported'}**",
            "",
        ]
        note = _ORDER_STATUS_NOTES.get(str(status)) if status is not None else None
        if note is not None:
            lines.append(f"_{note}_")
        elif status is None:
            lines.append(
                "_Binance returned no `orderStatus` for this acceptance, so whether the conversion completed "
                "is unknown from this response — check with `binance_get_convert_order_status`._"
            )
        else:
            lines.append(
                f"_Binance reported status **{status}**, which this server does not recognise — check with "
                "`binance_get_convert_order_status` before assuming the conversion completed._"
            )
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_convert_order_status",
    annotations=ToolAnnotations(
        title="Binance Convert Order Status",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_convert_order_status(params: ConvertOrderStatusInput) -> str:
    """Check one conversion's status by orderId or by the quoteId it came from.

    Calls `GET /sapi/v1/convert/orderStatus` (SIGNED, **UID weight 100**). Exactly one of
    `order_id` / `quote_id` — Binance rejects both together.

    When to Use:
    - After `binance_accept_convert_quote` returned PROCESS or ACCEPT_SUCCESS, to find out
      whether it settled.
    - After a timeout or 5xx on an acceptance, to learn whether the conversion happened
      before retrying anything.

    When NOT to Use:
    - For a list of past conversions — use `binance_get_convert_history`.
    - For resting limit orders — use `binance_get_convert_open_limit_orders`.

    Returns:
    The conversion's assets, amounts, ratio, status and creation time, with the status
    echoed verbatim.

    Examples:
    params = {"order_id": "933256278426274426"}
    params = {"quote_id": "12415572564"}

    Error Handling:
    -2015 means the key lacks permission or the IP is not allowlisted; an unknown id is
    rejected by Binance rather than returning an empty result.
    """
    try:
        query = {
            key: value
            for key, value in {"orderId": params.order_id, "quoteId": params.quote_id}.items()
            if value is not None
        }
        client = get_client()
        resp = await client.request("GET", CONVERT_ORDER_STATUS_PATH, params=query, auth="signed")
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope

        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))

        status = data.get("orderStatus")
        lines = [
            "# Convert order status",
            "",
            f"- **orderId**: `{data.get('orderId', 'N/A')}`",
            f"- **status**: **{status if status is not None else 'not reported'}**",
            f"- **from**: {fmt_num(data.get('fromAmount'))} {data.get('fromAsset', '?')}",
            f"- **to**: {fmt_num(data.get('toAmount'))} {data.get('toAsset', '?')}",
            _ratio_line(data, data.get("fromAsset", "?"), data.get("toAsset", "?")),
            f"- **createTime**: {epoch_to_human(data.get('createTime'))}",
        ]
        note = _ORDER_STATUS_NOTES.get(str(status)) if status is not None else None
        if note is not None:
            lines.extend(["", f"_{note}_"])
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


def _render_convert_history(
    rows: list[dict[str, Any]],
    *,
    result: ConvertWalkResult,
    since_ms: int,
    until_ms: int,
    max_calls: int,
    walking: bool,
    fmt: ResponseFormat,
) -> str:
    title = "Binance Convert History" + (" (Windowed Walk)" if walking else "")
    if fmt is ResponseFormat.JSON:
        return clip_response(
            to_json(
                {
                    "title": title,
                    "mode": "walk" if walking else "window",
                    "since": since_ms,
                    "until": until_ms,
                    "calls_used": result.calls_used,
                    "count": len(rows),
                    "resume_before": result.resume_before,
                    "no_progress": result.no_progress,
                    "possibly_incomplete": result.possibly_incomplete,
                    "stop_reason": result.stop_reason,
                    "stop_error": handle_api_error(result.stop_error) if result.stop_error is not None else None,
                    "items": rows,
                }
            )
        )

    lines = [
        f"# {title}",
        "",
        f"Window: **{epoch_to_human(since_ms)}** → **{epoch_to_human(until_ms)}** "
        f"({result.calls_used} API call(s) spent, UID weight 3000 each).",
        f"Found **{len(rows):,}** conversion(s) (deduped by orderId within this call, newest first).",
    ]
    if result.stop_reason == "error":
        assert result.stop_error is not None
        lines.append(f"⚠️ Stopped early — {handle_api_error(result.stop_error)}")
    elif result.stop_reason == "budget":
        lines.append(f"⚠️ Stopped early — the `max_calls` budget ({max_calls}) ran out before reaching the start.")
    if result.no_progress:
        lines.append(
            "No cursor is offered: the walk never got past its upper bound, so any resume would repeat the "
            "identical call. Raise `max_calls`, narrow the range, or retry once the underlying failure is fixed."
        )
    elif result.resume_before is not None:
        if walking:
            lines.append(
                f"Rows before {epoch_to_human(result.resume_before)} were not fetched; continue with "
                f"`since={since_ms}, resume_before={result.resume_before}`."
            )
        else:
            lines.append(
                f"Rows before {epoch_to_human(result.resume_before)} were not fetched; continue with "
                f"`start_time={since_ms}, end_time={result.resume_before}` (or switch to walk mode with "
                f"`since={since_ms}, resume_before={result.resume_before}`)."
            )
    if result.possibly_incomplete:
        lines.append(
            "⚠️ Binance flagged `moreData` on a page this endpoint offers no way to continue past — either its "
            "rows carried no usable `createTime`, or `limit` rows share one millisecond, which no narrower "
            "`endTime` can split. The remainder of that instant was not fetched; a larger `limit` may reach it."
        )
    if len(rows) > MAX_DISPLAY_ROWS:
        lines.append(
            f'_Showing the {MAX_DISPLAY_ROWS} most recent of {len(rows):,} — use response_format="json" '
            "for the full set._"
        )
    lines.append("")
    if rows:
        lines.extend(_format_convert_trade(row) for row in rows[:MAX_DISPLAY_ROWS])
    else:
        lines.append("_No conversions in this window._")
    return clip_response("\n".join(lines))


@mcp.tool(
    name="binance_get_convert_history",
    annotations=ToolAnnotations(
        title="Binance Convert History",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_convert_history(params: ConvertHistoryInput) -> str:
    """List past conversions, either for one <=30-day window or across a walked range.

    Calls `GET /sapi/v1/convert/tradeFlow` (SIGNED, **UID weight 3000 per call** of a
    180,000/min budget — 60 calls a minute at most, and the default budget here spends
    72,000 of it).

    Binance **requires both `startTime` and `endTime`** and caps the span at 30 days, and
    the endpoint has **no cursor/offset/page parameter of any kind**. The only
    continuation it offers is the `moreData` flag: when it is true, the same window is
    re-asked with `endTime = min(createTime)` (inclusive — an exclusive `- 1` would drop
    rows tied on that instant but cut off by `limit`; the re-read duplicates are deduped by
    orderId) until it comes back false. Both modes below automate that.

    - **Single window** (default): `start_time` / `end_time`. Give one and the other is
      filled locally by the 30-day rule; give neither and the last 30 days are used.
    - **Walk**: `since` (plus optional `until` / `resume_before`) slices the range into
      <=29-day windows, newest-first, until `since` is reached or `max_calls` runs out.

    The two sets are mutually exclusive — mixing them is rejected locally.

    When to Use:
    - To reconcile conversions for a period, or to find the orderId of a past conversion.
    - To pull more than 30 days of history without hand-rolling the windowing (walk mode).

    When NOT to Use:
    - To check one conversion you just made — `binance_get_convert_order_status` is UID 100
      against this endpoint's 3000.
    - For resting limit orders, which are not conversions yet —
      `binance_get_convert_open_limit_orders`.

    Returns:
    Conversions newest-first (time, from → to amounts, ratio, status, orderId), deduped by
    orderId within the call, the number of API calls spent, and — when the walk stopped
    early — a `resume_before` cursor marking the boundary of the next unfetched range.
    A failure mid-walk returns the rows collected so far **plus** that cursor and the
    error, never the error alone. Markdown display caps at 50 rows;
    `response_format="json"` carries every fetched row.

    Pagination/Windows:
    `start_time`/`end_time`/`since`/`until`/`resume_before` all accept epoch ms, a
    >=12-digit epoch-ms string, or ISO-8601. A single-window span over 30 days is rejected
    here with a clear message rather than sent on to become a Binance error. `limit` is
    <= 1000 (the endpoint's own max). `resume_before` is always the `endTime` the next
    request would have used, so resuming never leaves a gap; overlapping rows are deduped.

    Examples:
    params = {}  # the last 30 days, one window
    params = {"start_time": "2026-08-01", "end_time": "2026-08-20"}
    params = {"since": "2026-01-01", "max_calls": 12}
    params = {"since": "2026-01-01", "resume_before": 1756000000000}

    Error Handling:
    -1127 means the span exceeded Binance's cap (this tool validates first, so it should
    not appear); 429 on /sapi means the UID weight budget is gone — lower `max_calls` and
    wait; -2015 means the key lacks permission or the IP is not allowlisted.
    """
    try:
        walking = params.since is not None or params.until is not None or params.resume_before is not None
        now_ms = _now_ms()

        if walking:
            # `resume_before` beats `until`: it is the cursor of an interrupted walk.
            resume_ms = _as_ms(params.resume_before)
            until_input = _as_ms(params.until)
            if resume_ms is not None:
                until_ms = resume_ms
            elif until_input is not None:
                until_ms = until_input
            else:
                until_ms = now_ms
            since_ms = _as_ms(params.since)
            if since_ms is None:
                since_ms = until_ms - MAX_CONVERT_WINDOW_MS
            if since_ms >= until_ms:
                return (
                    "# Binance Convert History (Windowed Walk)\n\n"
                    f"_No window to walk: since ({epoch_to_human(since_ms)}) is not before the upper bound "
                    f"({epoch_to_human(until_ms)})._"
                )
        else:
            start_ms = _as_ms(params.start_time)
            end_ms = _as_ms(params.end_time)
            # Binance requires BOTH bounds: fill whichever is missing with the 30-day rule.
            if start_ms is None and end_ms is None:
                end_ms, start_ms = now_ms, now_ms - MAX_CONVERT_WINDOW_MS
            elif end_ms is None:
                assert start_ms is not None
                end_ms = min(start_ms + MAX_CONVERT_WINDOW_MS, now_ms)
            elif start_ms is None:
                start_ms = end_ms - MAX_CONVERT_WINDOW_MS
            assert start_ms is not None and end_ms is not None
            if start_ms > end_ms:
                return "Error: start_time must be before end_time."
            span = end_ms - start_ms
            if span > MAX_CONVERT_WINDOW_MS:
                return (
                    "Error: the startTime/endTime span for GET /sapi/v1/convert/tradeFlow cannot exceed 30 days "
                    f"(got {span / 86_400_000:.1f} days). Narrow the window, or use walk mode "
                    "(`since`/`until`) which slices a wider range for you."
                )
            since_ms, until_ms = start_ms, end_ms

        client = get_client()
        result = await _walk_trade_flow(
            client,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=params.limit,
            max_calls=params.max_calls,
            # Single-window mode asks Binance for exactly the window it was given.
            window_ms=WALK_WINDOW_MS if walking else MAX_CONVERT_WINDOW_MS,
        )
        rows = _dedupe_by_order_id(result.rows)
        rows.sort(key=_sort_key_create_time, reverse=True)

        return _render_convert_history(
            rows,
            result=result,
            since_ms=since_ms,
            until_ms=until_ms,
            max_calls=params.max_calls,
            walking=walking,
            fmt=params.response_format,
        )
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_place_convert_limit_order",
    annotations=ToolAnnotations(
        title="Binance Place Convert Limit Order",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def binance_place_convert_limit_order(params: PlaceConvertLimitOrderInput) -> str:
    """Place a convert LIMIT order: convert automatically if the ratio is reached.

    Calls `POST /sapi/v1/convert/limit/placeOrder` (SIGNED, **UID weight 500**). The
    order rests until `limit_price` is reached or `expired_type` (1_D / 3_D / 7_D / 30_D)
    expires it. When it triggers it **spends real funds**, without asking again.

    **Kill-switch.** Refused with `Error: … trading is disabled …` unless the server runs
    with `BINANCE_ALLOW_TRADING=1`. The gate lives in the HTTP client, so no tool can
    bypass it.

    Check the pair's limits with `binance_get_convert_pairs` and the amount precision with
    `binance_get_convert_asset_info` first; there is no dry-run for this endpoint.

    When to Use:
    - To convert at a ratio the market is not offering right now, after a human approved
      that price and size.

    When NOT to Use:
    - To convert at the current ratio — quote it with `binance_get_convert_quote` and
      accept it, which is immediate and shows you the exact ratio first.
    - For a spot LIMIT order on a trading pair — that is `binance_place_order`, a
      different book with different fees.

    Returns:
    A confirmation echoing exactly what Binance returned — orderId and `status` verbatim.
    A resting order is not a conversion: nothing has been exchanged until it triggers, and
    this confirmation never says otherwise.

    Examples:
    params = {"base_asset": "BTC", "quote_asset": "USDT", "limit_price": "50000",
              "side": "BUY", "expired_type": "7_D", "quote_amount": "500"}

    Error Handling:
    A rejection about limits means the amount is outside the pair's min/max
    (`binance_get_convert_pairs`); -1111 means too many decimals for the asset
    (`binance_get_convert_asset_info`); a 5xx or timeout means the order's status is
    **UNKNOWN** — check `binance_get_convert_open_limit_orders` before placing it again.
    """
    try:
        query = {
            key: value
            for key, value in {
                "baseAsset": params.base_asset,
                "quoteAsset": params.quote_asset,
                "limitPrice": params.limit_price,
                "side": params.side.value,
                "expiredType": params.expired_type.value,
                "baseAmount": params.base_amount,
                "quoteAmount": params.quote_amount,
                "walletType": params.wallet_type.value if params.wallet_type else None,
            }.items()
            if value is not None
        }
        client = get_client()
        resp = await client.request("POST", CONVERT_LIMIT_PLACE_PATH, params=query, auth="signed")
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope

        amount = (
            f"{fmt_num(params.base_amount)} {params.base_asset}"
            if params.base_amount is not None
            else f"{fmt_num(params.quote_amount)} {params.quote_asset}"
        )
        status = data.get("status")
        lines = [
            f"# Convert limit order placed — {params.side.value} {params.base_asset}/{params.quote_asset}",
            "",
            f"- **orderId**: `{data.get('orderId', 'N/A')}`",
            f"- **status**: **{status if status is not None else 'not reported'}**",
            f"- **limitPrice**: {fmt_num(params.limit_price)} {params.quote_asset} per {params.base_asset}",
            f"- **amount**: {amount}",
            f"- **expires**: {params.expired_type.value.replace('_D', ' day(s)')}",
            "",
            "_Nothing has been converted yet: the order rests until the limit price is reached or it expires. "
            "Track it with `binance_get_convert_open_limit_orders`; cancel it with "
            "`binance_cancel_convert_limit_order`._",
        ]
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_cancel_convert_limit_order",
    annotations=ToolAnnotations(
        title="Binance Cancel Convert Limit Order",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_cancel_convert_limit_order(params: CancelConvertLimitOrderInput) -> str:
    """Cancel a resting convert limit order.

    Calls `POST /sapi/v1/convert/limit/cancelOrder` (SIGNED, **UID weight 200**).

    **Kill-switch.** Refused with `Error: … trading is disabled …` unless the server runs
    with `BINANCE_ALLOW_TRADING=1` — cancellation is a signed non-GET like any other
    state change, so the same gate applies.

    When to Use:
    - To pull a convert limit order that no longer reflects the plan, before it triggers.

    When NOT to Use:
    - For a spot order — that is `binance_cancel_order`.
    - To undo a completed conversion: there is no such thing. Converting back needs a new
      quote at the current ratio.

    Returns:
    A confirmation echoing exactly what Binance returned — orderId and `status` verbatim.

    Examples:
    params = {"order_id": "1603680255057330400"}

    Error Handling:
    An already-filled, already-cancelled or unknown orderId is rejected by Binance — check
    `binance_get_convert_open_limit_orders` for what is actually resting. A 5xx or a timeout
    means the cancellation status is **UNKNOWN**: check
    `binance_get_convert_open_limit_orders` before cancelling again, because the order may
    already be gone.
    """
    try:
        client = get_client()
        resp = await client.request(
            "POST", CONVERT_LIMIT_CANCEL_PATH, params={"orderId": params.order_id}, auth="signed"
        )
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope

        status = data.get("status")
        lines = [
            "# Convert limit order cancellation",
            "",
            f"- **orderId**: `{data.get('orderId', params.order_id)}`",
            f"- **status**: **{status if status is not None else 'not reported'}**",
            "",
            f"_Binance reported status **{status if status is not None else 'not reported'}** for this "
            "cancellation; confirm with `binance_get_convert_open_limit_orders` if it matters._",
        ]
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_convert_open_limit_orders",
    annotations=ToolAnnotations(
        title="Binance Open Convert Limit Orders",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_convert_open_limit_orders(params: ConvertOpenLimitOrdersInput) -> str:
    """List the convert limit orders currently resting on the account.

    Calls `GET /sapi/v1/convert/limit/queryOpenOrders` (SIGNED, **UID weight 3000** of a
    180,000/min budget) — cross-asset, no filters, so poll it sparingly.

    When to Use:
    - To see what convert limit orders are live, with their expiry instants.
    - Before placing another one, to avoid stacking duplicates.
    - After a timeout on a placement, to learn whether the order actually rested.

    When NOT to Use:
    - For completed conversions — `binance_get_convert_history`.
    - For spot open orders — `binance_get_open_orders`.

    Returns:
    One row per resting order: creation time, from → to amounts, ratio, status, orderId
    and the expiry instant. Markdown display caps at 50 rows.

    Examples:
    params = {}

    Error Handling:
    -2015 means the key lacks permission or the IP is not allowlisted; 429 on /sapi means
    the UID budget is gone — this endpoint alone is 3000 per call.
    """
    try:
        client = get_client()
        resp = await client.request("GET", CONVERT_LIMIT_OPEN_ORDERS_PATH, auth="signed")
        body = resp.json()
        if (envelope := _envelope_error(body)) is not None:
            return envelope
        rows: list[dict[str, Any]] = list(body.get("list") or []) if isinstance(body, dict) else []

        if params.response_format is ResponseFormat.JSON:
            return clip_response(
                to_json({"title": "Binance Open Convert Limit Orders", "count": len(rows), "items": rows})
            )

        lines = [
            "# Binance Open Convert Limit Orders",
            "",
            f"**{len(rows):,}** resting order(s).",
        ]
        if len(rows) > MAX_DISPLAY_ROWS:
            lines.append(f'_Showing the first {MAX_DISPLAY_ROWS} — use response_format="json" for the full set._')
        lines.append("")
        if rows:
            lines.extend(_format_convert_trade(row) for row in rows[:MAX_DISPLAY_ROWS])
        else:
            lines.append("_No convert limit orders are resting._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)
