"""Wallet asset tools (inventory E) — `/sapi/v1/asset` + `/sapi/v3/asset`.

Eleven tools. Nine of them only read:

- `binance_get_funding_wallet`    POST /sapi/v1/asset/get-funding-asset  (IP 1)
- `binance_get_user_assets`       POST /sapi/v3/asset/getUserAsset       (IP 5)
- `binance_get_wallet_balances`   GET  /sapi/v1/asset/wallet/balance     (IP 60)
- `binance_get_transfer_history`  GET  /sapi/v1/asset/transfer           (IP 1)
- `binance_get_dust_log`          GET  /sapi/v1/asset/dribblet           (IP 1)
- `binance_get_dust_convertible`  POST /sapi/v1/asset/dust-btc           (IP 1)
- `binance_get_asset_detail`      GET  /sapi/v1/asset/assetDetail        (IP 1)
- `binance_get_trade_fees`        GET  /sapi/v1/asset/tradeFee           (IP 1)
- `binance_get_asset_dividends`   GET  /sapi/v1/asset/assetDividend      (IP 10)

Two MOVE FUNDS between the account's own wallets:

- `binance_transfer_between_wallets` POST /sapi/v1/asset/transfer (UID 300)
- `binance_convert_dust_to_bnb`      POST /sapi/v1/asset/dust     (UID 10, irreversible)

Both of those pass through the **kill-switch that lives in `binance_mcp.client`**, not
here: a signed non-GET request is refused with `TradingDisabledError` (surfaced as
`Error: … trading is disabled …`) unless the server runs with `BINANCE_ALLOW_TRADING=1`.
Three of the reads above are POSTs that Binance uses as queries — `get-funding-asset`,
`getUserAsset` and `dust-btc` are on the client's `POST_READ_ALLOWLIST`, so they keep
working with the kill-switch off. Nothing in this module can move funds OUT of Binance:
there is no withdrawal tool here and `/sapi/v1/capital/withdraw/apply` is refused by the
client under every configuration.

Money amounts are **strings** end to end: validated as positive decimals and forwarded
verbatim, because a float round-trip silently changes the amount that is transferred.

The Funding wallet read by `binance_get_funding_wallet` is the wallet behind Binance
Pay, Binance Card and Binance Gift Card (research/03 §1). Card *spending* has no API
surface at all; the Funding balance and the Spot⇄Funding transfer log are the closest
proxies that exist.

None of these endpoints exist on the spot testnet (`/sapi` is not served there).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Self

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from binance_mcp.client import Repeat, get_client
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import ResponseFormat, clip_response, epoch_to_human, fmt_num, to_json
from binance_mcp.server import mcp

# Context-window guard (rule 8): never render more rows than this, whatever the API
# returned or the caller's `limit` asked for.
MAX_DISPLAY_ROWS = 50

# `assetDividend` rejects a start/end span wider than 180 days ("There cannot be more
# than 180 days between startTime and endTime", inventory E / S3); checked before the
# call so the caller gets a clear message instead of Binance's generic rejection.
_DIVIDEND_WINDOW_MS = 180 * 24 * 60 * 60 * 1000

# Bare asset codes (BTC, and the real one-letter assets W and S) — never {2,20}.
_ASSET_PATTERN = r"^[A-Z0-9]{1,20}$"
# Trading pairs (BTCUSDT) for the isolated-margin transfer legs and the fee lookup.
_SYMBOL_PATTERN = r"^[A-Z0-9]{2,20}$"

# Client-side guard on one dust conversion: Binance documents no cap, but an unbounded
# list on a fund-moving call is not something to discover in production. Convert in
# batches if you somehow hold more than this many dust assets.
_MAX_DUST_ASSETS = 100

# A transferred `amount` must be a plain decimal: `123` or `123.45`. `Decimal()` also
# accepts '1E+2', '+5', 'Infinity' and 'NaN', none of which belong in a fund-moving
# query string — and '1E+2' is a hundred times '1' if the far side truncates instead
# of parsing. The signed payload is the literal string, so it is pinned here.
_PLAIN_DECIMAL_PATTERN = re.compile(r"^\d+(\.\d+)?$")


def _to_ms(value: int | str | None, field: str) -> int | None:
    """Accept an epoch-ms int or an ISO-8601 string on the tool surface; send ms.

    A numeric string counts as epoch ms only with at least 12 digits (a real ms
    timestamp is 13 today); anything shorter is parsed as ISO-8601 so a typo surfaces
    as an error instead of silently becoming a bogus timestamp.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = value.strip()
    if text.isdigit() and len(text) >= 12:
        return int(text)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RuntimeError(
            f"{field} must be an epoch-ms integer (>= 12 digits) or an ISO-8601 date/datetime "
            f"(e.g. '2026-09-01' or '2026-09-01T00:00:00Z'); got {value!r}."
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _envelope_error(data: Any) -> str | None:
    """Catch a `/sapi` 200 that carries `code`/`msg` but no `success` key.

    The client's envelope check only fires on `success: false`; several `/sapi/v1/asset`
    endpoints answer HTTP 200 with `{"code": -x, "msg": "..."}` instead, which would
    otherwise render as an empty result.
    """
    if not isinstance(data, dict):
        return None
    code = data.get("code")
    if code not in (None, 200):
        return f"Error: {data.get('msg')} (code {code})"
    return None


def _decimal_or_zero(value: Any) -> Decimal:
    """Parse a Binance decimal string for sorting/comparison; malformed counts as zero."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        return Decimal(0)


def _balance_rows(rows: list[dict[str, Any]], title: str, note: str) -> list[str]:
    """Render a `[{asset, free, locked, freeze, withdrawing, ipoable?, btcValuation?}]` list."""
    lines = [f"# {title}", "", note, ""]
    if not rows:
        lines.append("_No balances returned (the wallet is empty, or the asset filter matched nothing)._")
        return lines
    has_ipoable = any("ipoable" in row for row in rows)
    has_btc = any(row.get("btcValuation") is not None for row in rows)
    header = ["asset", "free", "locked", "freeze", "withdrawing"]
    if has_ipoable:
        header.append("ipoable")
    if has_btc:
        header.append("BTC value")
    ordered = sorted(rows, key=lambda row: _decimal_or_zero(row.get("btcValuation")), reverse=True) if has_btc else rows
    lines.append(f"Showing **{min(len(ordered), MAX_DISPLAY_ROWS):,}** of **{len(ordered):,}** asset(s).")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in ordered[:MAX_DISPLAY_ROWS]:
        cells = [
            str(row.get("asset", "N/A")),
            fmt_num(row.get("free")),
            fmt_num(row.get("locked")),
            fmt_num(row.get("freeze")),
            fmt_num(row.get("withdrawing")),
        ]
        if has_ipoable:
            cells.append(fmt_num(row.get("ipoable")))
        if has_btc:
            cells.append(fmt_num(row.get("btcValuation")))
        lines.append("| " + " | ".join(cells) + " |")
    if len(ordered) > MAX_DISPLAY_ROWS:
        lines.append("")
        lines.append(
            f"_[{len(ordered) - MAX_DISPLAY_ROWS} more asset(s) not shown — filter with `asset` or use "
            'response_format="json"]_'
        )
    if has_btc:
        total = sum((_decimal_or_zero(row.get("btcValuation")) for row in ordered), Decimal(0))
        lines.append("")
        lines.append(f"_Total across all returned assets: {fmt_num(total)} BTC._")
    return lines


# -- enums ---------------------------------------------------------------------------


class TransferType(StrEnum):
    """The 31 wallet pairs `POST /sapi/v1/asset/transfer` accepts (inventory E, S3).

    Read them as FROM_TO. MAIN = Spot, FUNDING = Funding (the Pay/Card/Gift-Card
    wallet), UMFUTURE = USDⓈ-M futures, CMFUTURE = COIN-M futures, MARGIN = cross
    margin, ISOLATEDMARGIN = isolated margin (needs the pair in fromSymbol/toSymbol),
    OPTION = options, PORTFOLIO_MARGIN = portfolio margin.
    """

    MAIN_UMFUTURE = "MAIN_UMFUTURE"
    MAIN_CMFUTURE = "MAIN_CMFUTURE"
    MAIN_MARGIN = "MAIN_MARGIN"
    UMFUTURE_MAIN = "UMFUTURE_MAIN"
    UMFUTURE_MARGIN = "UMFUTURE_MARGIN"
    CMFUTURE_MAIN = "CMFUTURE_MAIN"
    CMFUTURE_MARGIN = "CMFUTURE_MARGIN"
    MARGIN_MAIN = "MARGIN_MAIN"
    MARGIN_UMFUTURE = "MARGIN_UMFUTURE"
    MARGIN_CMFUTURE = "MARGIN_CMFUTURE"
    ISOLATEDMARGIN_MARGIN = "ISOLATEDMARGIN_MARGIN"
    MARGIN_ISOLATEDMARGIN = "MARGIN_ISOLATEDMARGIN"
    ISOLATEDMARGIN_ISOLATEDMARGIN = "ISOLATEDMARGIN_ISOLATEDMARGIN"
    MAIN_FUNDING = "MAIN_FUNDING"
    FUNDING_MAIN = "FUNDING_MAIN"
    FUNDING_UMFUTURE = "FUNDING_UMFUTURE"
    UMFUTURE_FUNDING = "UMFUTURE_FUNDING"
    MARGIN_FUNDING = "MARGIN_FUNDING"
    FUNDING_MARGIN = "FUNDING_MARGIN"
    FUNDING_CMFUTURE = "FUNDING_CMFUTURE"
    CMFUTURE_FUNDING = "CMFUTURE_FUNDING"
    MAIN_OPTION = "MAIN_OPTION"
    OPTION_MAIN = "OPTION_MAIN"
    UMFUTURE_OPTION = "UMFUTURE_OPTION"
    OPTION_UMFUTURE = "OPTION_UMFUTURE"
    MARGIN_OPTION = "MARGIN_OPTION"
    OPTION_MARGIN = "OPTION_MARGIN"
    FUNDING_OPTION = "FUNDING_OPTION"
    OPTION_FUNDING = "OPTION_FUNDING"
    MAIN_PORTFOLIO_MARGIN = "MAIN_PORTFOLIO_MARGIN"
    PORTFOLIO_MARGIN_MAIN = "PORTFOLIO_MARGIN_MAIN"


# `fromSymbol` / `toSymbol` name the isolated-margin pair; Binance requires them only
# for the legs that touch an isolated-margin account (inventory E).
_FROM_SYMBOL_TYPES = (TransferType.ISOLATEDMARGIN_MARGIN, TransferType.ISOLATEDMARGIN_ISOLATEDMARGIN)
_TO_SYMBOL_TYPES = (TransferType.MARGIN_ISOLATEDMARGIN, TransferType.ISOLATEDMARGIN_ISOLATEDMARGIN)


class DustAccountType(StrEnum):
    """Which account the dust preview/conversion reads: SPOT (default) or MARGIN."""

    SPOT = "SPOT"
    MARGIN = "MARGIN"


# -- input models --------------------------------------------------------------------


class _BaseInput(BaseModel):
    """Shared pydantic config for every input model in this module."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class _ReadInput(_BaseInput):
    """Every read tool offers the markdown/JSON switch."""

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: `markdown` (human-readable) or `json` (raw Binance payload).",
    )


class FundingWalletInput(_ReadInput):
    """Params for `POST /sapi/v1/asset/get-funding-asset`."""

    asset: str | None = Field(
        default=None,
        min_length=1,
        pattern=_ASSET_PATTERN,
        description="Only this asset, e.g. USDT. Omit for every asset with a balance. Uppercased automatically.",
    )
    need_btc_valuation: bool = Field(
        default=False,
        description="Also return each balance valued in BTC (Binance: needBtcValuation).",
    )

    @field_validator("asset", mode="before")
    @classmethod
    def _upper_asset(cls, value: str | None) -> str | None:
        return value.upper() if isinstance(value, str) else value


class UserAssetsInput(_ReadInput):
    """Params for `POST /sapi/v3/asset/getUserAsset`."""

    asset: str | None = Field(
        default=None,
        min_length=1,
        pattern=_ASSET_PATTERN,
        description="Only this asset, e.g. BTC. Omit for every asset with a positive balance.",
    )
    need_btc_valuation: bool = Field(
        default=False,
        description="Also return each balance valued in BTC (Binance: needBtcValuation).",
    )

    @field_validator("asset", mode="before")
    @classmethod
    def _upper_asset(cls, value: str | None) -> str | None:
        return value.upper() if isinstance(value, str) else value


class WalletBalancesInput(_ReadInput):
    """Params for `GET /sapi/v1/asset/wallet/balance`."""

    quote_asset: str | None = Field(
        default=None,
        min_length=1,
        pattern=_ASSET_PATTERN,
        description="Currency each wallet total is valued in (Binance: quoteAsset). Binance defaults to BTC.",
    )

    @field_validator("quote_asset", mode="before")
    @classmethod
    def _upper_asset(cls, value: str | None) -> str | None:
        return value.upper() if isinstance(value, str) else value


class TransferHistoryInput(_ReadInput):
    """Params for `GET /sapi/v1/asset/transfer`."""

    type: TransferType = Field(
        description=(
            "Which transfer direction to list — MANDATORY, Binance has no 'all types' option. "
            "One call per direction (e.g. MAIN_FUNDING and FUNDING_MAIN are two separate queries)."
        )
    )
    start_time: int | str | None = Field(
        default=None,
        description="Window start: epoch ms or an ISO-8601 date/datetime. Binance only keeps the last 6 months.",
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end: epoch ms or an ISO-8601 date/datetime. Defaults to now.",
    )
    page: int = Field(default=1, ge=1, description="1-based page number (Binance: current).")
    limit: int = Field(default=20, ge=1, le=100, description="Rows per page, 1-100 (Binance: size, default 10).")


class DustLogInput(_ReadInput):
    """Params for `GET /sapi/v1/asset/dribblet`."""

    account_type: DustAccountType | None = Field(
        default=None,
        description="SPOT or MARGIN (Binance: accountType). Omit for Binance's default, SPOT.",
    )
    start_time: int | str | None = Field(
        default=None,
        description="Window start: epoch ms or an ISO-8601 date/datetime.",
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end: epoch ms or an ISO-8601 date/datetime.",
    )


class DustConvertibleInput(_ReadInput):
    """Params for `POST /sapi/v1/asset/dust-btc` (a query despite the verb)."""

    account_type: DustAccountType | None = Field(
        default=None,
        description="SPOT or MARGIN (Binance: accountType). Omit for Binance's default, SPOT.",
    )


class AssetDetailInput(_ReadInput):
    """Params for `GET /sapi/v1/asset/assetDetail`."""

    asset: str | None = Field(
        default=None,
        min_length=1,
        pattern=_ASSET_PATTERN,
        description="Only this asset, e.g. BTC. Omit for every asset Binance lists.",
    )

    @field_validator("asset", mode="before")
    @classmethod
    def _upper_asset(cls, value: str | None) -> str | None:
        return value.upper() if isinstance(value, str) else value


class TradeFeesInput(_ReadInput):
    """Params for `GET /sapi/v1/asset/tradeFee`."""

    symbol: str | None = Field(
        default=None,
        min_length=1,
        pattern=_SYMBOL_PATTERN,
        description="Only this trading pair, e.g. BTCUSDT. Omit for every symbol (thousands of rows).",
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def _upper_symbol(cls, value: str | None) -> str | None:
        return value.upper() if isinstance(value, str) else value


class AssetDividendsInput(_ReadInput):
    """Params for `GET /sapi/v1/asset/assetDividend`."""

    asset: str | None = Field(
        default=None,
        min_length=1,
        pattern=_ASSET_PATTERN,
        description="Only distributions of this asset, e.g. BNB. Omit for all of them.",
    )
    start_time: int | str | None = Field(
        default=None,
        description=(
            "Window start: epoch ms or an ISO-8601 date/datetime. start_time..end_time must span "
            "at most 180 days — checked locally before the call."
        ),
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end: epoch ms or an ISO-8601 date/datetime. Defaults to now.",
    )
    limit: int = Field(default=20, ge=1, le=500, description="Rows to return, 1-500 (Binance default 20).")

    @field_validator("asset", mode="before")
    @classmethod
    def _upper_asset(cls, value: str | None) -> str | None:
        return value.upper() if isinstance(value, str) else value


class TransferBetweenWalletsInput(_BaseInput):
    """Params for `POST /sapi/v1/asset/transfer` — a REAL movement of funds.

    The isolated-margin legs need the pair named: `from_symbol` for
    ISOLATEDMARGIN_MARGIN and ISOLATEDMARGIN_ISOLATEDMARGIN, `to_symbol` for
    MARGIN_ISOLATEDMARGIN and ISOLATEDMARGIN_ISOLATEDMARGIN. Both rules are enforced
    here so a malformed transfer dies locally, before anything is signed or sent.
    """

    type: TransferType = Field(
        description=(
            "Which wallets to move between, as FROM_TO (e.g. MAIN_FUNDING moves Spot → Funding, "
            "FUNDING_MAIN moves Funding → Spot). 31 documented directions."
        )
    )
    asset: str = Field(
        min_length=1,
        pattern=_ASSET_PATTERN,
        description="Asset to move, e.g. USDT. Uppercased automatically.",
    )
    amount: str = Field(
        description="Amount to move as a decimal STRING (e.g. '12.5'). Sent verbatim — never a float.",
    )
    from_symbol: str | None = Field(
        default=None,
        min_length=1,
        pattern=_SYMBOL_PATTERN,
        description=(
            "Isolated-margin pair the funds leave, e.g. BTCUSDT (Binance: fromSymbol). Required for "
            "ISOLATEDMARGIN_MARGIN and ISOLATEDMARGIN_ISOLATEDMARGIN."
        ),
    )
    to_symbol: str | None = Field(
        default=None,
        min_length=1,
        pattern=_SYMBOL_PATTERN,
        description=(
            "Isolated-margin pair the funds arrive in, e.g. BTCUSDT (Binance: toSymbol). Required for "
            "MARGIN_ISOLATEDMARGIN and ISOLATEDMARGIN_ISOLATEDMARGIN."
        ),
    )

    @field_validator("asset", mode="before")
    @classmethod
    def _upper_asset(cls, value: str) -> str:
        return value.upper() if isinstance(value, str) else value

    @field_validator("from_symbol", "to_symbol", mode="before")
    @classmethod
    def _upper_symbol(cls, value: str | None) -> str | None:
        return value.upper() if isinstance(value, str) else value

    @field_validator("amount")
    @classmethod
    def _check_positive_decimal(cls, value: str, info: ValidationInfo) -> str:
        """Validate as a plain positive decimal and return the ORIGINAL string, unrounded.

        Only the `123` / `123.45` shapes pass. Scientific notation, a leading sign and
        `Infinity`/`NaN` all parse fine as a `Decimal` but are not what Binance expects
        in an `amount`, and `1E+2` is a hundred times `1` if the far side truncates
        instead of parsing. On a fund-moving call that is not a risk worth taking.
        """
        if not _PLAIN_DECIMAL_PATTERN.fullmatch(value):
            raise ValueError(
                f"{info.field_name} must be a plain decimal string such as '12.5' — no scientific "
                f"notation, no sign, no separators; got {value!r}."
            )
        if Decimal(value) <= 0:
            raise ValueError(f"{info.field_name} must be a positive decimal; got {value!r}.")
        return value

    @model_validator(mode="after")
    def _check_isolated_margin_legs(self) -> Self:
        if self.type in _FROM_SYMBOL_TYPES and self.from_symbol is None:
            raise ValueError(
                f"{self.type.value} transfers require from_symbol (Binance: fromSymbol) — the isolated-margin "
                "pair the funds leave, e.g. 'BTCUSDT'."
            )
        if self.type in _TO_SYMBOL_TYPES and self.to_symbol is None:
            raise ValueError(
                f"{self.type.value} transfers require to_symbol (Binance: toSymbol) — the isolated-margin "
                "pair the funds arrive in, e.g. 'BTCUSDT'."
            )
        return self


class ConvertDustInput(_BaseInput):
    """Params for `POST /sapi/v1/asset/dust` — an IRREVERSIBLE conversion to BNB."""

    assets: list[str] = Field(
        min_length=1,
        max_length=_MAX_DUST_ASSETS,
        description=(
            "Assets to convert to BNB, e.g. ['BTC', 'ETH']. At least one, at most 100 per call "
            "(a client-side guard — Binance documents no cap; split into batches if you need more). "
            "Sent as repeated `asset` query keys, which is what Binance expects. Preview them with "
            "`binance_get_dust_convertible` first — the conversion cannot be undone."
        ),
    )
    account_type: DustAccountType | None = Field(
        default=None,
        description="SPOT or MARGIN (Binance: accountType). Omit for Binance's default, SPOT.",
    )

    @field_validator("assets", mode="before")
    @classmethod
    def _upper_assets(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [item.upper().strip() if isinstance(item, str) else item for item in value]
        return value

    @field_validator("assets")
    @classmethod
    def _check_assets(cls, value: list[str]) -> list[str]:
        for item in value:
            if not re.fullmatch(_ASSET_PATTERN, item):
                raise ValueError(f"each entry of assets must be a bare asset code such as 'BTC'; got {item!r}.")
        return value


# -- tools ---------------------------------------------------------------------------


@mcp.tool(
    name="binance_get_funding_wallet",
    annotations=ToolAnnotations(
        title="Binance Funding Wallet Balances",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_funding_wallet(params: FundingWalletInput) -> str:
    """Read the Funding wallet — the wallet behind Binance Pay, Card and Gift Card.

    Calls `POST /sapi/v1/asset/get-funding-asset` (SIGNED, IP weight 1). Binance uses
    POST for this query; it is on the client's read allowlist, so it works with the
    trading kill-switch off. Per Binance's own documentation this endpoint "supports
    querying: Binance Pay, **Binance Card**, Binance Gift Card, Stock Token" — i.e. it
    is the Funding wallet, and the closest thing to a card balance the API exposes.

    When to Use:
    - To see what is sitting in Funding (Pay / Card / Gift Card / P2P proceeds).
    - Before a FUNDING_MAIN transfer, to check there is something to move.

    When NOT to Use:
    - For Spot balances — use `binance_get_user_assets` here, or
      `binance_get_spot_account` (spot_account.py) for the full account view.
    - For a wallet-by-wallet total across Spot/Funding/Earn/Futures — use
      `binance_get_wallet_balances`.
    - To list Binance **Card spending**: that has no API endpoint at all. Card-funded
      Binance Pay payments show up in `binance_get_pay_transactions` (pay.py) with
      walletType 4/6; nothing else is retrievable.

    Returns:
    A markdown table of asset / free / locked / freeze / withdrawing (plus a BTC
    valuation column and total when `need_btc_valuation` is set), capped at 50 rows, or
    the raw Binance array with `response_format="json"`.

    Examples:
        params = {}
        params = {"asset": "USDT"}
        params = {"need_btc_valuation": True, "response_format": "json"}

    Error Handling:
    An empty list means the Funding wallet holds nothing (common — a Spot-only account
    never funds it). -2015 means the key lacks Reading permission or this IP is not
    allowlisted. A 404 means the base URL has no `/sapi` (the spot testnet).
    """
    try:
        query: dict[str, Any] = {"needBtcValuation": params.need_btc_valuation}
        if params.asset is not None:
            query["asset"] = params.asset
        client = get_client()
        resp = await client.request("POST", "/sapi/v1/asset/get-funding-asset", auth="signed", params=query)
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        rows = data if isinstance(data, list) else []
        lines = _balance_rows(
            rows,
            "Binance Funding wallet",
            "_This is the wallet behind Binance Pay, Binance Card and Binance Gift Card._",
        )
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_user_assets",
    annotations=ToolAnnotations(
        title="Binance User Assets (Spot)",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_user_assets(params: UserAssetsInput) -> str:
    """List the Spot wallet's non-zero balances, optionally valued in BTC.

    Calls `POST /sapi/v3/asset/getUserAsset` (SIGNED, IP weight 5). Another Binance
    query that uses POST; it is on the client's read allowlist and works with the
    trading kill-switch off. With no `asset` filter it returns every asset with a
    positive balance — unlike `/api/v3/account`, zero balances are omitted by Binance
    itself.

    When to Use:
    - For a compact "what do I actually hold on Spot" answer, with a BTC valuation.
    - As the asset seed for `binance_discover_traded_symbols` (trade_history.py).

    When NOT to Use:
    - For the full account view (permissions, commission rates, canTrade) — use
      `binance_get_spot_account` (spot_account.py).
    - For the Funding wallet — use `binance_get_funding_wallet`.

    Returns:
    A markdown table of asset / free / locked / freeze / withdrawing / ipoable (plus BTC
    valuation and a total when `need_btc_valuation` is set), capped at 50 rows, or the
    raw Binance array with `response_format="json"`.

    Examples:
        params = {}
        params = {"asset": "BTC", "need_btc_valuation": True}

    Error Handling:
    -2015 means the key lacks Reading permission or this IP is not allowlisted. A 404
    means the base URL has no `/sapi` (the spot testnet).
    """
    try:
        query: dict[str, Any] = {"needBtcValuation": params.need_btc_valuation}
        if params.asset is not None:
            query["asset"] = params.asset
        client = get_client()
        resp = await client.request("POST", "/sapi/v3/asset/getUserAsset", auth="signed", params=query)
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        rows = data if isinstance(data, list) else []
        lines = _balance_rows(
            rows,
            "Binance Spot wallet assets",
            "_Binance omits zero balances from this endpoint._",
        )
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_wallet_balances",
    annotations=ToolAnnotations(
        title="Binance Wallet Balances (per wallet)",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_wallet_balances(params: WalletBalancesInput) -> str:
    """Show one total per wallet: Spot, Funding, Cross/Isolated Margin, Futures, Earn…

    Calls `GET /sapi/v1/asset/wallet/balance` (SIGNED, **IP weight 60** — 60 of the
    12000/min `/sapi` IP budget, so it is fine occasionally but not in a poll loop).
    Binance returns one row per wallet with its total value in `quote_asset` and whether
    the wallet is activated.

    When to Use:
    - First call when asking "where is my money" — it says which wallets hold anything
      before you spend weight listing assets wallet by wallet.
    - Before a transfer, to confirm the source wallet actually holds the balance.

    When NOT to Use:
    - For per-asset detail — use `binance_get_user_assets` (Spot) or
      `binance_get_funding_wallet` (Funding).
    - For a daily history of balances — use `binance_get_account_snapshot`
      (wallet_account.py), which is far heavier (IP 2400).

    Returns:
    A markdown table of wallet / activated / balance in the quote asset, plus the sum,
    or the raw Binance array with `response_format="json"`.

    Examples:
        params = {}
        params = {"quote_asset": "USDT"}

    Error Handling:
    -2015 means the key lacks Reading permission or this IP is not allowlisted. A 404
    means the base URL has no `/sapi` (the spot testnet).
    """
    try:
        query: dict[str, Any] = {}
        if params.quote_asset is not None:
            query["quoteAsset"] = params.quote_asset
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/asset/wallet/balance", auth="signed", params=query or None)
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        rows = data if isinstance(data, list) else []
        quote = params.quote_asset or "BTC"
        lines = [f"# Binance wallet balances (valued in {quote})", ""]
        if not rows:
            lines.append("_Binance returned no wallets._")
            return clip_response("\n".join(lines))
        lines.append(f"| wallet | activated | balance ({quote}) |")
        lines.append("|---|---|---|")
        for row in rows[:MAX_DISPLAY_ROWS]:
            lines.append(
                f"| {row.get('walletName', 'N/A')} | {bool(row.get('activate'))} | {fmt_num(row.get('balance'))} |"
            )
        if len(rows) > MAX_DISPLAY_ROWS:
            lines.append(f"_[{len(rows) - MAX_DISPLAY_ROWS} more wallet(s) not shown]_")
        total = sum((_decimal_or_zero(row.get("balance")) for row in rows), Decimal(0))
        lines.append("")
        lines.append(f"_Total across all wallets: {fmt_num(total)} {quote}._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_transfer_history",
    annotations=ToolAnnotations(
        title="Binance Universal Transfer History",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_transfer_history(params: TransferHistoryInput) -> str:
    """List past transfers between the account's own wallets, one direction at a time.

    Calls `GET /sapi/v1/asset/transfer` (SIGNED, IP weight 1). Unlike the POST that
    performs a transfer, this read does **not** need the key's "Permits Universal
    Transfer" flag.

    `type` is mandatory and Binance offers no "all directions" value: MAIN_FUNDING and
    FUNDING_MAIN are two separate queries, so a full Spot⇄Funding picture costs two
    calls. `MAIN_FUNDING` / `FUNDING_MAIN` is also the closest proxy to a Binance Card
    top-up log, since the card was funded out of the Funding wallet.

    When to Use:
    - To reconcile where a balance went between wallets.
    - To reconstruct Funding-wallet activity that Pay/fiat history does not explain.

    When NOT to Use:
    - For deposits/withdrawals to and from other platforms — use
      `binance_get_deposit_history` / `binance_get_withdraw_history` (wallet_capital.py).
    - To perform a transfer — that is `binance_transfer_between_wallets`.

    Returns:
    Binance's `{total, rows}` rendered as a markdown table of time / asset / amount /
    type / status / tranId, with the page position and the per-asset totals of the rows
    shown, or the raw envelope with `response_format="json"`.

    Pagination:
    `page` → Binance's `current` (1-based), `limit` → Binance's `size` (max 100, Binance
    default 10). `total` in the response is the full count for the filter, so
    `page * limit < total` means there is more.

    Windows:
    Binance "supports query within the last 6 months only" and defaults to the **last 7
    days** when `start_time`/`end_time` are omitted — so an empty result with no window
    given usually means "nothing in the last week", not "never". Pass `start_time` to
    look further back.

    Examples:
        params = {"type": "MAIN_FUNDING"}
        params = {"type": "FUNDING_MAIN", "start_time": "2026-03-01", "limit": 100}

    Error Handling:
    -2015 means the key lacks Reading permission or this IP is not allowlisted. Dates
    more than 6 months old simply return nothing.
    """
    try:
        start_ms = _to_ms(params.start_time, "start_time")
        end_ms = _to_ms(params.end_time, "end_time")
        if start_ms is not None and end_ms is not None and end_ms <= start_ms:
            return "Error: end_time must be after start_time."
        query: dict[str, Any] = {
            "type": params.type.value,
            "current": params.page,
            "size": params.limit,
        }
        if start_ms is not None:
            query["startTime"] = start_ms
        if end_ms is not None:
            query["endTime"] = end_ms
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/asset/transfer", auth="signed", params=query)
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        rows = data.get("rows") or [] if isinstance(data, dict) else []
        total = data.get("total") if isinstance(data, dict) else None
        lines = [f"# Binance transfers — {params.type.value}", ""]
        if not rows:
            lines.append(
                "_No transfers for this direction in range. With no start_time, Binance only looks at the "
                "last 7 days (and never further back than 6 months)._"
            )
            return clip_response("\n".join(lines))
        shown = min(len(rows), MAX_DISPLAY_ROWS)
        if total is not None:
            lines.append(f"Showing **{shown:,}** row(s) on page {params.page} of total **{total:,}**.")
        else:
            lines.append(f"Showing **{shown:,}** row(s) on page {params.page}.")
        lines.append("")
        lines.append("| time | asset | amount | type | status | tranId |")
        lines.append("|---|---|---|---|---|---|")
        for row in rows[:MAX_DISPLAY_ROWS]:
            lines.append(
                f"| {epoch_to_human(row.get('timestamp'))} | {row.get('asset', 'N/A')} | "
                f"{fmt_num(row.get('amount'))} | {row.get('type', 'N/A')} | {row.get('status', 'N/A')} | "
                f"{row.get('tranId', 'N/A')} |"
            )
        if len(rows) > MAX_DISPLAY_ROWS:
            lines.append(f"_[{len(rows) - MAX_DISPLAY_ROWS} more row(s) on this page not shown]_")
        totals: dict[str, Decimal] = {}
        for row in rows[:MAX_DISPLAY_ROWS]:
            asset = str(row.get("asset") or "?")
            totals[asset] = totals.get(asset, Decimal(0)) + _decimal_or_zero(row.get("amount"))
        if totals:
            rendered = ", ".join(f"{fmt_num(amount)} {asset}" for asset, amount in sorted(totals.items()))
            lines.append("")
            lines.append(f"_Total moved in the rows above: {rendered}._")
        if total is not None and params.page * params.limit < total:
            lines.append(f"_More available — request page {params.page + 1}._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_dust_log",
    annotations=ToolAnnotations(
        title="Binance Dust Conversion Log",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_dust_log(params: DustLogInput) -> str:
    """List past dust-to-BNB conversions, with the per-asset detail of each one.

    Calls `GET /sapi/v1/asset/dribblet` (SIGNED, IP weight 1). Binance returns "only the
    last 100 records" and "only records after 2020/12/01".

    When to Use:
    - To find out what a past "convert small balances to BNB" actually converted, and
      what the service charge was.
    - To check whether an asset disappeared because it was swept as dust.

    When NOT to Use:
    - To see what could be converted right now — use `binance_get_dust_convertible`.
    - To actually convert — that is `binance_convert_dust_to_bnb`.

    Returns:
    One markdown section per conversion batch (time, transId, total transferred BNB,
    total service charge) followed by a table of the assets in that batch, or the raw
    envelope with `response_format="json"`.

    Windows:
    `start_time`/`end_time` are optional; Binance caps the history at the last 100
    records regardless of the window, and keeps nothing before 2020-12-01.

    Examples:
        params = {}
        params = {"start_time": "2026-01-01", "end_time": "2026-06-30"}

    Error Handling:
    -2015 means the key lacks Reading permission or this IP is not allowlisted.
    """
    try:
        start_ms = _to_ms(params.start_time, "start_time")
        end_ms = _to_ms(params.end_time, "end_time")
        if start_ms is not None and end_ms is not None and end_ms <= start_ms:
            return "Error: end_time must be after start_time."
        query: dict[str, Any] = {}
        if params.account_type is not None:
            query["accountType"] = params.account_type.value
        if start_ms is not None:
            query["startTime"] = start_ms
        if end_ms is not None:
            query["endTime"] = end_ms
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/asset/dribblet", auth="signed", params=query or None)
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        batches = data.get("userAssetDribblets") or [] if isinstance(data, dict) else []
        total = data.get("total") if isinstance(data, dict) else None
        lines = ["# Binance dust conversion log", ""]
        if not batches:
            lines.append(
                "_No dust conversions in range. Binance keeps only the last 100 records, and nothing "
                "before 2020-12-01._"
            )
            return clip_response("\n".join(lines))
        lines.append(
            f"Showing **{min(len(batches), MAX_DISPLAY_ROWS):,}** of **{total if total is not None else len(batches):,}** conversion(s)."
        )
        for batch in batches[:MAX_DISPLAY_ROWS]:
            lines.append("")
            lines.append(f"## {epoch_to_human(batch.get('operateTime'))} — transId `{batch.get('transId', 'N/A')}`")
            lines.append("")
            lines.append(f"- **BNB received**: {fmt_num(batch.get('totalTransferedAmount'))}")
            lines.append(f"- **service charge (BNB)**: {fmt_num(batch.get('totalServiceChargeAmount'))}")
            details = batch.get("userAssetDribbletDetails") or []
            if details:
                lines.append("")
                lines.append("| from asset | amount | BNB received | service charge | time | transId |")
                lines.append("|---|---|---|---|---|---|")
                for detail in details[:MAX_DISPLAY_ROWS]:
                    lines.append(
                        f"| {detail.get('fromAsset', 'N/A')} | {fmt_num(detail.get('amount'))} | "
                        f"{fmt_num(detail.get('transferedAmount'))} | "
                        f"{fmt_num(detail.get('serviceChargeAmount'))} | "
                        f"{epoch_to_human(detail.get('operateTime'))} | {detail.get('transId', 'N/A')} |"
                    )
                if len(details) > MAX_DISPLAY_ROWS:
                    lines.append(f"_[{len(details) - MAX_DISPLAY_ROWS} more asset(s) in this batch not shown]_")
        if len(batches) > MAX_DISPLAY_ROWS:
            lines.append("")
            lines.append(f"_[{len(batches) - MAX_DISPLAY_ROWS} more conversion(s) not shown — narrow the window]_")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_dust_convertible",
    annotations=ToolAnnotations(
        title="Binance Convertible Dust (preview)",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_dust_convertible(params: DustConvertibleInput) -> str:
    """Preview which small balances can be converted to BNB, and what they are worth.

    Calls `POST /sapi/v1/asset/dust-btc` (SIGNED, IP weight 1). Binance uses POST for
    this query; it is on the client's read allowlist, so the preview works even with the
    trading kill-switch off. **Nothing is converted by this call.**

    When to Use:
    - Always, immediately before `binance_convert_dust_to_bnb` — it names the exact
      assets that are eligible and the BNB each one yields.

    When NOT to Use:
    - To see conversions that already happened — use `binance_get_dust_log`.

    Returns:
    A markdown table of asset / free amount / value in BTC / BNB you would receive
    (on-exchange and off-exchange rates when Binance sends both), plus the batch totals
    and the service-charge percentage, or the raw envelope with `response_format="json"`.

    Examples:
        params = {}
        params = {"account_type": "MARGIN"}

    Error Handling:
    An empty `details` list means nothing currently qualifies as dust. -2015 means the
    key lacks Reading permission or this IP is not allowlisted.
    """
    try:
        query: dict[str, Any] = {}
        if params.account_type is not None:
            query["accountType"] = params.account_type.value
        client = get_client()
        resp = await client.request("POST", "/sapi/v1/asset/dust-btc", auth="signed", params=query or None)
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        details = data.get("details") or [] if isinstance(data, dict) else []
        lines = ["# Binance convertible dust", "", "_Preview only — nothing was converted by this call._", ""]
        if not details:
            lines.append("_No balances currently qualify as convertible dust._")
            return clip_response("\n".join(lines))
        lines.append("| asset | name | free | value (BTC) | BNB received | BNB (off-exchange) |")
        lines.append("|---|---|---|---|---|---|")
        for row in details[:MAX_DISPLAY_ROWS]:
            lines.append(
                f"| {row.get('asset', 'N/A')} | {row.get('assetFullName', '')} | {fmt_num(row.get('amountFree'))} | "
                f"{fmt_num(row.get('toBTC'))} | {fmt_num(row.get('toBNB'))} | "
                f"{fmt_num(row.get('toBNBOffExchange'))} |"
            )
        if len(details) > MAX_DISPLAY_ROWS:
            lines.append(f"_[{len(details) - MAX_DISPLAY_ROWS} more asset(s) not shown]_")
        lines.append("")
        lines.append(f"- **total value**: {fmt_num(data.get('totalTransferBtc'))} BTC")
        lines.append(f"- **total BNB you would receive**: {fmt_num(data.get('totalTransferBNB'))}")
        lines.append(f"- **service charge**: {data.get('dribbletPercentage', 'N/A')}")
        lines.append("")
        lines.append(
            "_Convert with `binance_convert_dust_to_bnb`, passing the assets above — that call is "
            "irreversible and requires BINANCE_ALLOW_TRADING=1._"
        )
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_asset_detail",
    annotations=ToolAnnotations(
        title="Binance Asset Detail",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_asset_detail(params: AssetDetailInput) -> str:
    """Report per-asset deposit/withdraw status, withdraw fee and minimum.

    Calls `GET /sapi/v1/asset/assetDetail` (SIGNED, IP weight 1). Binance answers with a
    **map keyed by asset**, not a list.

    When to Use:
    - To check whether deposits or withdrawals are currently suspended for an asset.
    - To read an asset's withdraw fee and minimum before planning a movement elsewhere.

    When NOT to Use:
    - For per-network detail (which chain, its own fee/min) — use
      `binance_get_coin_config` (wallet_capital.py).
    - To actually withdraw: this server has no withdrawal tool, by design, and the HTTP
      client refuses `/sapi/v1/capital/withdraw/apply` under every configuration.

    Returns:
    A markdown table of asset / deposit / withdraw / withdraw fee / min withdraw / tip,
    capped at 50 assets (pass `asset` to narrow), or the raw map with
    `response_format="json"`.

    Examples:
        params = {"asset": "BTC"}
        params = {}

    Error Handling:
    -2015 means the key lacks Reading permission or this IP is not allowlisted. An
    unknown asset comes back as an empty map, not an error.
    """
    try:
        query: dict[str, Any] = {}
        if params.asset is not None:
            query["asset"] = params.asset
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/asset/assetDetail", auth="signed", params=query or None)
        data = resp.json()
        # The body is a map keyed by asset, but Binance asset codes are uppercase, so a
        # lowercase `code` key can only be an error envelope — never an asset. Without
        # this, a 200 carrying `{"code": -2015, …}` would render as a successful table
        # with no rows.
        if (envelope := _envelope_error(data)) is not None:
            return envelope
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        detail = data if isinstance(data, dict) else {}
        assets = sorted(detail.keys())
        lines = ["# Binance asset detail", ""]
        if not assets:
            lines.append("_Binance returned no assets (check the `asset` code)._")
            return clip_response("\n".join(lines))
        lines.append(f"Showing **{min(len(assets), MAX_DISPLAY_ROWS):,}** of **{len(assets):,}** asset(s).")
        lines.append("")
        lines.append("| asset | deposit | withdraw | withdraw fee | min withdraw | deposit tip |")
        lines.append("|---|---|---|---|---|---|")
        for name in assets[:MAX_DISPLAY_ROWS]:
            row = detail.get(name) or {}
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {name} | {bool(row.get('depositStatus'))} | {bool(row.get('withdrawStatus'))} | "
                f"{fmt_num(row.get('withdrawFee'))} | {fmt_num(row.get('minWithdrawAmount'))} | "
                f"{row.get('depositTip', '')} |"
            )
        if len(assets) > MAX_DISPLAY_ROWS:
            lines.append("")
            lines.append(
                f"_[{len(assets) - MAX_DISPLAY_ROWS} more asset(s) not shown — pass `asset` to narrow, or use "
                'response_format="json"]_'
            )
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_trade_fees",
    annotations=ToolAnnotations(
        title="Binance Trade Fees",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_trade_fees(params: TradeFeesInput) -> str:
    """Report the maker/taker commission rates that apply to this account.

    Calls `GET /sapi/v1/asset/tradeFee` (SIGNED, IP weight 1). Without `symbol` Binance
    returns **every** symbol — thousands of rows — so the rendering is capped at 50 and
    the JSON output is clipped; pass `symbol` whenever you know it.

    When to Use:
    - To price a trade properly before placing it.
    - To confirm a VIP-tier or BNB-discount fee change took effect.

    When NOT to Use:
    - For the fee actually charged on an executed order — that is in the order's fills
      (`binance_place_order`) or in `binance_get_my_trades` (trade_history.py).
    - For the account-level commission rates on one symbol with the order-book context —
      `binance_get_commission_rates` (spot_account.py) reads `/api/v3/account/commission`.

    Returns:
    A markdown table of symbol / maker / taker as percentages, capped at 50 rows, or the
    raw Binance array with `response_format="json"`.

    Examples:
        params = {"symbol": "BTCUSDT"}
        params = {}

    Error Handling:
    -1121 means the symbol does not exist. -2015 means the key lacks Reading permission
    or this IP is not allowlisted.
    """
    try:
        query: dict[str, Any] = {}
        if params.symbol is not None:
            query["symbol"] = params.symbol
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/asset/tradeFee", auth="signed", params=query or None)
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        rows = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        lines = ["# Binance trade fees", ""]
        if not rows:
            lines.append("_Binance returned no fee rows._")
            return clip_response("\n".join(lines))
        lines.append(f"Showing **{min(len(rows), MAX_DISPLAY_ROWS):,}** of **{len(rows):,}** symbol(s).")
        lines.append("")
        lines.append("| symbol | maker | taker |")
        lines.append("|---|---|---|")
        for row in rows[:MAX_DISPLAY_ROWS]:
            maker = _decimal_or_zero(row.get("makerCommission")) * 100
            taker = _decimal_or_zero(row.get("takerCommission")) * 100
            lines.append(f"| {row.get('symbol', 'N/A')} | {fmt_num(maker)}% | {fmt_num(taker)}% |")
        if len(rows) > MAX_DISPLAY_ROWS:
            lines.append("")
            lines.append(f"_[{len(rows) - MAX_DISPLAY_ROWS} more symbol(s) not shown — pass `symbol` to narrow]_")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_asset_dividends",
    annotations=ToolAnnotations(
        title="Binance Asset Dividend Record",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_asset_dividends(params: AssetDividendsInput) -> str:
    """List asset distributions credited to the account (airdrops, rebates, interest).

    Calls `GET /sapi/v1/asset/assetDividend` (SIGNED, IP weight 10). This is Binance's
    "asset dividend record": savings interest, BNB fee rebates, airdrops, referral
    payouts and similar credits, each with the reason in `enInfo`.

    When to Use:
    - To explain a balance that grew without a trade or a deposit.
    - To total up rebates/airdrops over a period.

    When NOT to Use:
    - For Simple Earn positions and their APR — use the earn tools (simple_earn.py).
    - For trades — use `binance_get_my_trades` (trade_history.py).

    Returns:
    A markdown table of time / asset / amount / description / tranId, capped at 50 rows,
    plus per-asset totals of the rows shown, or the raw envelope with
    `response_format="json"`.

    Windows:
    `start_time`..`end_time` must span **at most 180 days** — Binance rejects anything
    wider, and this is checked locally before the call. Omit both for Binance's own
    default window.

    Examples:
        params = {"asset": "BNB", "limit": 100}
        params = {"start_time": "2026-01-01", "end_time": "2026-06-01"}

    Error Handling:
    A window wider than 180 days is rejected locally with an `Error:` naming the cap,
    instead of round-tripping to Binance. -2015 means the key lacks Reading permission
    or this IP is not allowlisted.
    """
    try:
        start_ms = _to_ms(params.start_time, "start_time")
        end_ms = _to_ms(params.end_time, "end_time")
        if start_ms is not None:
            effective_end_ms = end_ms if end_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
            if effective_end_ms <= start_ms:
                return "Error: end_time must be after start_time."
            if effective_end_ms - start_ms > _DIVIDEND_WINDOW_MS:
                days = (effective_end_ms - start_ms) / (24 * 60 * 60 * 1000)
                return (
                    f"Error: the start_time/end_time window spans ~{days:.1f} days; Binance's assetDividend "
                    "allows at most 180 days between startTime and endTime. Narrow the window and page "
                    "through it."
                )
        query: dict[str, Any] = {"limit": params.limit}
        if params.asset is not None:
            query["asset"] = params.asset
        if start_ms is not None:
            query["startTime"] = start_ms
        if end_ms is not None:
            query["endTime"] = end_ms
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/asset/assetDividend", auth="signed", params=query)
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        rows = data.get("rows") or [] if isinstance(data, dict) else []
        total = data.get("total") if isinstance(data, dict) else None
        lines = ["# Binance asset dividend record", ""]
        if not rows:
            lines.append("_No distributions in range._")
            return clip_response("\n".join(lines))
        shown = min(len(rows), MAX_DISPLAY_ROWS)
        if total is not None:
            lines.append(f"Showing **{shown:,}** of total **{total:,}** distribution(s).")
        else:
            lines.append(f"Showing **{shown:,}** distribution(s).")
        lines.append("")
        lines.append("| time | asset | amount | description | tranId |")
        lines.append("|---|---|---|---|---|")
        for row in rows[:MAX_DISPLAY_ROWS]:
            lines.append(
                f"| {epoch_to_human(row.get('divTime'))} | {row.get('asset', 'N/A')} | "
                f"{fmt_num(row.get('amount'))} | {row.get('enInfo', '')} | {row.get('tranId', 'N/A')} |"
            )
        if len(rows) > MAX_DISPLAY_ROWS:
            lines.append(
                f"_[{len(rows) - MAX_DISPLAY_ROWS} more distribution(s) not shown — raise `limit` "
                'or use response_format="json"]_'
            )
        totals: dict[str, Decimal] = {}
        for row in rows[:MAX_DISPLAY_ROWS]:
            asset = str(row.get("asset") or "?")
            totals[asset] = totals.get(asset, Decimal(0)) + _decimal_or_zero(row.get("amount"))
        if totals:
            rendered = ", ".join(f"{fmt_num(amount)} {asset}" for asset, amount in sorted(totals.items()))
            lines.append("")
            lines.append(f"_Total credited in the rows above: {rendered}._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_transfer_between_wallets",
    annotations=ToolAnnotations(
        title="Binance Universal Transfer (moves funds)",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def binance_transfer_between_wallets(params: TransferBetweenWalletsInput) -> str:
    """Move funds between the account's OWN wallets (Spot ⇄ Funding ⇄ Margin ⇄ Futures).

    Calls `POST /sapi/v1/asset/transfer` (SIGNED, **UID weight 300** of the 180000/min
    UID budget). The funds stay inside this Binance account: this is an internal move
    between wallets, never a transfer to another user and never a withdrawal off the
    platform. No tool in this server can send funds out of Binance.

    **Kill-switch.** This call is refused with `Error: … trading is disabled …` unless
    the server runs with `BINANCE_ALLOW_TRADING=1`. The gate lives in the HTTP client,
    so no tool can bypass it. If you see that error, the operator has deliberately put
    the server in read-only mode — report it, do not try to work around it.

    **Key permission.** The API key additionally needs the **"Permits Universal
    Transfer"** flag; without it Binance rejects the call even with the kill-switch on.
    `binance_get_api_restrictions` (wallet_account.py) reports it as
    `permitsUniversalTransfer`.

    The isolated-margin directions need the pair named — `from_symbol` for
    ISOLATEDMARGIN_MARGIN and ISOLATEDMARGIN_ISOLATEDMARGIN, `to_symbol` for
    MARGIN_ISOLATEDMARGIN and ISOLATEDMARGIN_ISOLATEDMARGIN. Both rules are checked
    locally, so a malformed transfer fails before anything is signed or sent.

    When to Use:
    - After a human has approved this specific movement of this specific amount.
    - To fund Binance Pay (MAIN_FUNDING) or to sweep Funding back to Spot (FUNDING_MAIN).

    When NOT to Use:
    - To send crypto to another exchange or wallet — this server never withdraws.
    - To swap one asset for another — that is the convert tools (convert.py) or a spot
      order (`binance_place_order`).
    - To check what a past transfer did — use `binance_get_transfer_history`.

    Returns:
    A confirmation echoing exactly what Binance returned, which is **only the `tranId`**.
    Binance sends no status field on this endpoint, so the confirmation says the transfer
    was accepted and points at `binance_get_transfer_history` /
    `binance_get_wallet_balances` to verify it settled. It never claims a balance changed.

    Examples:
        params = {"type": "MAIN_FUNDING", "asset": "USDT", "amount": "25.5"}
        params = {"type": "FUNDING_MAIN", "asset": "BNB", "amount": "0.1"}
        params = {"type": "MARGIN_ISOLATEDMARGIN", "asset": "USDT", "amount": "100",
                  "to_symbol": "BTCUSDT"}

    Error Handling:
    - `Error: … trading is disabled …` → the kill-switch is off; nothing was sent.
    - -2015 / "permission denied" → the key lacks "Permits Universal Transfer", or this
      IP is not allowlisted.
    - -3020 / insufficient balance → the source wallet does not hold the amount.
    - **A 5xx or a timeout means the transfer status is UNKNOWN** — it may have gone
      through. Check `binance_get_transfer_history` for the same type before retrying;
      never resend blindly.
    """
    try:
        query: dict[str, Any] = {
            "type": params.type.value,
            "asset": params.asset,
            "amount": params.amount,
        }
        if params.from_symbol is not None:
            query["fromSymbol"] = params.from_symbol
        if params.to_symbol is not None:
            query["toSymbol"] = params.to_symbol
        client = get_client()
        resp = await client.request("POST", "/sapi/v1/asset/transfer", auth="signed", params=query)
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope
        tran_id = data.get("tranId") if isinstance(data, dict) else None
        lines = [
            "# Transfer accepted",
            "",
            # The amount is echoed VERBATIM — the exact string that was signed and sent.
            # fmt_num would normalise it ('25.50000000' -> '25.5'), so the confirmation
            # would no longer show what actually went on the wire.
            f"Binance accepted a **{params.type.value}** transfer of **{params.amount} "
            f"{params.asset}** between your own wallets.",
            "",
            f"- **tranId**: {tran_id if tran_id is not None else 'not returned'}",
        ]
        if params.from_symbol is not None:
            lines.append(f"- **fromSymbol**: {params.from_symbol}")
        if params.to_symbol is not None:
            lines.append(f"- **toSymbol**: {params.to_symbol}")
        lines.append("")
        lines.append(
            "_Binance returns only a transaction id for this endpoint — no status and no resulting "
            "balance. Confirm it settled with `binance_get_transfer_history` (same `type`) or "
            "`binance_get_wallet_balances`._"
        )
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_convert_dust_to_bnb",
    annotations=ToolAnnotations(
        title="Binance Convert Dust to BNB (irreversible)",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def binance_convert_dust_to_bnb(params: ConvertDustInput) -> str:
    """Convert small balances to BNB. **Irreversible** — the assets are sold for BNB.

    Calls `POST /sapi/v1/asset/dust` (SIGNED, UID weight 10). Every asset listed is
    swapped to BNB at Binance's dust rate, minus a service charge; there is no undo and
    no "cancel" endpoint.

    **Preview first.** Run `binance_get_dust_convertible` and pass only assets it
    listed: it names what qualifies and how much BNB each one yields, and it works even
    with the kill-switch off.

    **Kill-switch.** This call is refused with `Error: … trading is disabled …` unless
    the server runs with `BINANCE_ALLOW_TRADING=1`. The gate lives in the HTTP client,
    so no tool can bypass it. If you see that error, the operator has deliberately put
    the server in read-only mode — report it, do not try to work around it.

    **Key permission.** The API key needs **"Enable Spot & Margin Trading"**; this is a
    trade, not a transfer.

    At most **100 assets per call** — a client-side guard, not a Binance limit (Binance
    documents no cap). Split a longer list into batches; each batch is its own
    irreversible conversion.

    When to Use:
    - After a human approved converting these specific assets, and after the preview
      confirmed they qualify.

    When NOT to Use:
    - To swap a meaningful amount of one asset for another — the dust rate is worse than
      the market; use the convert tools (convert.py) or a spot order instead.
    - To see what happened in past conversions — use `binance_get_dust_log`.

    Returns:
    A confirmation echoing exactly what Binance returned: `totalTransfered` (BNB
    received), `totalServiceCharge`, and the per-asset `transferResult` rows with their
    tranIds. Nothing is inferred; an asset Binance silently skipped simply will not
    appear in the table.

    Examples:
        params = {"assets": ["ADA"]}
        params = {"assets": ["ADA", "DOT", "XRP"], "account_type": "SPOT"}

    Error Handling:
    - `Error: … trading is disabled …` → the kill-switch is off; nothing was converted.
    - -2015 means the key lacks Spot & Margin Trading permission, or this IP is not
      allowlisted.
    - "The asset does not have a dust balance" family of errors → re-run
      `binance_get_dust_convertible`; eligibility changes with price.
    - **A 5xx or a timeout means the conversion status is UNKNOWN** — check
      `binance_get_dust_log` before retrying; a duplicate conversion cannot be undone.
    """
    try:
        query: dict[str, Any] = {"asset": Repeat(params.assets)}
        if params.account_type is not None:
            query["accountType"] = params.account_type.value
        client = get_client()
        resp = await client.request("POST", "/sapi/v1/asset/dust", auth="signed", params=query)
        data = resp.json()
        if (envelope := _envelope_error(data)) is not None:
            return envelope
        body = data if isinstance(data, dict) else {}
        results = body.get("transferResult") or []
        lines = [
            "# Dust converted to BNB",
            "",
            f"Requested: **{', '.join(params.assets)}**. This cannot be undone.",
            "",
            f"- **BNB received (totalTransfered)**: {fmt_num(body.get('totalTransfered'))}",
            f"- **service charge (totalServiceCharge)**: {fmt_num(body.get('totalServiceCharge'))}",
        ]
        if results:
            lines.append("")
            lines.append("| from asset | amount | BNB received | service charge | time | tranId |")
            lines.append("|---|---|---|---|---|---|")
            for row in results[:MAX_DISPLAY_ROWS]:
                lines.append(
                    f"| {row.get('fromAsset', 'N/A')} | {fmt_num(row.get('amount'))} | "
                    f"{fmt_num(row.get('transferedAmount'))} | {fmt_num(row.get('serviceChargeAmount'))} | "
                    f"{epoch_to_human(row.get('operateTime'))} | {row.get('tranId', 'N/A')} |"
                )
            if len(results) > MAX_DISPLAY_ROWS:
                lines.append(f"_[{len(results) - MAX_DISPLAY_ROWS} more row(s) not shown]_")
        else:
            lines.append("")
            lines.append("_Binance returned no per-asset results for this conversion._")
        lines.append("")
        lines.append("_Cross-check the batch later with `binance_get_dust_log`._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)
