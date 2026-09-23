"""Spot TWAP algo orders (inventory J) — `/sapi/v1/algo/spot` endpoints.

Five tools around one product: a Time-Weighted Average Price order, which Binance
executes as a stream of sub-orders over a duration you choose instead of hitting the
book in one go.

- `binance_place_twap_order`      POST   /sapi/v1/algo/spot/newOrderTwap  (**real money**)
- `binance_cancel_algo_order`     DELETE /sapi/v1/algo/spot/order
- `binance_get_open_algo_orders`  GET    /sapi/v1/algo/spot/openOrders
- `binance_get_algo_order_history` GET   /sapi/v1/algo/spot/historicalOrders
- `binance_get_algo_sub_orders`   GET    /sapi/v1/algo/spot/subOrders

**Futures algo is OUT of scope.** `/sapi/v1/algo/futures/newOrderTwap` and `newOrderVp`
are a different product on a different wallet (USDⓈ-M, `positionSide`, its own key
permission and a 30-order cap), and so is the `/fapi/v1/algoOrder` conditional-order
family. This server is Spot + Wallet only; none of those endpoints is callable here.

Every call is SIGNED. The two mutating ones pass through the **kill-switch that lives in
`binance_mcp.client`**, not here: a signed non-GET request is refused with
`TradingDisabledError` (surfaced as `Error: … trading is disabled …`) unless
`BINANCE_ALLOW_TRADING=1`.

Quantities and prices are **strings** end to end — validated as positive decimals and
forwarded verbatim, because Binance's LOT_SIZE / PRICE_FILTER / NOTIONAL filters are
precision-sensitive and a float round-trip silently breaks them.

Two things about this API that the renderers here take seriously:

- `success: true` on a placement means **accepted, not executed**. The only way to learn
  what actually traded is to poll `openOrders` / `historicalOrders` / `subOrders`.
- These `/sapi` endpoints can answer **HTTP 200 with a rejection** in the body
  (`success: false`, non-200 `code`). That is checked on every response and rendered as
  an `Error:` line rather than being mistaken for a confirmation.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from binance_mcp.client import get_client
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import ResponseFormat, clip_response, epoch_to_human, fmt_num, to_json
from binance_mcp.server import mcp

# Context-window guard (rule 8): cap displayed rows on top of the API's own pageSize.
MAX_DISPLAY_ROWS = 50

# Binance refuses a TWAP shorter than 5 minutes or longer than 24 hours (S3 / research 03 §6).
MIN_DURATION_SECONDS = 300
MAX_DURATION_SECONDS = 86_400

# A `clientAlgoId`, when supplied, must be exactly this long (S3).
CLIENT_ALGO_ID_LENGTH = 32

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,20}$")


# -- enums ---------------------------------------------------------------------------


class AlgoSide(StrEnum):
    """Which way the TWAP goes."""

    BUY = "BUY"
    SELL = "SELL"


# -- helpers -------------------------------------------------------------------------


def _normalize_symbol(value: str) -> str:
    """Uppercase a trading pair and reject anything that is not a Binance symbol."""
    symbol = value.strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"symbol must be a Binance trading pair such as 'BTCUSDT' (2-20 of A-Z0-9); got {value!r}.")
    return symbol


def _to_ms(value: int | str | None, field: str) -> int | None:
    """Accept an epoch-ms int or an ISO-8601 string on the tool surface; send ms.

    A numeric string counts as epoch-ms only with at least 12 digits (a real ms
    timestamp is 13 today); anything shorter is parsed as ISO-8601 so a typo surfaces
    as an error instead of becoming a bogus timestamp.
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
    """Return an `Error:` line when a 200 body is actually a rejection, else None.

    `/sapi` answers some failures with HTTP 200 and `{"success": false, "code": …,
    "msg": …}`. The client raises `BinanceEnvelopeError` for the `success: false` shape
    on a real response; this second check costs nothing, covers the `code`-only variant
    (which the client does not see as a failure), and guarantees that no renderer in this
    module can turn a rejection into a confirmation.

    The algo endpoints spell success as `code: 0, msg: "OK"` (other `/sapi` families use
    200), so both count as "no error"; every Binance failure code is negative.
    """
    if not isinstance(data, dict):
        return None
    code = data.get("code")
    rejected = "success" in data and not data["success"]
    if rejected or code not in (None, 0, 200):
        message = data.get("msg") or data.get("message") or "Binance rejected the request"
        return f"Error: {message} (code {code})"
    return None


def _single_algo_id(algo_id: int | None, client_algo_id: str | None) -> dict[str, Any]:
    """Return the id param for a cancel — exactly one id, checked here.

    Binance accepts either; sending both leaves it to the server to decide which one
    wins, and on a destructive call an ambiguous request is worse than a refused one.
    """
    if (algo_id is None) == (client_algo_id is None):
        raise RuntimeError(
            "Pass exactly one of algo_id (Binance: algoId) or client_algo_id (Binance: clientAlgoId) — "
            "not both, not neither. Sending both would leave it to Binance to choose which order to "
            "cancel, and a cancel is not a call to guess on."
        )
    if algo_id is not None:
        return {"algoId": algo_id}
    return {"clientAlgoId": client_algo_id}


def _algo_order_rows(orders: list[dict[str, Any]]) -> list[str]:
    """Markdown table of algo orders — the same shape for open and historical."""
    lines = [
        "| bookTime | algoId | symbol | side | algoStatus | algoType | totalQty | executedQty | "
        "executedAmt | avgPrice | urgency | endTime |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for order in orders[:MAX_DISPLAY_ROWS]:
        lines.append(
            f"| {epoch_to_human(order.get('bookTime'))} | {order.get('algoId', 'N/A')} | "
            f"{order.get('symbol', 'N/A')} | {order.get('side', 'N/A')} | "
            f"{order.get('algoStatus', 'N/A')} | {order.get('algoType', 'N/A')} | "
            f"{fmt_num(order.get('totalQty'))} | {fmt_num(order.get('executedQty'))} | "
            f"{fmt_num(order.get('executedAmt'))} | {fmt_num(order.get('avgPrice'))} | "
            f"{order.get('urgency', 'N/A')} | {epoch_to_human(order.get('endTime'))} |"
        )
    if len(orders) > MAX_DISPLAY_ROWS:
        lines.append("")
        lines.append(
            f"_[{len(orders) - MAX_DISPLAY_ROWS} more algo order(s) not shown — narrow the query or use "
            'response_format="json"]_'
        )
    return lines


def _algo_order_table(orders: list[dict[str, Any]], title: str, total: Any) -> list[str]:
    """Render `{total, orders[]}` — the response shape of openOrders and historicalOrders."""
    lines = [f"# {title}", ""]
    if not orders:
        lines.append("_No algo orders._")
        return lines
    shown = min(len(orders), MAX_DISPLAY_ROWS)
    reported = f" (Binance reports **{total:,}** in total)" if isinstance(total, int) else ""
    lines.append(f"Showing **{shown:,}** of **{len(orders):,}** algo order(s) on this page{reported}.")
    lines.append("")
    lines.extend(_algo_order_rows(orders))
    lines.append("")
    lines.append(
        '_`clientAlgoId` is not in the table — read it with `response_format="json"`. Drill into the '
        "individual fills of one order with `binance_get_algo_sub_orders`._"
    )
    return lines


def _sub_order_table(data: dict[str, Any], algo_id: int) -> list[str]:
    """Render `{total, executedQty, executedAmt, subOrders[]}` for one algo order."""
    sub_orders = data.get("subOrders") or []
    lines = [f"# Sub-orders of algo order {algo_id}", ""]
    total = data.get("total")
    if isinstance(total, int):
        lines.append(f"- **total sub-orders**: {total:,}")
    lines.append(f"- **executedQty**: {fmt_num(data.get('executedQty'))}")
    lines.append(f"- **executedAmt**: {fmt_num(data.get('executedAmt'))}")
    lines.append("")
    if not sub_orders:
        lines.append("_No sub-orders yet — the TWAP has not traded._")
        return lines
    lines.append(f"Showing **{min(len(sub_orders), MAX_DISPLAY_ROWS):,}** of **{len(sub_orders):,}** sub-order(s).")
    lines.append("")
    lines.append(
        "| bookTime | subId | orderId | symbol | side | orderStatus | executedQty | executedAmt | avgPrice | fee |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for sub in sub_orders[:MAX_DISPLAY_ROWS]:
        fee = f"{fmt_num(sub.get('feeAmt'))} {sub.get('feeAsset', '')}".strip()
        lines.append(
            f"| {epoch_to_human(sub.get('bookTime'))} | {sub.get('subId', 'N/A')} | "
            f"{sub.get('orderId', 'N/A')} | {sub.get('symbol', 'N/A')} | {sub.get('side', 'N/A')} | "
            f"{sub.get('orderStatus', 'N/A')} | {fmt_num(sub.get('executedQty'))} | "
            f"{fmt_num(sub.get('executedAmt'))} | {fmt_num(sub.get('avgPrice'))} | {fee} |"
        )
    if len(sub_orders) > MAX_DISPLAY_ROWS:
        lines.append("")
        lines.append(
            f"_[{len(sub_orders) - MAX_DISPLAY_ROWS} more sub-order(s) not shown — raise `page` or use "
            'response_format="json"]_'
        )
    return lines


# -- input models --------------------------------------------------------------------


class _BaseInput(BaseModel):
    """Shared pydantic config for every input model in this module."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class _PagedInput(_BaseInput):
    """The `page` / `pageSize` pair shared by the two paged read endpoints."""

    page: int = Field(default=1, ge=1, description="1-based page number (Binance: page, default 1).")
    page_size: int = Field(
        default=100,
        ge=1,
        le=100,
        description="Rows per page, 1-100 (Binance: pageSize, default 100).",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: `markdown` (human-readable) or `json` (raw Binance payload).",
    )


class PlaceTwapOrderInput(_BaseInput):
    """Params for `POST /sapi/v1/algo/spot/newOrderTwap` — this spends real money."""

    symbol: str = Field(description="Trading pair, e.g. `BTCUSDT` (uppercased automatically).")
    side: AlgoSide = Field(description="BUY or SELL.")
    quantity: str = Field(
        description=(
            "Total base-asset amount to work through the TWAP, as a decimal STRING "
            "(e.g. '0.5' BTC). Sent verbatim. Binance requires roughly 1,000 USDT of notional "
            "as a minimum for an algo order."
        )
    )
    duration: int = Field(
        ge=MIN_DURATION_SECONDS,
        le=MAX_DURATION_SECONDS,
        description=(
            "How long Binance spreads the execution over, in SECONDS. Minimum 300 (5 minutes), "
            "maximum 86400 (24 hours)."
        ),
    )
    client_algo_id: str | None = Field(
        default=None,
        min_length=CLIENT_ALGO_ID_LENGTH,
        max_length=CLIENT_ALGO_ID_LENGTH,
        description=(
            "Your own id for this algo order (Binance: clientAlgoId). Must be EXACTLY 32 characters; "
            "Binance generates one when it is omitted. Set it to make a retry traceable."
        ),
    )
    limit_price: str | None = Field(
        default=None,
        description=(
            "Optional worst acceptable price as a decimal STRING (Binance: limitPrice). Sub-orders are "
            "placed as LIMIT orders at this price; omit it and Binance works the order at market."
        ),
    )

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @field_validator("quantity", "limit_price")
    @classmethod
    def _check_positive_decimal(cls, value: str | None, info: ValidationInfo) -> str | None:
        """Validate as a positive decimal but return the ORIGINAL string, unrounded."""
        if value is None:
            return None
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(
                f"{info.field_name} must be a decimal number sent as a string (e.g. '0.001'); got {value!r}."
            ) from exc
        if not parsed.is_finite() or parsed <= 0:
            raise ValueError(f"{info.field_name} must be a positive, finite decimal; got {value!r}.")
        return value

    def to_params(self) -> dict[str, Any]:
        """Map to Binance's camelCase query parameters, dropping everything unset."""
        raw: dict[str, Any] = {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "duration": self.duration,
            "clientAlgoId": self.client_algo_id,
            "limitPrice": self.limit_price,
        }
        return {key: value for key, value in raw.items() if value is not None}


class CancelAlgoOrderInput(_BaseInput):
    """Params for `DELETE /sapi/v1/algo/spot/order`."""

    algo_id: int | None = Field(
        default=None,
        description="Binance algoId of the algo order to cancel. Pass this OR client_algo_id.",
    )
    client_algo_id: str | None = Field(
        default=None,
        min_length=1,
        description="The clientAlgoId you sent when placing. Pass this OR algo_id. Binance: clientAlgoId.",
    )


class GetOpenAlgoOrdersInput(_BaseInput):
    """Params for `GET /sapi/v1/algo/spot/openOrders` — no filters, all symbols."""

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: `markdown` (human-readable) or `json` (raw Binance payload).",
    )


class GetAlgoOrderHistoryInput(_PagedInput):
    """Params for `GET /sapi/v1/algo/spot/historicalOrders`."""

    symbol: str | None = Field(
        default=None,
        description="Trading pair filter, e.g. `BTCUSDT`. Optional — omit for every symbol.",
    )
    side: AlgoSide | None = Field(default=None, description="BUY or SELL. Optional — omit for both.")
    start_time: int | str | None = Field(
        default=None,
        description="Window start: epoch ms or ISO-8601 (e.g. '2026-09-01'). Optional.",
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end: epoch ms or ISO-8601. Optional.",
    )

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, value: str | None) -> str | None:
        return _normalize_symbol(value) if value is not None else None


class GetAlgoSubOrdersInput(_PagedInput):
    """Params for `GET /sapi/v1/algo/spot/subOrders`."""

    algo_id: int = Field(description="Binance algoId of the algo order whose fills you want (mandatory).")


# -- tools ---------------------------------------------------------------------------


@mcp.tool(
    name="binance_place_twap_order",
    annotations=ToolAnnotations(
        title="Binance Place Spot TWAP Order",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def binance_place_twap_order(params: PlaceTwapOrderInput) -> str:
    """Place a REAL spot TWAP algo order on Binance. This spends real money.

    Calls `POST /sapi/v1/algo/spot/newOrderTwap` (SIGNED, **UID weight 3000** of a
    180,000/min budget — call it sparingly). Binance splits `quantity` into sub-orders and
    works them over `duration` seconds, aiming at the time-weighted average price instead
    of taking the book in one hit.

    **Kill-switch.** This call is refused with `Error: … trading is disabled …` unless the
    server runs with `BINANCE_ALLOW_TRADING=1`. The gate lives in the HTTP client, so no
    tool can bypass it. If you see that error, the operator has deliberately put the
    server in read-only mode — report it, do not try to work around it.

    Binance's own constraints, all rejected server-side if broken:
    - `duration` 300-86400 seconds (5 minutes to 24 hours) — checked locally too.
    - **Minimum notional ≈ 1,000 USDT equivalent** per algo order (the docs also quote a
      per-symbol maximum of 200k / 2mm / 10mm; let the API arbitrate the ceiling).
    - **At most 20 open algo orders** at a time — check with `binance_get_open_algo_orders`
      before adding another.
    - `client_algo_id`, when supplied, must be exactly 32 characters.

    **`success: true` means ACCEPTED, NOT EXECUTED.** The response carries no fill
    information at all; it only says Binance took the order. What actually traded is
    visible through `binance_get_open_algo_orders`, `binance_get_algo_order_history` and
    `binance_get_algo_sub_orders`.

    When to Use:
    - Working a position that is large relative to the book, where a single MARKET order
      would move the price against you.
    - Spreading an entry or exit over minutes or hours on purpose.

    When NOT to Use:
    - For an ordinary immediate or resting order — use `binance_place_order`
      (spot_orders.py); it is weight 1, not 3000, and has no notional floor.
    - Below ~1,000 USDT of notional — the API rejects it; place a normal order instead.
    - For USDⓈ-M / COIN-M futures TWAP or VP — that is a different product on a different
      wallet and this server does not implement it.

    Returns:
    A confirmation echoing exactly the four fields Binance returned — `clientAlgoId`,
    `success`, `code`, `msg` — plus an explicit note that the order is accepted and not
    executed, and which tool to poll. A `success: false` body (Binance answers those with
    HTTP 200) is rendered as `Error: <msg> (code <code>)`, never as a confirmation.

    Examples:
        params = {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.5", "duration": 3600}
        params = {"symbol": "BTCUSDT", "side": "SELL", "quantity": "1.25", "duration": 7200,
                  "limit_price": "65000.00",
                  "client_algo_id": "abcdefghijklmnopqrstuvwxyz012345"}

    Error Handling:
    - `Error: … trading is disabled …` → the kill-switch is off; nothing was sent.
    - A duration outside 300-86400 or a 31-character `client_algo_id` fails locally,
      before anything is signed.
    - -2010 / -1013 point at balance, the notional floor or a symbol filter; -2015 means
      the key lacks Spot & Margin Trading permission or this IP is not allowlisted.
    - **A 5xx or a timeout means the execution status is UNKNOWN** — the algo order may
      well be live. Check `binance_get_open_algo_orders` (and
      `binance_get_algo_order_history`) before doing anything else. NEVER resend blindly:
      a duplicate TWAP is a second position, and each one eats one of the 20 slots.
    """
    try:
        client = get_client()
        resp = await client.request("POST", "/sapi/v1/algo/spot/newOrderTwap", auth="signed", params=params.to_params())
        data = resp.json()
        if (error := _envelope_error(data)) is not None:
            return error
        lines = [
            f"# TWAP order accepted on {params.symbol}",
            "",
            f"- **clientAlgoId**: {data.get('clientAlgoId', 'N/A')}",
            f"- **success**: {data.get('success', 'N/A')}",
            f"- **code**: {data.get('code', 'N/A')}",
            f"- **msg**: {data.get('msg', 'N/A')}",
            "",
            f"Binance accepted a **{params.side.value} {params.quantity}** TWAP on **{params.symbol}** "
            f"to be worked over **{params.duration:,} seconds**.",
            "",
            "_The order is **accepted, not executed** — this response says nothing about fills. Poll "
            "`binance_get_open_algo_orders` / `binance_get_algo_order_history` for its status, and "
            "`binance_get_algo_sub_orders` for the individual fills._",
        ]
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_cancel_algo_order",
    annotations=ToolAnnotations(
        title="Binance Cancel Algo Order",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_cancel_algo_order(params: CancelAlgoOrderInput) -> str:
    """Cancel a working spot TWAP algo order.

    Calls `DELETE /sapi/v1/algo/spot/order` (SIGNED, IP weight 1). Pass exactly one id:
    `algo_id` (Binance's numeric algoId) or `client_algo_id` (the 32-character id you
    supplied when placing). Both at once is refused locally — on a destructive call an
    ambiguous request is worse than a refused one.

    **Kill-switch.** Refused with `Error: … trading is disabled …` unless the server runs
    with `BINANCE_ALLOW_TRADING=1`.

    Cancelling is idempotent in effect: a second cancel of the same algo order changes
    nothing and comes back as a rejection. What it cannot undo is what already traded —
    the sub-orders already filled stay filled, and only the unexecuted remainder is
    called off. Read `binance_get_algo_sub_orders` to see what that remainder is.

    When to Use:
    - Stopping a TWAP whose thesis no longer holds, mid-execution.
    - Freeing one of the 20 open-algo-order slots.

    When NOT to Use:
    - To cancel an ordinary spot order — that is `binance_cancel_order` (spot_orders.py);
      these endpoints do not see each other's orders.
    - Before checking what is open — `binance_get_open_algo_orders` is free of
      consequence and gives you the algoId this tool needs.

    Returns:
    A confirmation echoing exactly the three fields Binance returned — `algoId`,
    `success`, `code`, `msg` — with no claim about how much of the order had executed.
    A `success: false` body (HTTP 200) is rendered as `Error: <msg> (code <code>)`.

    Examples:
        params = {"algo_id": 14511}
        params = {"client_algo_id": "abcdefghijklmnopqrstuvwxyz012345"}

    Error Handling:
    A rejection usually means the algo order is not cancellable: already finished, already
    cancelled, or the id belongs to another account. -2015 means the key lacks Spot &
    Margin Trading permission or this IP is not allowlisted. A 5xx/timeout leaves the
    cancel UNKNOWN — re-read `binance_get_open_algo_orders` before assuming either way.
    """
    try:
        query = _single_algo_id(params.algo_id, params.client_algo_id)
        client = get_client()
        resp = await client.request("DELETE", "/sapi/v1/algo/spot/order", auth="signed", params=query)
        data = resp.json()
        if (error := _envelope_error(data)) is not None:
            return error
        lines = [
            "# Algo order cancelled",
            "",
            f"- **algoId**: {data.get('algoId', 'N/A')}",
            f"- **success**: {data.get('success', 'N/A')}",
            f"- **code**: {data.get('code', 'N/A')}",
            f"- **msg**: {data.get('msg', 'N/A')}",
            "",
            "_Only the UNEXECUTED remainder was called off — sub-orders that already filled are "
            "unaffected. See `binance_get_algo_sub_orders` for what traded before the cancel._",
        ]
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_open_algo_orders",
    annotations=ToolAnnotations(
        title="Binance Open Algo Orders",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_open_algo_orders(params: GetOpenAlgoOrdersInput) -> str:
    """List the spot TWAP algo orders that are still working, across every symbol.

    Calls `GET /sapi/v1/algo/spot/openOrders` (SIGNED, IP weight 1). It takes no filters:
    one call returns everything currently running, which is also how you check the
    **20 open algo orders** ceiling before placing another.

    When to Use:
    - Right after `binance_place_twap_order`, to confirm the order is actually working —
      the placement response only says "accepted".
    - Before placing a new TWAP, to see how many of the 20 slots are free.
    - After a 5xx/timeout on a placement, to find out whether the order exists.

    When NOT to Use:
    - For orders that have finished — use `binance_get_algo_order_history`.
    - For the fills of one order — use `binance_get_algo_sub_orders`.
    - For ordinary spot orders — use `binance_get_open_orders` (spot_orders.py); algo and
      ordinary orders live in separate endpoints and neither lists the other.

    Returns:
    A markdown table (bookTime, algoId, symbol, side, algoStatus, algoType, totalQty,
    executedQty, executedAmt, avgPrice, urgency, endTime) capped at 50 rows, or the raw
    `{total, orders[]}` payload with `response_format="json"` (which also carries
    `clientAlgoId`).

    Examples:
        params = {}
        params = {"response_format": "json"}

    Error Handling:
    An empty list is a valid answer: nothing is working. -2015 means the key lacks Reading
    permission or this IP is not allowlisted. `/sapi` does not exist on the spot testnet —
    a 404 there is expected, not a bug.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/algo/spot/openOrders", auth="signed", params={})
        data = resp.json()
        if (error := _envelope_error(data)) is not None:
            return error
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        orders = data.get("orders") or [] if isinstance(data, dict) else []
        total = data.get("total") if isinstance(data, dict) else None
        return clip_response("\n".join(_algo_order_table(orders, "Open spot algo orders", total)))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_algo_order_history",
    annotations=ToolAnnotations(
        title="Binance Algo Order History",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_algo_order_history(params: GetAlgoOrderHistoryInput) -> str:
    """List finished spot TWAP algo orders — filled, cancelled or expired.

    Calls `GET /sapi/v1/algo/spot/historicalOrders` (SIGNED, IP weight 1). Every filter is
    optional: `symbol`, `side`, `start_time`, `end_time`. Binance's older reference marks
    symbol and side as mandatory; the current per-endpoint page makes both optional, which
    is what this tool follows — omit them for the whole account.

    No maximum window is documented for this endpoint, so none is enforced here.

    When to Use:
    - Reviewing how a TWAP actually executed once it is no longer open.
    - Reconciling a period: what algo orders ran, on what symbols, for how much.

    When NOT to Use:
    - For orders still working — `binance_get_open_algo_orders`.
    - For the individual fills and fees of one order — `binance_get_algo_sub_orders`;
      a row here only carries the aggregate.
    - For ordinary spot order history — `binance_get_all_orders` (spot_orders.py).

    Pagination:
    `page` (1-based) and `page_size` (1-100, default 100) map to Binance's `page` /
    `pageSize`. At most 50 rows are rendered; raise `page` or use
    `response_format="json"` for the rest.

    Examples:
        params = {}
        params = {"symbol": "BTCUSDT", "side": "BUY", "page_size": 20}
        params = {"start_time": "2026-09-01", "end_time": "2026-09-23T23:59:59Z"}

    Error Handling:
    `start_time` / `end_time` accept epoch ms or ISO-8601; anything else fails locally
    with a readable message. -2015 means the key lacks Reading permission or this IP is
    not allowlisted. `/sapi` does not exist on the spot testnet.
    """
    try:
        start_ms = _to_ms(params.start_time, "start_time")
        end_ms = _to_ms(params.end_time, "end_time")
        if start_ms is not None and end_ms is not None and end_ms <= start_ms:
            return "Error: end_time must be after start_time."
        query: dict[str, Any] = {"page": params.page, "pageSize": params.page_size}
        if params.symbol is not None:
            query["symbol"] = params.symbol
        if params.side is not None:
            query["side"] = params.side.value
        if start_ms is not None:
            query["startTime"] = start_ms
        if end_ms is not None:
            query["endTime"] = end_ms
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/algo/spot/historicalOrders", auth="signed", params=query)
        data = resp.json()
        if (error := _envelope_error(data)) is not None:
            return error
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        orders = data.get("orders") or [] if isinstance(data, dict) else []
        total = data.get("total") if isinstance(data, dict) else None
        scope = f" on {params.symbol}" if params.symbol else ""
        title = f"Historical spot algo orders{scope} (page {params.page})"
        return clip_response("\n".join(_algo_order_table(orders, title, total)))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_algo_sub_orders",
    annotations=ToolAnnotations(
        title="Binance Algo Sub-Orders",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_algo_sub_orders(params: GetAlgoSubOrdersInput) -> str:
    """List the individual orders a TWAP placed on the book, with fills and fees.

    Calls `GET /sapi/v1/algo/spot/subOrders` (SIGNED, IP weight 1). A TWAP is executed as
    a stream of ordinary orders; this is the only place to see them — what each slice
    filled, at what average price, and what it cost in fees.

    When to Use:
    - Answering "what did this TWAP actually get me?" — the executedQty / executedAmt /
      fee totals for one algoId.
    - Auditing a cancelled TWAP: what traded before the cancel landed.

    When NOT to Use:
    - To find the algoId in the first place — that comes from
      `binance_get_open_algo_orders` or `binance_get_algo_order_history`.
    - For a symbol-wide trade list — `binance_get_my_trades` (trade_history.py) covers
      every fill, algo or not.

    Pagination:
    `page` (1-based) and `page_size` (1-100, default 100) map to Binance's `page` /
    `pageSize`. At most 50 sub-orders are rendered; raise `page` or use
    `response_format="json"` for the rest.

    Returns:
    The order-level totals (`total`, `executedQty`, `executedAmt`) followed by a table of
    sub-orders: bookTime, subId, orderId, symbol, side, orderStatus, executedQty,
    executedAmt, avgPrice and the fee with its asset.

    Examples:
        params = {"algo_id": 14511}
        params = {"algo_id": 14511, "page": 2, "page_size": 50}

    Error Handling:
    An unknown algoId comes back as a Binance rejection, not an empty page. An empty
    `subOrders` array means the TWAP has not traded yet. -2015 means the key lacks Reading
    permission or this IP is not allowlisted. `/sapi` does not exist on the spot testnet.
    """
    try:
        query: dict[str, Any] = {
            "algoId": params.algo_id,
            "page": params.page,
            "pageSize": params.page_size,
        }
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/algo/spot/subOrders", auth="signed", params=query)
        data = resp.json()
        if (error := _envelope_error(data)) is not None:
            return error
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        payload = data if isinstance(data, dict) else {}
        return clip_response("\n".join(_sub_order_table(payload, params.algo_id)))
    except Exception as exc:
        return handle_api_error(exc)
