"""Order lists — OCO / OTO / OTOCO (inventory C) — the `/api/v3/orderList*` endpoints.

Seven tools:

- `binance_place_oco_order`      POST   /api/v3/orderList/oco    (**real money**)
- `binance_place_oto_order`      POST   /api/v3/orderList/oto    (**real money**)
- `binance_place_otoco_order`    POST   /api/v3/orderList/otoco  (**real money**)
- `binance_get_order_list`       GET    /api/v3/orderList
- `binance_cancel_order_list`    DELETE /api/v3/orderList
- `binance_get_all_order_lists`  GET    /api/v3/allOrderList
- `binance_get_open_order_lists` GET    /api/v3/openOrderList

An order list is several orders placed and cancelled as a unit:

- **OCO** (one-cancels-the-other): two legs, the `above` one and the `below` one. One is
  the take-profit side (LIMIT_MAKER / TAKE_PROFIT / TAKE_PROFIT_LIMIT), the other the
  stop side (STOP_LOSS / STOP_LOSS_LIMIT); whichever triggers first cancels the other.
- **OTO** (one-triggers-the-other): a `working` leg (LIMIT or LIMIT_MAKER) that rests on
  the book, and a `pending` leg that is only placed once the working leg is **fully
  filled**.
- **OTOCO**: the same working leg, with an OCO pair as the pending side.

Every call is SIGNED. The three placements and the cancel additionally pass through the
**kill-switch that lives in `binance_mcp.client`**, not here: a signed non-GET request is
refused with `TradingDisabledError` (surfaced as `Error: … trading is disabled …`) unless
`BINANCE_ALLOW_TRADING=1`. **There is no dry-run for order lists** — `POST
/api/v3/order/test` validates a single order, not a list — so the local validation below
is the only pre-flight there is, and `binance_test_order` on one leg at a time is the
closest available approximation.

Prices and quantities are **strings** end to end. They are validated as positive decimals
and forwarded verbatim: Binance's LOT_SIZE / PRICE_FILTER / NOTIONAL filters are
precision-sensitive and a float round-trip (0.1 + 0.2) silently breaks them.

Out of scope here: the pegged variants (`POST /api/v3/orderList/opo` and `/opoco`) and the
`*PegPriceType` / `*PegOffsetType` / `*PegOffsetValue` parameters, and the deprecated
legacy `POST /api/v3/order/oco`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Self

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from binance_mcp.client import get_client
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import ResponseFormat, clip_response, epoch_to_human, fmt_num, to_json
from binance_mcp.server import mcp

# Context-window guard (rule 8): cap displayed rows on top of the API's own `limit`.
MAX_DISPLAY_ROWS = 50

# `allOrderList` rejects a start/end span of more than 24 h (inventory C / S2 L4455);
# checked before the call so the caller gets a clear message, not Binance's -1127.
_ORDER_LIST_WINDOW_MS = 24 * 60 * 60 * 1000

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,20}$")

# A list whose legs are all dead, or whose listOrderStatus says so, is NOT working.
_NOT_LIVE_LIST_STATUSES = ("ALL_DONE", "REJECT")
_NOT_LIVE_ORDER_STATUSES = ("EXPIRED", "EXPIRED_IN_MATCH", "REJECTED")


# -- enums ---------------------------------------------------------------------------


class OrderSide(StrEnum):
    """Which way the order list goes; both legs share it."""

    BUY = "BUY"
    SELL = "SELL"


class OrderListLegType(StrEnum):
    """The five order types an OCO leg may take (S2 L3256 / L3268, L3538 / L3550)."""

    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"


class WorkingOrderType(StrEnum):
    """The working leg of an OTO/OTOCO must rest on the book: LIMIT or LIMIT_MAKER."""

    LIMIT = "LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"


class PendingOrderType(StrEnum):
    """The pending leg of an OTO can be any type (MARKET with quoteOrderQty excepted)."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"


class TimeInForce(StrEnum):
    """How long a leg stays on the book: GTC rests, IOC/FOK do not."""

    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class NewOrderRespType(StrEnum):
    """How much of the result Binance returns (FULL includes the fills)."""

    ACK = "ACK"
    RESULT = "RESULT"
    FULL = "FULL"


class SelfTradePreventionMode(StrEnum):
    """What Binance does when a leg would match the account's own resting order."""

    NONE = "NONE"
    EXPIRE_TAKER = "EXPIRE_TAKER"
    EXPIRE_MAKER = "EXPIRE_MAKER"
    EXPIRE_BOTH = "EXPIRE_BOTH"
    DECREMENT = "DECREMENT"


# The take-profit family sits on the profitable side of the market, the stop family on the
# losing side. An OCO pair needs exactly one of each (S2 L3232).
_PROFIT_TYPES = (
    OrderListLegType.LIMIT_MAKER,
    OrderListLegType.TAKE_PROFIT,
    OrderListLegType.TAKE_PROFIT_LIMIT,
)


# -- helpers -------------------------------------------------------------------------


def _as_str(value: Any) -> str:
    return str(value)


def _enum_value(value: StrEnum | None) -> str | None:
    """Unwrap an optional enum for the query string."""
    return value.value if value is not None else None


def _normalize_symbol(value: str) -> str:
    """Uppercase a trading pair and reject anything that is not a Binance symbol."""
    symbol = value.strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"symbol must be a Binance trading pair such as 'BTCUSDT' (2-20 of A-Z0-9); got {value!r}.")
    return symbol


def _positive_decimal(value: str | None, info: ValidationInfo) -> str | None:
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


def _api_name(prefix: str, suffix: str) -> str:
    """`("pending_above", "stop_price")` → `pendingAboveStopPrice` (Binance's spelling)."""
    head, *rest = prefix.split("_")
    camel_prefix = head + "".join(word.capitalize() for word in rest)
    return camel_prefix + "".join(word.capitalize() for word in suffix.split("_"))


def _require_leg_fields(prefix: str, leg_type: str, required: tuple[tuple[str, Any], ...]) -> None:
    """Raise a readable ValueError naming every missing field of a leg's mandatory set."""
    missing = [suffix for suffix, value in required if value is None]
    if missing:
        raise ValueError(
            f"{prefix}_type={leg_type} requires {', '.join(f'{prefix}_{name}' for name in missing)} "
            f"(Binance: {', '.join(_api_name(prefix, name) for name in missing)})."
        )


def _require_trigger(prefix: str, leg_type: str, stop_price: str | None, trailing_delta: str | None) -> None:
    """The STOP/TAKE_PROFIT family needs `stopPrice` and/or `trailingDelta` (S2 L3423)."""
    if stop_price is None and trailing_delta is None:
        raise ValueError(
            f"{prefix}_type={leg_type} requires a trigger: {prefix}_stop_price "
            f"(Binance: {_api_name(prefix, 'stop_price')}) and/or {prefix}_trailing_delta "
            f"(Binance: {_api_name(prefix, 'trailing_delta')}, in BIPS) — either one, or both."
        )


def _check_leg(
    prefix: str,
    leg_type: OrderListLegType,
    *,
    price: str | None,
    stop_price: str | None,
    trailing_delta: str | None,
    time_in_force: TimeInForce | None,
    iceberg_qty: str | None,
) -> None:
    """Enforce one OCO leg's mandatory parameter set (S2 L3571-L3579, the OTOCO table).

    The OCO section itself carries no such table; its per-field notes say the same thing
    ("Required if aboveType is STOP_LOSS_LIMIT or TAKE_PROFIT_LIMIT", "Either
    aboveStopPrice or aboveTrailingDelta or both"), so the OTOCO table is used for both.
    """
    if leg_type is OrderListLegType.LIMIT_MAKER:
        _require_leg_fields(prefix, leg_type.value, (("price", price),))
    elif leg_type in (OrderListLegType.STOP_LOSS, OrderListLegType.TAKE_PROFIT):
        _require_trigger(prefix, leg_type.value, stop_price, trailing_delta)
    else:  # STOP_LOSS_LIMIT / TAKE_PROFIT_LIMIT
        _require_leg_fields(prefix, leg_type.value, (("price", price), ("time_in_force", time_in_force)))
        _require_trigger(prefix, leg_type.value, stop_price, trailing_delta)
    _check_iceberg(prefix, leg_type.value, iceberg_qty=iceberg_qty, time_in_force=time_in_force)


def _check_iceberg(
    prefix: str,
    leg_type: str,
    *,
    iceberg_qty: str | None,
    time_in_force: TimeInForce | None,
) -> None:
    """`icebergQty` only applies to a GTC leg, or to a LIMIT_MAKER (S2 L3389, L3543)."""
    if iceberg_qty is None:
        return
    if time_in_force is not TimeInForce.GTC and leg_type != OrderListLegType.LIMIT_MAKER.value:
        raise ValueError(
            f"{prefix}_iceberg_qty (Binance: {_api_name(prefix, 'iceberg_qty')}) is only accepted when "
            f"{prefix}_time_in_force is GTC or {prefix}_type is LIMIT_MAKER; got {prefix}_type={leg_type} "
            f"and {prefix}_time_in_force={time_in_force.value if time_in_force else None}."
        )


def _leg_reference_price(
    leg_type: OrderListLegType, price: str | None, stop_price: str | None
) -> tuple[str | None, str]:
    """The price Binance compares against the last traded price, for this leg type.

    LIMIT_MAKER and TAKE_PROFIT_LIMIT are compared on their limit `price`; the rest of the
    family is compared on `stopPrice` (S2 L3234-L3239).
    """
    if leg_type in (OrderListLegType.LIMIT_MAKER, OrderListLegType.TAKE_PROFIT_LIMIT):
        return price, "price"
    return stop_price, "stop_price"


def _check_oco_pair(
    *,
    side: OrderSide,
    above_prefix: str,
    below_prefix: str,
    above_type: OrderListLegType,
    above_price: str | None,
    above_stop_price: str | None,
    below_type: OrderListLegType,
    below_price: str | None,
    below_stop_price: str | None,
) -> None:
    """Enforce the OCO pairing and price-ordering rules from S2 L3232-L3239.

    Two things are checked, both locally and both before anything is signed:

    1. Exactly one leg from the take-profit family and one from the stop family, oriented
       by side — on a SELL the take-profit is the ABOVE leg, on a BUY it is the BELOW one.
    2. The above leg's reference price is strictly greater than the below leg's.

    **The last traded price cannot be checked here.** Binance's real rule is
    `above > last traded price > below`; this server has no live price, so only the
    relationship *between the two given prices* is verified. A pair that sits entirely on
    one side of the market passes here and is rejected by Binance (-2010/-2021).
    """
    profit_above = above_type in _PROFIT_TYPES
    profit_below = below_type in _PROFIT_TYPES
    if profit_above == profit_below:
        raise ValueError(
            "An OCO pair needs exactly one take-profit leg (LIMIT_MAKER / TAKE_PROFIT / TAKE_PROFIT_LIMIT) "
            "and exactly one stop leg (STOP_LOSS / STOP_LOSS_LIMIT); got "
            f"{above_prefix}_type={above_type.value} and {below_prefix}_type={below_type.value}."
        )
    if side is OrderSide.SELL and not profit_above:
        raise ValueError(
            "On a SELL order list the take-profit leg sits ABOVE the last traded price and the stop leg "
            f"BELOW it: set {above_prefix}_type to LIMIT_MAKER / TAKE_PROFIT / TAKE_PROFIT_LIMIT and "
            f"{below_prefix}_type to STOP_LOSS / STOP_LOSS_LIMIT (got {above_type.value} above, "
            f"{below_type.value} below)."
        )
    if side is OrderSide.BUY and profit_above:
        raise ValueError(
            "On a BUY order list the stop leg sits ABOVE the last traded price and the limit leg BELOW it: "
            f"set {above_prefix}_type to STOP_LOSS / STOP_LOSS_LIMIT and {below_prefix}_type to "
            f"LIMIT_MAKER / TAKE_PROFIT / TAKE_PROFIT_LIMIT (got {above_type.value} above, "
            f"{below_type.value} below)."
        )
    above_ref, above_field = _leg_reference_price(above_type, above_price, above_stop_price)
    below_ref, below_field = _leg_reference_price(below_type, below_price, below_stop_price)
    if above_ref is None or below_ref is None:
        # A trailing-delta-only leg has no price to compare; Binance resolves it at trigger time.
        return
    if Decimal(above_ref) <= Decimal(below_ref):
        raise ValueError(
            f"{above_prefix}_{above_field} ({above_ref}) must be strictly greater than "
            f"{below_prefix}_{below_field} ({below_ref}): Binance requires "
            "above leg > last traded price > below leg. Only the relationship between the two prices you "
            "passed is checked here — this server does not know the last traded price, so a pair sitting "
            "entirely on one side of the market still gets rejected by Binance (-2010/-2021)."
        )


def _single_list_id(
    order_list_id: int | None, client_id: str | None, client_field: str, api_name: str
) -> dict[str, Any]:
    """Return the id param for an order-list lookup/cancel — exactly one id, checked here.

    Binance accepts both and resolves `orderListId` first, then checks the client id
    against that result; refusing the ambiguous call keeps a mutating path unambiguous.
    """
    if (order_list_id is None) == (client_id is None):
        raise RuntimeError(
            f"Pass exactly one of order_list_id (Binance: orderListId) or {client_field} "
            f"(Binance: {api_name}) — not both, not neither. Binance resolves orderListId first when both "
            "are sent and rejects the request if the two do not describe the same list."
        )
    if order_list_id is not None:
        return {"orderListId": order_list_id}
    return {api_name: client_id}


def _list_detail_lines(data: dict[str, Any]) -> list[str]:
    """Render only the order-list fields Binance actually returned."""
    lines: list[str] = []

    def add(label: str, key: str, fmt: Callable[[Any], str] = _as_str) -> None:
        value = data.get(key)
        if value is not None:
            lines.append(f"- **{label}**: {fmt(value)}")

    add("orderListId", "orderListId")
    add("contingencyType", "contingencyType")
    add("listStatusType", "listStatusType")
    add("listOrderStatus", "listOrderStatus")
    add("listClientOrderId", "listClientOrderId")
    add("symbol", "symbol")
    add("transactionTime", "transactionTime", epoch_to_human)
    return lines


def _legs_lines(data: dict[str, Any]) -> list[str]:
    """Render the legs: `orderReports[]` when present, otherwise the thin `orders[]`."""
    reports = data.get("orderReports")
    orders = data.get("orders")
    rows: list[dict[str, Any]]
    detailed: bool
    if isinstance(reports, list) and reports:
        rows, detailed = reports, True
    elif isinstance(orders, list) and orders:
        rows, detailed = orders, False
    else:
        return ["", "### Legs", "", "_Binance returned no legs in this response._"]
    lines = [
        "",
        "### Legs",
        "",
        "| orderId | clientOrderId | origClientOrderId | type | side | status | price | stopPrice | origQty |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows[:MAX_DISPLAY_ROWS]:
        lines.append(
            f"| {row.get('orderId', 'N/A')} | {row.get('clientOrderId', 'N/A')} | "
            f"{row.get('origClientOrderId', 'N/A')} | {row.get('type', 'N/A')} | {row.get('side', 'N/A')} | "
            f"{row.get('status', 'N/A')} | {fmt_num(row.get('price'))} | {fmt_num(row.get('stopPrice'))} | "
            f"{fmt_num(row.get('origQty'))} |"
        )
    if len(rows) > MAX_DISPLAY_ROWS:
        lines.append(f"_[{len(rows) - MAX_DISPLAY_ROWS} more leg(s) not shown]_")
    if not detailed:
        lines.append("")
        lines.append(
            "_Binance returned ids only (no `orderReports`): the per-leg type, side and status are NOT in "
            "this response. Read them with `binance_get_order` per leg._"
        )
    return lines


def _placement_lines(data: dict[str, Any], symbol: str, kind: str) -> list[str]:
    """Confirmation for a placement: never claims more than the response carries."""
    list_status = data.get("listOrderStatus")
    reports = data.get("orderReports")
    leg_statuses = [row.get("status") for row in reports] if isinstance(reports, list) and reports else []
    legs_dead = bool(leg_statuses) and all(status in _NOT_LIVE_ORDER_STATUSES for status in leg_statuses)
    not_live = list_status in _NOT_LIVE_LIST_STATUSES or legs_dead
    if not_live:
        label = (
            list_status if list_status in _NOT_LIVE_LIST_STATUSES else "/".join(sorted({str(s) for s in leg_statuses}))
        )
        lines = [f"# {kind} order list NOT live ({label}) on {symbol}", ""]
    else:
        lines = [f"# {kind} order list placed on {symbol}", ""]
    lines.extend(_list_detail_lines(data))
    lines.extend(_legs_lines(data))
    if not_live:
        lines.append("")
        lines.append(
            f"_The list is **not working**: Binance reports listOrderStatus={list_status!r} and the legs "
            "above. Nothing is resting on the book — re-place deliberately._"
        )
    elif list_status is not None:
        lines.append("")
        lines.append(
            f"_listOrderStatus **{list_status}**. Track the list with `binance_get_order_list` "
            "(or `binance_get_open_order_lists`) and cancel it with `binance_cancel_order_list`._"
        )
    else:
        lines.append("")
        lines.append(
            "_Binance answered without a listOrderStatus. The list was accepted; its state is unknown from "
            "this response — read it with `binance_get_order_list`._"
        )
    return lines


def _order_list_table(rows: list[dict[str, Any]], title: str) -> list[str]:
    """Compact table for the listing tools (all order lists / open order lists)."""
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_No order lists._")
        return lines
    lines.append(f"Showing **{min(len(rows), MAX_DISPLAY_ROWS):,}** of **{len(rows):,}** order list(s).")
    lines.append("")
    lines.append(
        "| time | symbol | orderListId | contingencyType | listStatusType | listOrderStatus | listClientOrderId | legs |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in rows[:MAX_DISPLAY_ROWS]:
        orders = row.get("orders")
        legs = len(orders) if isinstance(orders, list) else "N/A"
        lines.append(
            f"| {epoch_to_human(row.get('transactionTime'))} | {row.get('symbol', 'N/A')} | "
            f"{row.get('orderListId', 'N/A')} | {row.get('contingencyType', 'N/A')} | "
            f"{row.get('listStatusType', 'N/A')} | {row.get('listOrderStatus', 'N/A')} | "
            f"{row.get('listClientOrderId', 'N/A')} | {legs} |"
        )
    if len(rows) > MAX_DISPLAY_ROWS:
        lines.append("")
        lines.append(
            f"_[{len(rows) - MAX_DISPLAY_ROWS} more order list(s) not shown — narrow the window or use "
            'response_format="json"]_'
        )
    return lines


# -- input models --------------------------------------------------------------------


class _BaseInput(BaseModel):
    """Shared pydantic config for every input model in this module."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class _SymbolInput(_BaseInput):
    """Anything that requires a trading pair."""

    symbol: str = Field(description="Trading pair, e.g. `BTCUSDT` (uppercased automatically).")

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)


class _ListPlacementInput(_SymbolInput):
    """The parameters every order-list placement shares."""

    list_client_order_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=36,
        description=(
            "Your own id for the LIST (Binance: listClientOrderId); auto-generated if omitted. Distinct "
            "from the per-leg client ids. Reusable only once the previous list with that id has filled or "
            "fully expired — set it to make a retry traceable and to cancel by a name you chose."
        ),
    )
    new_order_resp_type: NewOrderRespType | None = Field(
        default=None,
        description=(
            "How much detail each leg's report carries: ACK (ids only), RESULT (adds status/quantities) or "
            "FULL (adds the fills). Binance: newOrderRespType."
        ),
    )
    self_trade_prevention_mode: SelfTradePreventionMode | None = Field(
        default=None,
        description=(
            "What Binance does if a leg would match your own resting order: NONE, EXPIRE_TAKER, "
            "EXPIRE_MAKER, EXPIRE_BOTH or DECREMENT. Binance: selfTradePreventionMode."
        ),
    )

    def _shared_params(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "listClientOrderId": self.list_client_order_id,
            "newOrderRespType": _enum_value(self.new_order_resp_type),
            "selfTradePreventionMode": _enum_value(self.self_trade_prevention_mode),
        }


class PlaceOcoOrderInput(_ListPlacementInput):
    """Params for `POST /api/v3/orderList/oco` — two legs, one quantity, real money."""

    side: OrderSide = Field(description="BUY or SELL — both legs share it.")
    quantity: str = Field(
        description="Base-asset amount for BOTH legs, as a decimal STRING (e.g. '0.001'). Sent verbatim."
    )
    above_type: OrderListLegType = Field(
        description=(
            "Type of the leg ABOVE the last traded price: STOP_LOSS, STOP_LOSS_LIMIT, LIMIT_MAKER, "
            "TAKE_PROFIT or TAKE_PROFIT_LIMIT. On a SELL list this is the take-profit leg "
            "(LIMIT_MAKER / TAKE_PROFIT / TAKE_PROFIT_LIMIT); on a BUY list it is the stop leg "
            "(STOP_LOSS / STOP_LOSS_LIMIT). Binance: aboveType."
        )
    )
    above_client_order_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=36,
        description="Your own id for the above leg (Binance: aboveClientOrderId).",
    )
    above_price: str | None = Field(
        default=None,
        description=(
            "Limit price of the above leg as a decimal STRING. Required for LIMIT_MAKER, STOP_LOSS_LIMIT "
            "and TAKE_PROFIT_LIMIT. Binance: abovePrice."
        ),
    )
    above_stop_price: str | None = Field(
        default=None,
        description=(
            "Trigger price of the above leg as a decimal STRING. For STOP_LOSS, STOP_LOSS_LIMIT, "
            "TAKE_PROFIT and TAKE_PROFIT_LIMIT: this and/or above_trailing_delta is required. "
            "Binance: aboveStopPrice."
        ),
    )
    above_trailing_delta: str | None = Field(
        default=None,
        description=(
            "Trailing-stop distance of the above leg in BIPS as a STRING ('100' = 1 percent). "
            "Binance: aboveTrailingDelta."
        ),
    )
    above_time_in_force: TimeInForce | None = Field(
        default=None,
        description="GTC/IOC/FOK for the above leg. Required when above_type is STOP_LOSS_LIMIT or TAKE_PROFIT_LIMIT.",
    )
    above_iceberg_qty: str | None = Field(
        default=None,
        description=(
            "Visible slice of the above leg as a decimal STRING (Binance: aboveIcebergQty). Only with "
            "above_time_in_force=GTC, or a LIMIT_MAKER leg."
        ),
    )
    above_strategy_id: int | None = Field(
        default=None,
        description="Arbitrary numeric id for the above leg within your strategy (Binance: aboveStrategyId).",
    )
    above_strategy_type: int | None = Field(
        default=None,
        ge=1000000,
        description="Arbitrary strategy id for the above leg; values below 1000000 are reserved (Binance: aboveStrategyType).",
    )
    below_type: OrderListLegType = Field(
        description=(
            "Type of the leg BELOW the last traded price. On a SELL list this is the stop leg "
            "(STOP_LOSS / STOP_LOSS_LIMIT); on a BUY list it is the limit leg (LIMIT_MAKER / TAKE_PROFIT / "
            "TAKE_PROFIT_LIMIT). Binance: belowType."
        )
    )
    below_client_order_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=36,
        description="Your own id for the below leg (Binance: belowClientOrderId).",
    )
    below_price: str | None = Field(
        default=None,
        description=(
            "Limit price of the below leg as a decimal STRING. Required for LIMIT_MAKER, STOP_LOSS_LIMIT "
            "and TAKE_PROFIT_LIMIT. Binance: belowPrice."
        ),
    )
    below_stop_price: str | None = Field(
        default=None,
        description=(
            "Trigger price of the below leg as a decimal STRING; this and/or below_trailing_delta is "
            "required for the STOP_LOSS / TAKE_PROFIT family. Binance: belowStopPrice."
        ),
    )
    below_trailing_delta: str | None = Field(
        default=None,
        description="Trailing-stop distance of the below leg in BIPS as a STRING. Binance: belowTrailingDelta.",
    )
    below_time_in_force: TimeInForce | None = Field(
        default=None,
        description="GTC/IOC/FOK for the below leg. Required when below_type is STOP_LOSS_LIMIT or TAKE_PROFIT_LIMIT.",
    )
    below_iceberg_qty: str | None = Field(
        default=None,
        description=(
            "Visible slice of the below leg as a decimal STRING (Binance: belowIcebergQty). Only with "
            "below_time_in_force=GTC, or a LIMIT_MAKER leg."
        ),
    )
    below_strategy_id: int | None = Field(
        default=None,
        description="Arbitrary numeric id for the below leg within your strategy (Binance: belowStrategyId).",
    )
    below_strategy_type: int | None = Field(
        default=None,
        ge=1000000,
        description="Arbitrary strategy id for the below leg; values below 1000000 are reserved (Binance: belowStrategyType).",
    )

    @field_validator(
        "quantity",
        "above_price",
        "above_stop_price",
        "above_trailing_delta",
        "above_iceberg_qty",
        "below_price",
        "below_stop_price",
        "below_trailing_delta",
        "below_iceberg_qty",
    )
    @classmethod
    def _check_positive_decimal(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _positive_decimal(value, info)

    @model_validator(mode="after")
    def _check_legs(self) -> Self:
        _check_leg(
            "above",
            self.above_type,
            price=self.above_price,
            stop_price=self.above_stop_price,
            trailing_delta=self.above_trailing_delta,
            time_in_force=self.above_time_in_force,
            iceberg_qty=self.above_iceberg_qty,
        )
        _check_leg(
            "below",
            self.below_type,
            price=self.below_price,
            stop_price=self.below_stop_price,
            trailing_delta=self.below_trailing_delta,
            time_in_force=self.below_time_in_force,
            iceberg_qty=self.below_iceberg_qty,
        )
        _check_oco_pair(
            side=self.side,
            above_prefix="above",
            below_prefix="below",
            above_type=self.above_type,
            above_price=self.above_price,
            above_stop_price=self.above_stop_price,
            below_type=self.below_type,
            below_price=self.below_price,
            below_stop_price=self.below_stop_price,
        )
        return self

    def to_params(self) -> dict[str, Any]:
        """Map to Binance's camelCase query parameters, dropping everything unset."""
        raw = self._shared_params()
        raw.update(
            {
                "side": self.side.value,
                "quantity": self.quantity,
                "aboveType": self.above_type.value,
                "aboveClientOrderId": self.above_client_order_id,
                "abovePrice": self.above_price,
                "aboveStopPrice": self.above_stop_price,
                "aboveTrailingDelta": self.above_trailing_delta,
                "aboveTimeInForce": _enum_value(self.above_time_in_force),
                "aboveIcebergQty": self.above_iceberg_qty,
                "aboveStrategyId": self.above_strategy_id,
                "aboveStrategyType": self.above_strategy_type,
                "belowType": self.below_type.value,
                "belowClientOrderId": self.below_client_order_id,
                "belowPrice": self.below_price,
                "belowStopPrice": self.below_stop_price,
                "belowTrailingDelta": self.below_trailing_delta,
                "belowTimeInForce": _enum_value(self.below_time_in_force),
                "belowIcebergQty": self.below_iceberg_qty,
                "belowStrategyId": self.below_strategy_id,
                "belowStrategyType": self.below_strategy_type,
            }
        )
        return {key: value for key, value in raw.items() if value is not None}


class _WorkingLegInput(_ListPlacementInput):
    """The working leg shared by OTO and OTOCO — it rests on the book first."""

    working_type: WorkingOrderType = Field(
        description="LIMIT or LIMIT_MAKER — the working leg must be able to rest on the book. Binance: workingType."
    )
    working_side: OrderSide = Field(description="BUY or SELL for the working leg. Binance: workingSide.")
    working_price: str = Field(description="Limit price of the working leg as a decimal STRING. Binance: workingPrice.")
    working_quantity: str = Field(
        description="Base-asset amount of the working leg as a decimal STRING. Binance: workingQuantity."
    )
    working_time_in_force: TimeInForce | None = Field(
        default=None,
        description=(
            "GTC/IOC/FOK for the working leg. **Required when working_type is LIMIT**; a LIMIT_MAKER leg "
            "does not take one. Binance: workingTimeInForce."
        ),
    )
    working_client_order_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=36,
        description="Your own id for the working leg (Binance: workingClientOrderId).",
    )
    working_iceberg_qty: str | None = Field(
        default=None,
        description=(
            "Visible slice of the working leg as a decimal STRING (Binance: workingIcebergQty). Only with "
            "working_time_in_force=GTC, or a LIMIT_MAKER working leg."
        ),
    )
    working_strategy_id: int | None = Field(
        default=None,
        description="Arbitrary numeric id for the working leg within your strategy (Binance: workingStrategyId).",
    )
    working_strategy_type: int | None = Field(
        default=None,
        ge=1000000,
        description="Arbitrary strategy id for the working leg; values below 1000000 are reserved (Binance: workingStrategyType).",
    )

    def _check_working_leg(self) -> None:
        if self.working_type is WorkingOrderType.LIMIT and self.working_time_in_force is None:
            raise ValueError(
                "working_type=LIMIT requires working_time_in_force (Binance: workingTimeInForce); "
                "use LIMIT_MAKER if you want a post-only working leg with no time-in-force."
            )
        _check_iceberg(
            "working",
            self.working_type.value,
            iceberg_qty=self.working_iceberg_qty,
            time_in_force=self.working_time_in_force,
        )

    def _working_params(self) -> dict[str, Any]:
        raw = self._shared_params()
        raw.update(
            {
                "workingType": self.working_type.value,
                "workingSide": self.working_side.value,
                "workingPrice": self.working_price,
                "workingQuantity": self.working_quantity,
                "workingTimeInForce": _enum_value(self.working_time_in_force),
                "workingClientOrderId": self.working_client_order_id,
                "workingIcebergQty": self.working_iceberg_qty,
                "workingStrategyId": self.working_strategy_id,
                "workingStrategyType": self.working_strategy_type,
            }
        )
        return raw


class PlaceOtoOrderInput(_WorkingLegInput):
    """Params for `POST /api/v3/orderList/oto` — working leg, then pending leg."""

    pending_type: PendingOrderType = Field(
        description=(
            "Type of the pending leg — any order type except a MARKET order priced with quoteOrderQty "
            "(this server never sends quoteOrderQty on a list). Mandatory extras: LIMIT → pending_price + "
            "pending_time_in_force; STOP_LOSS / TAKE_PROFIT → pending_stop_price and/or "
            "pending_trailing_delta; STOP_LOSS_LIMIT / TAKE_PROFIT_LIMIT → pending_price + "
            "pending_time_in_force + (pending_stop_price and/or pending_trailing_delta); LIMIT_MAKER → "
            "pending_price. Binance: pendingType."
        )
    )
    pending_side: OrderSide = Field(description="BUY or SELL for the pending leg. Binance: pendingSide.")
    pending_quantity: str = Field(
        description="Base-asset amount of the pending leg as a decimal STRING. Binance: pendingQuantity."
    )
    pending_price: str | None = Field(
        default=None, description="Limit price of the pending leg as a decimal STRING. Binance: pendingPrice."
    )
    pending_stop_price: str | None = Field(
        default=None, description="Trigger price of the pending leg as a decimal STRING. Binance: pendingStopPrice."
    )
    pending_trailing_delta: str | None = Field(
        default=None,
        description="Trailing-stop distance of the pending leg in BIPS as a STRING. Binance: pendingTrailingDelta.",
    )
    pending_time_in_force: TimeInForce | None = Field(
        default=None, description="GTC/IOC/FOK for the pending leg. Binance: pendingTimeInForce."
    )
    pending_client_order_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=36,
        description="Your own id for the pending leg (Binance: pendingClientOrderId).",
    )
    pending_iceberg_qty: str | None = Field(
        default=None,
        description=(
            "Visible slice of the pending leg as a decimal STRING (Binance: pendingIcebergQty). Only with "
            "pending_time_in_force=GTC, or a LIMIT_MAKER pending leg."
        ),
    )
    pending_strategy_id: int | None = Field(
        default=None,
        description="Arbitrary numeric id for the pending leg within your strategy (Binance: pendingStrategyId).",
    )
    pending_strategy_type: int | None = Field(
        default=None,
        ge=1000000,
        description="Arbitrary strategy id for the pending leg; values below 1000000 are reserved (Binance: pendingStrategyType).",
    )

    @field_validator(
        "working_price",
        "working_quantity",
        "working_iceberg_qty",
        "pending_quantity",
        "pending_price",
        "pending_stop_price",
        "pending_trailing_delta",
        "pending_iceberg_qty",
    )
    @classmethod
    def _check_positive_decimal(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _positive_decimal(value, info)

    @model_validator(mode="after")
    def _check_legs(self) -> Self:
        self._check_working_leg()
        pending = self.pending_type
        if pending is PendingOrderType.LIMIT:
            _require_leg_fields(
                "pending", pending.value, (("price", self.pending_price), ("time_in_force", self.pending_time_in_force))
            )
        elif pending in (PendingOrderType.STOP_LOSS, PendingOrderType.TAKE_PROFIT):
            _require_trigger("pending", pending.value, self.pending_stop_price, self.pending_trailing_delta)
        elif pending in (PendingOrderType.STOP_LOSS_LIMIT, PendingOrderType.TAKE_PROFIT_LIMIT):
            _require_leg_fields(
                "pending", pending.value, (("price", self.pending_price), ("time_in_force", self.pending_time_in_force))
            )
            _require_trigger("pending", pending.value, self.pending_stop_price, self.pending_trailing_delta)
        elif pending is PendingOrderType.LIMIT_MAKER:
            # Not in S2's OTO table, which omits LIMIT_MAKER entirely; taken from the OTOCO
            # table (L3574), where a LIMIT_MAKER pending leg does require its price.
            _require_leg_fields("pending", pending.value, (("price", self.pending_price),))
        _check_iceberg(
            "pending",
            pending.value,
            iceberg_qty=self.pending_iceberg_qty,
            time_in_force=self.pending_time_in_force,
        )
        return self

    def to_params(self) -> dict[str, Any]:
        """Map to Binance's camelCase query parameters, dropping everything unset."""
        raw = self._working_params()
        raw.update(
            {
                "pendingType": self.pending_type.value,
                "pendingSide": self.pending_side.value,
                "pendingQuantity": self.pending_quantity,
                "pendingPrice": self.pending_price,
                "pendingStopPrice": self.pending_stop_price,
                "pendingTrailingDelta": self.pending_trailing_delta,
                "pendingTimeInForce": _enum_value(self.pending_time_in_force),
                "pendingClientOrderId": self.pending_client_order_id,
                "pendingIcebergQty": self.pending_iceberg_qty,
                "pendingStrategyId": self.pending_strategy_id,
                "pendingStrategyType": self.pending_strategy_type,
            }
        )
        return {key: value for key, value in raw.items() if value is not None}


class PlaceOtocoOrderInput(_WorkingLegInput):
    """Params for `POST /api/v3/orderList/otoco` — working leg, then an OCO pair."""

    pending_side: OrderSide = Field(description="BUY or SELL for both pending legs. Binance: pendingSide.")
    pending_quantity: str = Field(
        description="Base-asset amount for both pending legs as a decimal STRING. Binance: pendingQuantity."
    )
    pending_above_type: OrderListLegType = Field(
        description=(
            "Type of the pending leg ABOVE the last traded price: STOP_LOSS, STOP_LOSS_LIMIT, LIMIT_MAKER, "
            "TAKE_PROFIT or TAKE_PROFIT_LIMIT. Binance: pendingAboveType."
        )
    )
    pending_above_client_order_id: str | None = Field(
        default=None, min_length=1, max_length=36, description="Your own id for the pending above leg."
    )
    pending_above_price: str | None = Field(
        default=None,
        description="Limit price of the pending above leg as a decimal STRING. Binance: pendingAbovePrice.",
    )
    pending_above_stop_price: str | None = Field(
        default=None,
        description="Trigger price of the pending above leg as a decimal STRING. Binance: pendingAboveStopPrice.",
    )
    pending_above_trailing_delta: str | None = Field(
        default=None,
        description="Trailing-stop distance of the pending above leg in BIPS. Binance: pendingAboveTrailingDelta.",
    )
    pending_above_time_in_force: TimeInForce | None = Field(
        default=None,
        description=(
            "GTC/IOC/FOK for the pending above leg; required when its type is STOP_LOSS_LIMIT or "
            "TAKE_PROFIT_LIMIT. Binance: pendingAboveTimeInForce."
        ),
    )
    pending_above_iceberg_qty: str | None = Field(
        default=None, description="Visible slice of the pending above leg. Binance: pendingAboveIcebergQty."
    )
    pending_above_strategy_id: int | None = Field(
        default=None, description="Arbitrary numeric id for the pending above leg (Binance: pendingAboveStrategyId)."
    )
    pending_above_strategy_type: int | None = Field(
        default=None,
        ge=1000000,
        description="Strategy id for the pending above leg; below 1000000 is reserved (Binance: pendingAboveStrategyType).",
    )
    pending_below_type: OrderListLegType | None = Field(
        default=None,
        description=(
            "Type of the pending leg BELOW the last traded price. Binance marks it optional, but a list "
            "without it is an OTO, not an OTOCO — pass it together with the above leg so the two form the "
            "OCO pair (exactly one take-profit leg and one stop leg). Binance: pendingBelowType."
        ),
    )
    pending_below_client_order_id: str | None = Field(
        default=None, min_length=1, max_length=36, description="Your own id for the pending below leg."
    )
    pending_below_price: str | None = Field(
        default=None,
        description="Limit price of the pending below leg as a decimal STRING. Binance: pendingBelowPrice.",
    )
    pending_below_stop_price: str | None = Field(
        default=None,
        description="Trigger price of the pending below leg as a decimal STRING. Binance: pendingBelowStopPrice.",
    )
    pending_below_trailing_delta: str | None = Field(
        default=None,
        description="Trailing-stop distance of the pending below leg in BIPS. Binance: pendingBelowTrailingDelta.",
    )
    pending_below_time_in_force: TimeInForce | None = Field(
        default=None,
        description=(
            "GTC/IOC/FOK for the pending below leg; required when its type is STOP_LOSS_LIMIT or "
            "TAKE_PROFIT_LIMIT. Binance: pendingBelowTimeInForce."
        ),
    )
    pending_below_iceberg_qty: str | None = Field(
        default=None, description="Visible slice of the pending below leg. Binance: pendingBelowIcebergQty."
    )
    pending_below_strategy_id: int | None = Field(
        default=None, description="Arbitrary numeric id for the pending below leg (Binance: pendingBelowStrategyId)."
    )
    pending_below_strategy_type: int | None = Field(
        default=None,
        ge=1000000,
        description="Strategy id for the pending below leg; below 1000000 is reserved (Binance: pendingBelowStrategyType).",
    )

    @field_validator(
        "working_price",
        "working_quantity",
        "working_iceberg_qty",
        "pending_quantity",
        "pending_above_price",
        "pending_above_stop_price",
        "pending_above_trailing_delta",
        "pending_above_iceberg_qty",
        "pending_below_price",
        "pending_below_stop_price",
        "pending_below_trailing_delta",
        "pending_below_iceberg_qty",
    )
    @classmethod
    def _check_positive_decimal(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _positive_decimal(value, info)

    @model_validator(mode="after")
    def _check_legs(self) -> Self:
        self._check_working_leg()
        _check_leg(
            "pending_above",
            self.pending_above_type,
            price=self.pending_above_price,
            stop_price=self.pending_above_stop_price,
            trailing_delta=self.pending_above_trailing_delta,
            time_in_force=self.pending_above_time_in_force,
            iceberg_qty=self.pending_above_iceberg_qty,
        )
        if self.pending_below_type is None:
            return self
        _check_leg(
            "pending_below",
            self.pending_below_type,
            price=self.pending_below_price,
            stop_price=self.pending_below_stop_price,
            trailing_delta=self.pending_below_trailing_delta,
            time_in_force=self.pending_below_time_in_force,
            iceberg_qty=self.pending_below_iceberg_qty,
        )
        _check_oco_pair(
            side=self.pending_side,
            above_prefix="pending_above",
            below_prefix="pending_below",
            above_type=self.pending_above_type,
            above_price=self.pending_above_price,
            above_stop_price=self.pending_above_stop_price,
            below_type=self.pending_below_type,
            below_price=self.pending_below_price,
            below_stop_price=self.pending_below_stop_price,
        )
        return self

    def to_params(self) -> dict[str, Any]:
        """Map to Binance's camelCase query parameters, dropping everything unset."""
        raw = self._working_params()
        raw.update(
            {
                "pendingSide": self.pending_side.value,
                "pendingQuantity": self.pending_quantity,
                "pendingAboveType": self.pending_above_type.value,
                "pendingAboveClientOrderId": self.pending_above_client_order_id,
                "pendingAbovePrice": self.pending_above_price,
                "pendingAboveStopPrice": self.pending_above_stop_price,
                "pendingAboveTrailingDelta": self.pending_above_trailing_delta,
                "pendingAboveTimeInForce": _enum_value(self.pending_above_time_in_force),
                "pendingAboveIcebergQty": self.pending_above_iceberg_qty,
                "pendingAboveStrategyId": self.pending_above_strategy_id,
                "pendingAboveStrategyType": self.pending_above_strategy_type,
                "pendingBelowType": _enum_value(self.pending_below_type),
                "pendingBelowClientOrderId": self.pending_below_client_order_id,
                "pendingBelowPrice": self.pending_below_price,
                "pendingBelowStopPrice": self.pending_below_stop_price,
                "pendingBelowTrailingDelta": self.pending_below_trailing_delta,
                "pendingBelowTimeInForce": _enum_value(self.pending_below_time_in_force),
                "pendingBelowIcebergQty": self.pending_below_iceberg_qty,
                "pendingBelowStrategyId": self.pending_below_strategy_id,
                "pendingBelowStrategyType": self.pending_below_strategy_type,
            }
        )
        return {key: value for key, value in raw.items() if value is not None}


class GetOrderListInput(_BaseInput):
    """Params for `GET /api/v3/orderList` — no symbol needed."""

    order_list_id: int | None = Field(
        default=None, description="Binance orderListId. Pass this OR orig_client_order_id."
    )
    orig_client_order_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The listClientOrderId you sent (or that Binance generated) when placing the list. Pass this "
            "OR order_list_id. Binance spells the query parameter origClientOrderId here."
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: `markdown` (human-readable) or `json` (raw Binance payload).",
    )


class CancelOrderListInput(_SymbolInput):
    """Params for `DELETE /api/v3/orderList` — cancels every leg of the list."""

    order_list_id: int | None = Field(
        default=None, description="Binance orderListId. Pass this OR list_client_order_id."
    )
    list_client_order_id: str | None = Field(
        default=None,
        min_length=1,
        description="The listClientOrderId of the list to cancel. Pass this OR order_list_id.",
    )
    new_client_order_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=36,
        description="Your own id for the cancel itself (Binance: newClientOrderId); auto-generated if omitted.",
    )


class GetAllOrderListsInput(_BaseInput):
    """Params for `GET /api/v3/allOrderList` — cross-symbol history."""

    from_id: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Cursor: return lists with orderListId >= this. Binance forbids combining it with "
            "start_time/end_time. Binance: fromId."
        ),
    )
    start_time: int | str | None = Field(
        default=None,
        description="Window start: epoch ms or ISO-8601. With end_time, the span must be 24 h or less.",
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end: epoch ms or ISO-8601. With start_time, the span must be 24 h or less.",
    )
    limit: int = Field(default=500, ge=1, le=1000, description="Order lists to fetch, 1-1000 (Binance default 500).")
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: `markdown` (human-readable) or `json` (raw Binance payload).",
    )


class GetOpenOrderListsInput(_BaseInput):
    """Params for `GET /api/v3/openOrderList` — it takes no filters at all."""

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: `markdown` (human-readable) or `json` (raw Binance payload).",
    )


# -- tools ---------------------------------------------------------------------------


@mcp.tool(
    name="binance_place_oco_order",
    annotations=ToolAnnotations(
        title="Binance Place OCO Order List",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def binance_place_oco_order(params: PlaceOcoOrderInput) -> str:
    """Place a REAL one-cancels-the-other pair (take-profit + stop). This spends real money.

    Calls `POST /api/v3/orderList/oco` (SIGNED, IP weight 1, unfilled-order count 2). Both
    legs carry the same `quantity` and the same `side`; when one triggers, Binance cancels
    the other. This is the bracket around a position you already hold (SELL) or the
    breakout/dip pair for one you want (BUY).

    **Kill-switch.** This call is refused with `Error: … trading is disabled …` unless the
    server runs with `BINANCE_ALLOW_TRADING=1`. The gate lives in the HTTP client, so no
    tool can bypass it. If you see that error, the operator has deliberately put the server
    in read-only mode — report it, do not try to work around it.

    **There is no dry-run for a list.** `binance_test_order` validates ONE order, not a
    list; run it per leg if you want Binance's filter check before committing.

    Leg rules, enforced locally before anything is signed:
    - exactly one take-profit leg (LIMIT_MAKER / TAKE_PROFIT / TAKE_PROFIT_LIMIT) and one
      stop leg (STOP_LOSS / STOP_LOSS_LIMIT);
    - on a **SELL** the take-profit leg is the `above` one, on a **BUY** it is the `below`
      one;
    - the above leg's price must be strictly greater than the below leg's. Binance's full
      rule is `above > last traded price > below`, and **this server does not know the last
      traded price** — only the relationship between the two prices you pass is checked
      here. Read the market with `binance_get_ticker_price` first.

    When to Use:
    - Bracketing an open position with a target and a stop in one atomic request.
    - Any time two orders must be mutually exclusive — placing them separately risks both
      filling.

    When NOT to Use:
    - For a single order — `binance_place_order` (spot_orders.py).
    - When the bracket should only arm after an entry fills — that is
      `binance_place_otoco_order`.
    - To change an existing list: cancel it with `binance_cancel_order_list` and place a
      new one; there is no amend for lists.

    Returns:
    A confirmation echoing exactly what Binance returned: orderListId, contingencyType,
    listStatusType, listOrderStatus, listClientOrderId, and a `### Legs` table built from
    `orderReports` when the response carries one (ids only otherwise, and it says so).
    Nothing is inferred.

    Examples:
        params = {"symbol": "BTCUSDT", "side": "SELL", "quantity": "0.001",
                  "above_type": "LIMIT_MAKER", "above_price": "72000.00",
                  "below_type": "STOP_LOSS_LIMIT", "below_price": "58000.00",
                  "below_stop_price": "58500.00", "below_time_in_force": "GTC",
                  "list_client_order_id": "btc-bracket-001"}
        params = {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.001",
                  "above_type": "STOP_LOSS_LIMIT", "above_price": "71000.00",
                  "above_stop_price": "70500.00", "above_time_in_force": "GTC",
                  "below_type": "LIMIT_MAKER", "below_price": "60000.00"}

    Error Handling:
    - `Error: … trading is disabled …` → the kill-switch is off; nothing was sent.
    - **-2010** → insufficient balance, a symbol filter (LOT_SIZE / PRICE_FILTER /
      NOTIONAL), or the pair sits on the wrong side of the last traded price.
    - -2021 means a LIMIT_MAKER leg would have taken liquidity immediately.
    - -1013 / -1111 are precision / filter errors — read
      `binance_get_exchange_info`.
    - **A 5xx or a timeout means the execution status is UNKNOWN** — the list may well be
      live. Query it with `binance_get_order_list` (by `list_client_order_id` if you set
      one) before doing anything else. NEVER resend blindly.
    """
    try:
        client = get_client()
        resp = await client.request("POST", "/api/v3/orderList/oco", auth="signed", params=params.to_params())
        return clip_response("\n".join(_placement_lines(resp.json(), params.symbol, "OCO")))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_place_oto_order",
    annotations=ToolAnnotations(
        title="Binance Place OTO Order List",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def binance_place_oto_order(params: PlaceOtoOrderInput) -> str:
    """Place a REAL one-triggers-the-other pair (entry, then follow-up). Real money.

    Calls `POST /api/v3/orderList/oto` (SIGNED, IP weight 1, unfilled-order count 2). The
    **working** leg (LIMIT or LIMIT_MAKER) goes on the book immediately. The **pending**
    leg is only placed once the working leg is **fully filled** — until then it sits in
    `PENDING_NEW` and does nothing. Cancelling either leg kills the whole list.

    **Kill-switch.** Refused with `Error: … trading is disabled …` unless the server runs
    with `BINANCE_ALLOW_TRADING=1`.

    **There is no dry-run for a list**; `binance_test_order` validates one order at a time.

    Mandatory extras, enforced locally (S2 L3419):
    - `working_type=LIMIT` → `working_time_in_force`;
    - `pending_type=LIMIT` → `pending_price`, `pending_time_in_force`;
    - `pending_type=STOP_LOSS|TAKE_PROFIT` → `pending_stop_price` **and/or**
      `pending_trailing_delta`;
    - `pending_type=STOP_LOSS_LIMIT|TAKE_PROFIT_LIMIT` → `pending_price`,
      `pending_time_in_force`, and `pending_stop_price` and/or `pending_trailing_delta`;
    - `pending_type=LIMIT_MAKER` → `pending_price`.
    A MARKET pending leg is allowed, but only by `pending_quantity` — Binance does not
    support `quoteOrderQty` inside a list, and this server never sends it.

    When to Use:
    - Entry plus a single exit: buy at a limit, and the moment it fills, arm one stop.
    - Chaining two orders where the second must not exist until the first is done.

    When NOT to Use:
    - When the follow-up should be a target AND a stop — use
      `binance_place_otoco_order`.
    - When both orders should be live at once — that is `binance_place_oco_order`.

    Returns:
    A confirmation echoing orderListId, contingencyType, listStatusType, listOrderStatus,
    listClientOrderId and a `### Legs` table. A pending leg reported as `PENDING_NEW` is
    NOT on the book yet; the table shows exactly what Binance said and nothing more.

    Examples:
        params = {"symbol": "BTCUSDT", "working_type": "LIMIT", "working_side": "BUY",
                  "working_price": "60000.00", "working_quantity": "0.001",
                  "working_time_in_force": "GTC", "pending_type": "LIMIT",
                  "pending_side": "SELL", "pending_quantity": "0.001",
                  "pending_price": "66000.00", "pending_time_in_force": "GTC",
                  "list_client_order_id": "entry-then-target-001"}
        params = {"symbol": "BTCUSDT", "working_type": "LIMIT_MAKER", "working_side": "BUY",
                  "working_price": "60000.00", "working_quantity": "0.001",
                  "pending_type": "STOP_LOSS", "pending_side": "SELL",
                  "pending_quantity": "0.001", "pending_stop_price": "57000.00"}

    Error Handling:
    - `Error: … trading is disabled …` → the kill-switch is off; nothing was sent.
    - -2010 / -1013 / -1111 are balance, filter and precision failures on either leg.
    - -2021 means the working LIMIT_MAKER would have taken liquidity immediately.
    - **A 5xx or a timeout means the execution status is UNKNOWN** — query with
      `binance_get_order_list` before retrying; a duplicate entry is real money.
    """
    try:
        client = get_client()
        resp = await client.request("POST", "/api/v3/orderList/oto", auth="signed", params=params.to_params())
        return clip_response("\n".join(_placement_lines(resp.json(), params.symbol, "OTO")))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_place_otoco_order",
    annotations=ToolAnnotations(
        title="Binance Place OTOCO Order List",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def binance_place_otoco_order(params: PlaceOtocoOrderInput) -> str:
    """Place a REAL entry that arms a take-profit/stop pair when it fills. Real money.

    Calls `POST /api/v3/orderList/otoco` (SIGNED, IP weight 1, unfilled-order count 3).
    The **working** leg (LIMIT or LIMIT_MAKER) rests on the book; once it is **fully
    filled**, the two **pending** legs go on as an OCO pair, so the first of them to
    trigger cancels the other. Cancelling any leg kills the whole list.

    **Kill-switch.** Refused with `Error: … trading is disabled …` unless the server runs
    with `BINANCE_ALLOW_TRADING=1`.

    **There is no dry-run for a list**; `binance_test_order` validates one order at a time.

    Enforced locally before anything is signed (S2 L3571-L3579):
    - `working_type=LIMIT` → `working_time_in_force`;
    - per pending leg — LIMIT_MAKER → price; STOP_LOSS / TAKE_PROFIT → stop price and/or
      trailing delta; STOP_LOSS_LIMIT / TAKE_PROFIT_LIMIT → price, time-in-force, and stop
      price and/or trailing delta;
    - when both pending legs are given, the OCO pairing and the price ordering
      (`pending_above` price > `pending_below` price). **The last traded price is unknown
      to this server**, so only the relationship between the prices you pass is checked;
      Binance applies the full `above > last traded price > below` rule at trigger time.

    Binance marks `pending_below_type` optional; a list without it is really an OTO, so
    pass both pending legs unless you mean to place an OTO.

    When to Use:
    - The full bracket in one request: entry, target and stop, with nothing armed until
      the entry fills.

    When NOT to Use:
    - When you already hold the position — the bracket alone is
      `binance_place_oco_order`.
    - For entry plus a single follow-up — `binance_place_oto_order`.

    Returns:
    A confirmation echoing orderListId, contingencyType, listStatusType, listOrderStatus,
    listClientOrderId and a `### Legs` table of all three legs exactly as Binance reported
    them (the pending pair shows as `PENDING_NEW` until the working leg fills).

    Examples:
        params = {"symbol": "BTCUSDT", "working_type": "LIMIT", "working_side": "BUY",
                  "working_price": "60000.00", "working_quantity": "0.001",
                  "working_time_in_force": "GTC", "pending_side": "SELL",
                  "pending_quantity": "0.001",
                  "pending_above_type": "LIMIT_MAKER", "pending_above_price": "66000.00",
                  "pending_below_type": "STOP_LOSS_LIMIT", "pending_below_price": "57000.00",
                  "pending_below_stop_price": "57500.00", "pending_below_time_in_force": "GTC",
                  "list_client_order_id": "full-bracket-001"}

    Error Handling:
    - `Error: … trading is disabled …` → the kill-switch is off; nothing was sent.
    - -2010 / -1013 / -1111 are balance, filter and precision failures on any leg.
    - -2021 means a LIMIT_MAKER leg would have taken liquidity immediately.
    - **A 5xx or a timeout means the execution status is UNKNOWN** — query with
      `binance_get_order_list` before retrying.
    """
    try:
        client = get_client()
        resp = await client.request("POST", "/api/v3/orderList/otoco", auth="signed", params=params.to_params())
        return clip_response("\n".join(_placement_lines(resp.json(), params.symbol, "OTOCO")))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_order_list",
    annotations=ToolAnnotations(
        title="Binance Get Order List",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_order_list(params: GetOrderListInput) -> str:
    """Look up one order list — OCO, OTO or OTOCO — by id.

    Calls `GET /api/v3/orderList` (SIGNED, IP weight 4). No symbol is needed: pass exactly
    one id, `order_list_id` (Binance's numeric orderListId) or `orig_client_order_id` (the
    listClientOrderId used when placing). Both at once is rejected locally — Binance
    resolves the numeric id first and then rejects a mismatch, so the ambiguous call buys
    nothing.

    When to Use:
    - **After a 5xx or a timeout on a placement** — this is how you find out whether the
      list exists before considering a retry.
    - To check whether a bracket is still working, or which leg ended it.

    When NOT to Use:
    - To see every working list — `binance_get_open_order_lists` (weight 6).
    - For history across a period — `binance_get_all_order_lists`.
    - For an individual leg's fills — `binance_get_order` / `binance_get_my_trades` with
      the leg's orderId.

    Returns:
    orderListId, contingencyType, listStatusType, listOrderStatus, listClientOrderId,
    symbol, transactionTime and the `### Legs` table. This endpoint returns `orders[]`
    only (ids, no per-leg status), and the answer says so rather than implying more.
    `response_format="json"` returns the raw payload.

    Examples:
        params = {"order_list_id": 27}
        params = {"orig_client_order_id": "btc-bracket-001"}

    Error Handling:
    -2011/-2013 mean no such order list for this account. -1102 means neither id reached
    Binance. -2015 means the key lacks permission or this IP is not allowlisted.
    """
    try:
        query = _single_list_id(
            params.order_list_id, params.orig_client_order_id, "orig_client_order_id", "origClientOrderId"
        )
        client = get_client()
        resp = await client.request("GET", "/api/v3/orderList", auth="signed", params=query)
        data = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        lines = [f"# Order list {data.get('orderListId', '?')} on {data.get('symbol', 'N/A')}", ""]
        lines.extend(_list_detail_lines(data))
        lines.extend(_legs_lines(data))
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_cancel_order_list",
    annotations=ToolAnnotations(
        title="Binance Cancel Order List",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_cancel_order_list(params: CancelOrderListInput) -> str:
    """Cancel an ENTIRE order list — every leg of it — by id.

    Calls `DELETE /api/v3/orderList` (SIGNED, IP weight 1). Pass the symbol plus exactly
    one id: `order_list_id` or `list_client_order_id`. Both at once is rejected locally.

    **Kill-switch.** Refused with `Error: … trading is disabled …` unless the server runs
    with `BINANCE_ALLOW_TRADING=1`.

    Cancelling is idempotent in effect: a second cancel of the same list returns -2011
    ("unknown order") and changes nothing. What it cannot undo is a fill — a leg that has
    already triggered is gone, and its sibling with it. Note that cancelling ONE leg (via
    `binance_cancel_order`) also cancels the whole list; this tool just makes the intent
    explicit.

    When to Use:
    - Pulling a bracket that is no longer wanted, before replacing it.
    - Cleaning up after a partial fill changed the position the bracket was sized for.

    When NOT to Use:
    - To cancel everything on a symbol, lists included — `binance_cancel_all_open_orders`
      (spot_orders.py) does it in one call.
    - To see what would be cancelled — `binance_get_order_list` first; that read is free
      of consequence and this one is not.

    Returns:
    A confirmation echoing the cancelled list: orderListId, contingencyType,
    listStatusType, listOrderStatus and the `### Legs` table from `orderReports[]` with
    each leg's status (`CANCELED`) and its original client id.

    Examples:
        params = {"symbol": "BTCUSDT", "order_list_id": 27}
        params = {"symbol": "BTCUSDT", "list_client_order_id": "btc-bracket-001",
                  "new_client_order_id": "cancel-bracket-001"}

    Error Handling:
    -2011 means the list is not cancellable: it does not exist, already completed, or was
    already cancelled. A 5xx/timeout leaves the cancel UNKNOWN: check with
    `binance_get_order_list` before assuming either way — do not assume the legs are gone.
    """
    try:
        query: dict[str, Any] = {"symbol": params.symbol}
        query.update(
            _single_list_id(
                params.order_list_id, params.list_client_order_id, "list_client_order_id", "listClientOrderId"
            )
        )
        if params.new_client_order_id is not None:
            query["newClientOrderId"] = params.new_client_order_id
        client = get_client()
        resp = await client.request("DELETE", "/api/v3/orderList", auth="signed", params=query)
        data = resp.json()
        lines = [f"# Order list cancelled on {data.get('symbol', params.symbol)}", ""]
        lines.extend(_list_detail_lines(data))
        lines.extend(_legs_lines(data))
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_all_order_lists",
    annotations=ToolAnnotations(
        title="Binance All Order Lists",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_all_order_lists(params: GetAllOrderListsInput) -> str:
    """List this account's order lists — working, completed and cancelled — across symbols.

    Calls `GET /api/v3/allOrderList` (SIGNED, IP weight 20). There is **no symbol filter**:
    the endpoint is account-wide. Narrow it with `from_id` (lists with orderListId >= it)
    **or** a `start_time`/`end_time` window — Binance forbids combining them, and that is
    rejected locally. **The window may not exceed 24 hours**, also checked here so you get
    a clear message instead of Binance's -1127.

    When to Use:
    - Reconstructing which brackets existed during a given day.
    - Paging list history forward with a `from_id` cursor.

    When NOT to Use:
    - For what is armed right now — `binance_get_open_order_lists` costs weight 6.
    - For one known list — `binance_get_order_list` costs weight 4.
    - For plain (non-list) orders — `binance_get_all_orders` (spot_orders.py).

    Pagination:
    `limit` is 1-1000 (Binance default 500) and at most 50 rows are rendered; use
    `response_format="json"` or narrow the window for the rest. `from_id` pages forward:
    pass the last orderListId you saw, plus one.

    Windows:
    `start_time`/`end_time` accept epoch ms or ISO-8601 and must span 24 h or less
    together. Omit everything to get the most recent `limit` lists.

    Examples:
        params = {"limit": 10}
        params = {"start_time": "2026-09-22T00:00:00Z", "end_time": "2026-09-22T23:59:59Z"}
        params = {"from_id": 27}

    Error Handling:
    A window wider than 24 h, and `from_id` combined with a time bound, are both rejected
    locally. -1127 from Binance means a too-wide window reached it anyway; -1128 is an
    invalid parameter combination. -2015 means the key lacks permission or this IP is not
    allowlisted.
    """
    try:
        start_ms = _to_ms(params.start_time, "start_time")
        end_ms = _to_ms(params.end_time, "end_time")
        if params.from_id is not None and (start_ms is not None or end_ms is not None):
            return (
                "Error: from_id cannot be combined with start_time/end_time — Binance's allOrderList "
                "endpoint accepts either the id cursor or the time window, not both. Drop one."
            )
        if start_ms is not None and end_ms is not None:
            if end_ms <= start_ms:
                return "Error: end_time must be after start_time."
            if end_ms - start_ms > _ORDER_LIST_WINDOW_MS:
                hours = (end_ms - start_ms) / (60 * 60 * 1000)
                return (
                    f"Error: the start_time/end_time window spans ~{hours:.1f} hours; Binance's allOrderList "
                    "endpoint caps it at 24 hours. Slice the range into 24 h windows, or page with from_id."
                )
        query: dict[str, Any] = {"limit": params.limit}
        if params.from_id is not None:
            query["fromId"] = params.from_id
        if start_ms is not None:
            query["startTime"] = start_ms
        if end_ms is not None:
            query["endTime"] = end_ms
        client = get_client()
        resp = await client.request("GET", "/api/v3/allOrderList", auth="signed", params=query)
        payload = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(payload))
        rows = payload if isinstance(payload, list) else []
        return clip_response("\n".join(_order_list_table(rows, "Order lists (all symbols)")))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_open_order_lists",
    annotations=ToolAnnotations(
        title="Binance Open Order Lists",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_open_order_lists(params: GetOpenOrderListsInput) -> str:
    """List the order lists that are still working, across every symbol.

    Calls `GET /api/v3/openOrderList` (SIGNED, IP weight 6). The endpoint takes no
    filters — it is account-wide by construction, which is exactly what makes it the right
    first read before placing another bracket.

    When to Use:
    - Before placing a new bracket, to see what is already armed on the same position.
    - After a 5xx/timeout on a placement, when you have no id to query.

    When NOT to Use:
    - For one known list — `binance_get_order_list` (weight 4).
    - For lists that already finished — `binance_get_all_order_lists`.
    - For plain open orders — `binance_get_open_orders` (spot_orders.py); a list's legs
      also appear there individually.

    Returns:
    A markdown table (time, symbol, orderListId, contingencyType, listStatusType,
    listOrderStatus, listClientOrderId, leg count) capped at 50 rows, or the raw array
    with `response_format="json"`.

    Examples:
        params = {}
        params = {"response_format": "json"}

    Error Handling:
    -2015 means the key lacks permission or this IP is not allowlisted. An empty list is a
    valid answer: no bracket is armed.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/api/v3/openOrderList", auth="signed", params={})
        payload = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(payload))
        rows = payload if isinstance(payload, list) else []
        return clip_response("\n".join(_order_list_table(rows, "Open order lists (all symbols)")))
    except Exception as exc:
        return handle_api_error(exc)
