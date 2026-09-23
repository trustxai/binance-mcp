"""Spot order placement, query and cancellation (inventory C) — `/api/v3` TRADE endpoints.

Eight tools built on one shared order model:

- `binance_test_order`             POST /api/v3/order/test      (dry-run, always allowed)
- `binance_place_order`            POST /api/v3/order           (**real money**)
- `binance_get_order`              GET  /api/v3/order
- `binance_cancel_order`           DELETE /api/v3/order
- `binance_cancel_all_open_orders` DELETE /api/v3/openOrders
- `binance_cancel_replace_order`   POST /api/v3/order/cancelReplace
- `binance_get_open_orders`        GET  /api/v3/openOrders
- `binance_get_all_orders`         GET  /api/v3/allOrders

Every call is SIGNED. The mutating ones additionally pass through the **kill-switch that
lives in `binance_mcp.client`**, not here: a signed non-GET request is refused with
`TradingDisabledError` (surfaced as `Error: … trading is disabled …`) unless
`BINANCE_ALLOW_TRADING=1`. `POST /api/v3/order/test` is on the client's
`POST_READ_ALLOWLIST`, so the dry-run works even with trading switched off — which is
why every placement docstring tells the caller to validate there first.

Prices and quantities are **strings** end to end. They are validated as positive decimals
and forwarded verbatim: Binance's LOT_SIZE / PRICE_FILTER / NOTIONAL filters are
precision-sensitive and a float round-trip (0.1 + 0.2) silently breaks them.

The per-type mandatory parameter sets are enforced locally by `_OrderParams`, so a
malformed order fails with a readable message before any request is signed or sent.
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

# `allOrders` rejects a start/end span of more than 24 h (inventory C / S2 L4367);
# checked before the call so the caller gets a clear message, not Binance's -1127.
_ORDER_WINDOW_MS = 24 * 60 * 60 * 1000

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,20}$")


# -- enums ---------------------------------------------------------------------------


class OrderSide(StrEnum):
    """Which way the order goes."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """The seven spot order types; each has its own mandatory parameter set."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"


class TimeInForce(StrEnum):
    """How long the order stays on the book: GTC rests, IOC/FOK do not."""

    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class NewOrderRespType(StrEnum):
    """How much of the result Binance returns (FULL includes the fills)."""

    ACK = "ACK"
    RESULT = "RESULT"
    FULL = "FULL"


class SelfTradePreventionMode(StrEnum):
    """What Binance does when this order would match the account's own resting order."""

    NONE = "NONE"
    EXPIRE_TAKER = "EXPIRE_TAKER"
    EXPIRE_MAKER = "EXPIRE_MAKER"
    EXPIRE_BOTH = "EXPIRE_BOTH"
    DECREMENT = "DECREMENT"


class CancelRestrictions(StrEnum):
    """Refuse the cancel unless the order is in this state (avoids racing a fill)."""

    ONLY_NEW = "ONLY_NEW"
    ONLY_PARTIALLY_FILLED = "ONLY_PARTIALLY_FILLED"


class CancelReplaceMode(StrEnum):
    """Whether the replacement is attempted when the cancel fails."""

    STOP_ON_FAILURE = "STOP_ON_FAILURE"
    ALLOW_FAILURE = "ALLOW_FAILURE"


class OrderRateLimitExceededMode(StrEnum):
    """What to do when the unfilled-order-count limit is already exceeded."""

    DO_NOTHING = "DO_NOTHING"
    CANCEL_ONLY = "CANCEL_ONLY"


# -- helpers -------------------------------------------------------------------------


def _as_str(value: Any) -> str:
    return str(value)


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


def _require_fields(params: _OrderParams, names: tuple[str, ...], binance_names: str) -> None:
    """Raise a readable ValueError naming every missing field of a mandatory set."""
    missing = [name for name in names if getattr(params, name) is None]
    if missing:
        raise ValueError(
            f"{params.type.value} orders require {', '.join(names)} (Binance: {binance_names}); "
            f"missing: {', '.join(missing)}."
        )


def _single_order_id(order_id: int | None, orig_client_order_id: str | None) -> dict[str, Any]:
    """Return the id param for an order lookup/cancel — exactly one id, checked here.

    Binance accepts both and searches `orderId` first, which silently ignores a
    mismatched client id; refusing the ambiguous call is safer on a mutating path.
    """
    if (order_id is None) == (orig_client_order_id is None):
        raise RuntimeError(
            "Pass exactly one of order_id (Binance: orderId) or orig_client_order_id "
            "(Binance: origClientOrderId) — not both, not neither. Binance searches orderId first when "
            "both are sent, so a mismatched client id would be ignored rather than rejected."
        )
    if order_id is not None:
        return {"orderId": order_id}
    return {"origClientOrderId": orig_client_order_id}


def _order_detail_lines(order: dict[str, Any]) -> list[str]:
    """Render only the fields Binance actually returned — never invent a status."""
    lines: list[str] = []

    def add(label: str, key: str, fmt: Callable[[Any], str] = _as_str) -> None:
        value = order.get(key)
        if value is not None:
            lines.append(f"- **{label}**: {fmt(value)}")

    add("symbol", "symbol")
    add("orderId", "orderId")
    add("clientOrderId", "clientOrderId")
    add("origClientOrderId", "origClientOrderId")
    add("status", "status")
    add("side", "side")
    add("type", "type")
    add("timeInForce", "timeInForce")
    add("price", "price", fmt_num)
    add("stopPrice", "stopPrice", fmt_num)
    add("trailingDelta", "trailingDelta")
    add("origQty", "origQty", fmt_num)
    add("executedQty", "executedQty", fmt_num)
    add("cummulativeQuoteQty", "cummulativeQuoteQty", fmt_num)
    add("origQuoteOrderQty", "origQuoteOrderQty", fmt_num)
    add("icebergQty", "icebergQty", fmt_num)
    add("selfTradePreventionMode", "selfTradePreventionMode")
    add("isWorking", "isWorking")
    if order.get("orderListId") not in (None, -1):
        lines.append(f"- **orderListId**: {order['orderListId']} (leg of an order list)")
    add("time", "time", epoch_to_human)
    add("transactTime", "transactTime", epoch_to_human)
    add("updateTime", "updateTime", epoch_to_human)
    add("workingTime", "workingTime", epoch_to_human)
    return lines


def _fills_lines(fills: list[dict[str, Any]]) -> list[str]:
    """Render the `fills[]` array of a FULL response plus its commission totals."""
    if not fills:
        return []
    lines = ["", "**Fills**", "", "| price | qty | commission | commission asset | tradeId |", "|---|---|---|---|---|"]
    for fill in fills[:MAX_DISPLAY_ROWS]:
        lines.append(
            f"| {fmt_num(fill.get('price'))} | {fmt_num(fill.get('qty'))} | "
            f"{fmt_num(fill.get('commission'))} | {fill.get('commissionAsset', 'N/A')} | "
            f"{fill.get('tradeId', 'N/A')} |"
        )
    if len(fills) > MAX_DISPLAY_ROWS:
        lines.append(f"_[{len(fills) - MAX_DISPLAY_ROWS} more fill(s) not shown]_")
    commissions: dict[str, Decimal] = {}
    for fill in fills:
        asset = str(fill.get("commissionAsset") or "?")
        try:
            commissions[asset] = commissions.get(asset, Decimal(0)) + Decimal(str(fill.get("commission", "0")))
        except InvalidOperation:  # pragma: no cover - Binance always sends decimal strings
            continue
    if commissions:
        totals = ", ".join(f"{fmt_num(amount)} {asset}" for asset, amount in sorted(commissions.items()))
        lines.append("")
        lines.append(f"_Commission across the fills above: {totals}._")
    return lines


def _order_table(orders: list[dict[str, Any]], title: str) -> list[str]:
    """Compact table for the list tools (open orders / all orders)."""
    lines = [f"# {title}", ""]
    if not orders:
        lines.append("_No orders._")
        return lines
    lines.append(f"Showing **{min(len(orders), MAX_DISPLAY_ROWS):,}** of **{len(orders):,}** order(s).")
    lines.append("")
    lines.append("| time | symbol | orderId | side | type | status | price | origQty | executedQty | cumQuote |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for order in orders[:MAX_DISPLAY_ROWS]:
        stamp = order.get("time") or order.get("transactTime") or order.get("updateTime")
        lines.append(
            f"| {epoch_to_human(stamp)} | {order.get('symbol', 'N/A')} | {order.get('orderId', 'N/A')} | "
            f"{order.get('side', 'N/A')} | {order.get('type', 'N/A')} | {order.get('status', 'N/A')} | "
            f"{fmt_num(order.get('price'))} | {fmt_num(order.get('origQty'))} | "
            f"{fmt_num(order.get('executedQty'))} | {fmt_num(order.get('cummulativeQuoteQty'))} |"
        )
    if len(orders) > MAX_DISPLAY_ROWS:
        lines.append("")
        lines.append(
            f"_[{len(orders) - MAX_DISPLAY_ROWS} more order(s) not shown — narrow with symbol/time or use "
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


class _OrderParams(_SymbolInput):
    """The new-order parameter set shared by test / place / cancel-replace.

    The `model_validator` below enforces Binance's per-type mandatory sets (inventory C,
    S2 L2170) locally, so a malformed order never reaches the signing path.
    """

    side: OrderSide = Field(description="BUY or SELL.")
    type: OrderType = Field(
        description=(
            "Order type. Mandatory extras per type — LIMIT: time_in_force + quantity + price; "
            "MARKET: quantity OR quote_order_qty (exactly one); STOP_LOSS / TAKE_PROFIT: quantity + "
            "(stop_price OR trailing_delta, exactly one); STOP_LOSS_LIMIT / TAKE_PROFIT_LIMIT: "
            "time_in_force + quantity + price + stop_price and/or trailing_delta; "
            "LIMIT_MAKER: quantity + price."
        )
    )
    time_in_force: TimeInForce | None = Field(
        default=None,
        description="GTC (rest on the book), IOC (fill what you can now) or FOK (all or nothing).",
    )
    quantity: str | None = Field(
        default=None,
        description="Base-asset amount as a decimal STRING (e.g. '0.001' BTC). Sent verbatim — must respect LOT_SIZE.",
    )
    quote_order_qty: str | None = Field(
        default=None,
        description=(
            "MARKET only: spend/receive this much QUOTE asset instead of a base quantity "
            "(e.g. '50' = 50 USDT). Binance name: quoteOrderQty."
        ),
    )
    price: str | None = Field(
        default=None,
        description="Limit price as a decimal STRING. Sent verbatim — must respect PRICE_FILTER tick size.",
    )
    stop_price: str | None = Field(
        default=None,
        description="Trigger price for the STOP_LOSS / TAKE_PROFIT family, as a decimal STRING. Binance: stopPrice.",
    )
    trailing_delta: str | None = Field(
        default=None,
        description=(
            "Trailing-stop distance in BIPS as a STRING ('100' = 1 percent). Binance: trailingDelta. "
            "An alternative trigger to stop_price."
        ),
    )
    iceberg_qty: str | None = Field(
        default=None,
        description="Visible slice of the order as a decimal STRING; requires time_in_force=GTC. Binance: icebergQty.",
    )
    new_client_order_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=36,
        description=(
            "Your own id for this order (Binance: newClientOrderId). Reused only after the previous "
            "order with that id has filled. Use it to make a retry traceable."
        ),
    )
    new_order_resp_type: NewOrderRespType | None = Field(
        default=None,
        description=(
            "How much detail Binance returns: ACK (ids only), RESULT (adds status/quantities) or "
            "FULL (adds the fills). MARKET and LIMIT default to FULL. Binance: newOrderRespType."
        ),
    )
    self_trade_prevention_mode: SelfTradePreventionMode | None = Field(
        default=None,
        description=(
            "What Binance does if this order would match your own resting order: NONE, EXPIRE_TAKER, "
            "EXPIRE_MAKER, EXPIRE_BOTH or DECREMENT. Binance: selfTradePreventionMode."
        ),
    )

    @field_validator("quantity", "quote_order_qty", "price", "stop_price", "trailing_delta", "iceberg_qty")
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

    @model_validator(mode="after")
    def _check_type_mandatory_set(self) -> Self:
        order_type = self.type
        if order_type is OrderType.LIMIT:
            _require_fields(self, ("time_in_force", "quantity", "price"), "timeInForce, quantity, price")
        elif order_type is OrderType.MARKET:
            if (self.quantity is None) == (self.quote_order_qty is None):
                raise ValueError(
                    "MARKET orders require exactly one of quantity (base asset) or quote_order_qty "
                    "(quote asset; Binance: quoteOrderQty) — not both, not neither."
                )
        elif order_type in (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT):
            _require_fields(self, ("quantity",), "quantity")
            if (self.stop_price is None) == (self.trailing_delta is None):
                raise ValueError(
                    f"{order_type.value} orders require exactly one trigger: stop_price (Binance: stopPrice) "
                    "or trailing_delta (Binance: trailingDelta, in BIPS) — not both, not neither."
                )
        elif order_type in (OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT_LIMIT):
            _require_fields(self, ("time_in_force", "quantity", "price"), "timeInForce, quantity, price")
            if self.stop_price is None and self.trailing_delta is None:
                raise ValueError(
                    f"{order_type.value} orders require a trigger: stop_price (Binance: stopPrice) "
                    "and/or trailing_delta (Binance: trailingDelta, in BIPS)."
                )
        elif order_type is OrderType.LIMIT_MAKER:
            _require_fields(self, ("quantity", "price"), "quantity, price")
        if self.iceberg_qty is not None and self.time_in_force is not TimeInForce.GTC:
            raise ValueError(
                "iceberg_qty (Binance: icebergQty) is only accepted with time_in_force=GTC; "
                f"got time_in_force={self.time_in_force.value if self.time_in_force else None}."
            )
        return self

    def to_params(self) -> dict[str, Any]:
        """Map to Binance's camelCase query parameters, dropping everything unset."""
        raw: dict[str, Any] = {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.type.value,
            "timeInForce": self.time_in_force.value if self.time_in_force is not None else None,
            "quantity": self.quantity,
            "quoteOrderQty": self.quote_order_qty,
            "price": self.price,
            "stopPrice": self.stop_price,
            "trailingDelta": self.trailing_delta,
            "icebergQty": self.iceberg_qty,
            "newClientOrderId": self.new_client_order_id,
            "newOrderRespType": self.new_order_resp_type.value if self.new_order_resp_type is not None else None,
            "selfTradePreventionMode": (
                self.self_trade_prevention_mode.value if self.self_trade_prevention_mode is not None else None
            ),
        }
        return {key: value for key, value in raw.items() if value is not None}


class TestOrderInput(_OrderParams):
    """Params for `POST /api/v3/order/test` — the same order, never executed."""

    # The class name starts with "Test", so pytest would try to collect it as a test
    # class when the test module imports it; this opts out of that.
    __test__ = False

    compute_commission_rates: bool = Field(
        default=False,
        description=(
            "Also return the commission rates that would apply to this order "
            "(Binance: computeCommissionRates). Raises the weight from 1 to 20."
        ),
    )


class PlaceOrderInput(_OrderParams):
    """Params for `POST /api/v3/order` — this spends real money."""


class CancelReplaceOrderInput(_OrderParams):
    """Params for `POST /api/v3/order/cancelReplace`: cancel one order, place another."""

    cancel_replace_mode: CancelReplaceMode = Field(
        description=(
            "STOP_ON_FAILURE (do not place the new order if the cancel fails) or ALLOW_FAILURE "
            "(place it anyway). Binance: cancelReplaceMode."
        )
    )
    cancel_order_id: int | None = Field(
        default=None,
        description="Binance orderId of the order to cancel. Pass this OR cancel_orig_client_order_id.",
    )
    cancel_orig_client_order_id: str | None = Field(
        default=None,
        min_length=1,
        description="Your client id of the order to cancel. Pass this OR cancel_order_id.",
    )
    cancel_restrictions: CancelRestrictions | None = Field(
        default=None,
        description=(
            "Only cancel when the order is in this state: ONLY_NEW or ONLY_PARTIALLY_FILLED. "
            "Binance: cancelRestrictions; a mismatch fails the cancel with -2011."
        ),
    )
    order_rate_limit_exceeded_mode: OrderRateLimitExceededMode | None = Field(
        default=None,
        description=(
            "What to do when the unfilled-order-count limit is already exceeded: DO_NOTHING "
            "(reject both) or CANCEL_ONLY (cancel, skip the replacement). "
            "Binance: orderRateLimitExceededMode."
        ),
    )

    @model_validator(mode="after")
    def _check_cancel_id(self) -> Self:
        if (self.cancel_order_id is None) == (self.cancel_orig_client_order_id is None):
            raise ValueError(
                "Pass exactly one of cancel_order_id (Binance: cancelOrderId) or cancel_orig_client_order_id "
                "(Binance: cancelOrigClientOrderId) — not both, not neither."
            )
        return self

    def to_cancel_replace_params(self) -> dict[str, Any]:
        """New-order params plus the cancel half; None dropped."""
        params = self.to_params()
        params["cancelReplaceMode"] = self.cancel_replace_mode.value
        if self.cancel_order_id is not None:
            params["cancelOrderId"] = self.cancel_order_id
        if self.cancel_orig_client_order_id is not None:
            params["cancelOrigClientOrderId"] = self.cancel_orig_client_order_id
        if self.cancel_restrictions is not None:
            params["cancelRestrictions"] = self.cancel_restrictions.value
        if self.order_rate_limit_exceeded_mode is not None:
            params["orderRateLimitExceededMode"] = self.order_rate_limit_exceeded_mode.value
        return params


class GetOrderInput(_SymbolInput):
    """Params for `GET /api/v3/order`."""

    order_id: int | None = Field(default=None, description="Binance orderId. Pass this OR orig_client_order_id.")
    orig_client_order_id: str | None = Field(
        default=None,
        min_length=1,
        description="The clientOrderId you sent when placing. Pass this OR order_id. Binance: origClientOrderId.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: `markdown` (human-readable) or `json` (raw Binance payload).",
    )


class CancelOrderInput(_SymbolInput):
    """Params for `DELETE /api/v3/order`."""

    order_id: int | None = Field(default=None, description="Binance orderId. Pass this OR orig_client_order_id.")
    orig_client_order_id: str | None = Field(
        default=None,
        min_length=1,
        description="The clientOrderId of the order to cancel. Pass this OR order_id. Binance: origClientOrderId.",
    )
    new_client_order_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=36,
        description="Your own id for the cancel itself (Binance: newClientOrderId); auto-generated if omitted.",
    )
    cancel_restrictions: CancelRestrictions | None = Field(
        default=None,
        description=(
            "Refuse the cancel unless the order is ONLY_NEW or ONLY_PARTIALLY_FILLED — the safe way to "
            "avoid racing a fill. Binance: cancelRestrictions; a mismatch returns -2011."
        ),
    )


class CancelAllOpenOrdersInput(_SymbolInput):
    """Params for `DELETE /api/v3/openOrders` — every open order on one symbol."""


class GetOpenOrdersInput(_BaseInput):
    """Params for `GET /api/v3/openOrders`."""

    symbol: str | None = Field(
        default=None,
        description="Trading pair, e.g. `BTCUSDT`. Omit for EVERY symbol — that costs IP weight 80 instead of 6.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: `markdown` (human-readable) or `json` (raw Binance payload).",
    )

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, value: str | None) -> str | None:
        return _normalize_symbol(value) if value is not None else None


class GetAllOrdersInput(_SymbolInput):
    """Params for `GET /api/v3/allOrders`."""

    order_id: int | None = Field(
        default=None,
        ge=0,
        description="Cursor: return orders with orderId >= this. Omit for the most recent orders.",
    )
    start_time: int | str | None = Field(
        default=None,
        description="Window start: epoch ms or ISO-8601. With end_time, the span must be 24 h or less.",
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end: epoch ms or ISO-8601. With start_time, the span must be 24 h or less.",
    )
    limit: int = Field(default=500, ge=1, le=1000, description="Orders to fetch, 1-1000 (Binance default 500).")
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: `markdown` (human-readable) or `json` (raw Binance payload).",
    )


# -- tools ---------------------------------------------------------------------------


@mcp.tool(
    name="binance_test_order",
    annotations=ToolAnnotations(
        title="Binance Test Order (dry-run)",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_test_order(params: TestOrderInput) -> str:
    """Validate an order against Binance's filters WITHOUT sending it to the order book.

    Calls `POST /api/v3/order/test` (SIGNED, IP weight 1 — **20** with
    `compute_commission_rates`). Binance runs the same validation as a real placement
    (signature, recvWindow, symbol status, LOT_SIZE / PRICE_FILTER / NOTIONAL, balance
    rules) and returns `{}` on success: nothing is matched, nothing rests on the book,
    no funds move.

    **This tool is always allowed.** It is on the client's `POST_READ_ALLOWLIST`, so it
    works with the trading kill-switch off (`BINANCE_ALLOW_TRADING` unset) — which makes
    it the right first step before every `binance_place_order` call.

    When to Use:
    - Always, immediately before placing a real order, to catch a filter or precision
      error for free.
    - With `compute_commission_rates=true` to learn the fee rates that would apply.

    When NOT to Use:
    - To actually trade — that is `binance_place_order`.
    - To check a symbol's filters in the abstract — `binance_get_exchange_info`
      (market_data.py) lists tick size, step size and min notional directly.

    Returns:
    A confirmation that the order passed validation (and the commission-rate breakdown
    when requested). A rejection comes back as an `Error:` line quoting Binance's reason.

    Examples:
        params = {"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT",
                  "time_in_force": "GTC", "quantity": "0.001", "price": "20000.00"}
        params = {"symbol": "BTCUSDT", "side": "SELL", "type": "MARKET",
                  "quantity": "0.001", "compute_commission_rates": True}

    Error Handling:
    -1013/-2010 point at a symbol filter (step size, tick size, min notional); -1111 is a
    precision error — check `binance_get_exchange_info`. -2015 means the key lacks Spot
    Trading permission or this IP is not allowlisted. The per-type mandatory parameter
    sets are checked locally, so those failures never reach Binance.
    """
    try:
        query = params.to_params()
        if params.compute_commission_rates:
            query["computeCommissionRates"] = True
        client = get_client()
        resp = await client.request("POST", "/api/v3/order/test", auth="signed", params=query)
        data = resp.json()
        lines = [
            "# Order validation passed",
            "",
            f"Binance accepted this **{params.type.value} {params.side.value}** on **{params.symbol}** as valid.",
            "It was NOT sent to the order book — nothing was matched and no funds moved.",
            "",
            "Place it for real with `binance_place_order` using the same parameters "
            "(requires BINANCE_ALLOW_TRADING=1).",
        ]
        if isinstance(data, dict) and data:
            lines.append("")
            lines.append("## Commission rates for this order")
            lines.append("")
            lines.append(f"```json\n{to_json(data)}\n```")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_place_order",
    annotations=ToolAnnotations(
        title="Binance Place Spot Order",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def binance_place_order(params: PlaceOrderInput) -> str:
    """Place a REAL spot order on Binance. This spends real money.

    Calls `POST /api/v3/order` (SIGNED, IP weight 1, unfilled-order count 1). A MARKET
    order executes immediately at whatever the book offers; a LIMIT order rests until it
    fills, expires or is cancelled.

    **Kill-switch.** This call is refused with `Error: … trading is disabled …` unless
    the server runs with `BINANCE_ALLOW_TRADING=1`. The gate lives in the HTTP client, so
    no tool can bypass it. If you see that error, the operator has deliberately put the
    server in read-only mode — report it, do not try to work around it.

    **Always run `binance_test_order` first** with identical parameters: it is allowed
    even with the kill-switch off and catches filter/precision rejections for free.

    When to Use:
    - After a dry-run passed and the human has approved this specific order.
    - To act on a decision that already names symbol, side, type, quantity and price.

    When NOT to Use:
    - To "see if it would work" — that is `binance_test_order`.
    - For a bracket/OCO (entry plus stop plus target) — use the order-list tools in
      `order_lists.py`, which place the legs atomically.
    - To modify a resting order — use `binance_cancel_replace_order`, which does not
      leave you unhedged between the two calls.

    Returns:
    A confirmation echoing exactly what Binance returned: symbol, orderId,
    clientOrderId, status, executedQty, cummulativeQuoteQty, and the fills table when the
    response carries one. Nothing is inferred: with `new_order_resp_type="ACK"` Binance
    reports only the ids, and the confirmation says so rather than implying a fill.

    Examples:
        params = {"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT",
                  "time_in_force": "GTC", "quantity": "0.001", "price": "20000.00",
                  "new_client_order_id": "my-entry-001"}
        params = {"symbol": "BTCUSDT", "side": "SELL", "type": "MARKET", "quantity": "0.001"}
        params = {"symbol": "BTCUSDT", "side": "SELL", "type": "STOP_LOSS_LIMIT",
                  "time_in_force": "GTC", "quantity": "0.001", "price": "19000.00",
                  "stop_price": "19100.00"}

    Error Handling:
    - `Error: … trading is disabled …` → the kill-switch is off; nothing was sent.
    - **-2010 (order rejected)** → insufficient balance, or a symbol filter: quantity off
      the LOT_SIZE step, price off the PRICE_FILTER tick, or the order under the NOTIONAL
      minimum. Read the filters with `binance_get_exchange_info` and re-run the dry-run.
    - -1013 / -1111 are the same family (precision / filter).
    - -2021 means a LIMIT_MAKER would have taken liquidity immediately.
    - **A 5xx or a timeout means the execution status is UNKNOWN** — the order may well be
      live. Query it with `binance_get_order` (by `new_client_order_id` if you set one) or
      `binance_get_open_orders` before doing anything else. NEVER resend blindly: a
      duplicate market order is real money lost.
    """
    try:
        query = params.to_params()
        client = get_client()
        resp = await client.request("POST", "/api/v3/order", auth="signed", params=query)
        order = resp.json()
        lines = [f"# Order placed on {params.symbol}", ""]
        lines.extend(_order_detail_lines(order))
        fills = order.get("fills")
        if isinstance(fills, list):
            lines.extend(_fills_lines(fills))
        if "status" not in order:
            lines.append("")
            lines.append(
                "_Binance answered with an acknowledgement only (no status or quantities). The order was "
                "accepted; its execution state is unknown from this response — query it with "
                "`binance_get_order`._"
            )
        elif order.get("status") in ("NEW", "PARTIALLY_FILLED"):
            lines.append("")
            lines.append(
                f"_Status **{order['status']}**: the order is still working on the book. Track it with "
                "`binance_get_open_orders` or cancel it with `binance_cancel_order`._"
            )
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_order",
    annotations=ToolAnnotations(
        title="Binance Get Order",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_order(params: GetOrderInput) -> str:
    """Look up one order — open, filled, cancelled or expired — by id.

    Calls `GET /api/v3/order` (SIGNED, IP weight 4). Pass the symbol plus exactly one id:
    `order_id` (Binance's numeric orderId) or `orig_client_order_id` (the id you supplied
    when placing). Both at once is rejected locally: Binance searches orderId first and
    would silently ignore a mismatched client id.

    When to Use:
    - **After a 5xx or a timeout on a placement** — this is how you find out whether the
      order exists before considering a retry.
    - To check the final state of an order that is no longer open.

    When NOT to Use:
    - To list what is currently resting — use `binance_get_open_orders`.
    - To page through history — use `binance_get_all_orders`.
    - For the individual trades that filled the order — use `binance_get_my_trades`
      (trade_history.py).

    Returns:
    A markdown detail block (status, side, type, prices, quantities, timestamps), or the
    raw Binance object with `response_format="json"`.

    Examples:
        params = {"symbol": "BTCUSDT", "order_id": 123456789}
        params = {"symbol": "BTCUSDT", "orig_client_order_id": "my-entry-001"}

    Error Handling:
    -2011/-2013 mean no such order for that symbol — check the symbol, or the order may be
    older than 90 days and archived (-2026). Orders are scoped per symbol: the right id on
    the wrong symbol looks identical to a missing order.
    """
    try:
        query: dict[str, Any] = {"symbol": params.symbol}
        query.update(_single_order_id(params.order_id, params.orig_client_order_id))
        client = get_client()
        resp = await client.request("GET", "/api/v3/order", auth="signed", params=query)
        order = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(order))
        lines = [f"# Order {order.get('orderId', '?')} on {order.get('symbol', params.symbol)}", ""]
        lines.extend(_order_detail_lines(order))
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_cancel_order",
    annotations=ToolAnnotations(
        title="Binance Cancel Order",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_cancel_order(params: CancelOrderInput) -> str:
    """Cancel one open spot order by id.

    Calls `DELETE /api/v3/order` (SIGNED, IP weight 1). Pass the symbol plus exactly one
    id — `order_id` or `orig_client_order_id`; both at once is rejected locally because
    Binance would resolve the numeric id and ignore a mismatched client id.

    **Kill-switch.** Refused with `Error: … trading is disabled …` unless the server runs
    with `BINANCE_ALLOW_TRADING=1`.

    Cancelling is idempotent in effect: a second cancel of the same order returns -2011
    ("unknown order") and changes nothing. What it cannot undo is a fill — use
    `cancel_restrictions="ONLY_NEW"` to make the cancel fail rather than succeed against
    an order that has already started filling.

    When to Use:
    - To pull a resting order that is no longer wanted.
    - Before replacing an order, when you do not need the atomicity of
      `binance_cancel_replace_order`.

    When NOT to Use:
    - To cancel everything on a symbol — use `binance_cancel_all_open_orders` (one call,
      one weight unit).
    - To cancel one leg of an OCO/OTO list — that cancels the whole list; use
      `binance_cancel_order_list` (order_lists.py) so the intent is explicit.

    Returns:
    A confirmation echoing Binance's cancelled-order object: symbol, orderId,
    origClientOrderId, status (`CANCELED`), and the executed quantities at cancellation.

    Examples:
        params = {"symbol": "BTCUSDT", "order_id": 123456789}
        params = {"symbol": "BTCUSDT", "orig_client_order_id": "my-entry-001",
                  "cancel_restrictions": "ONLY_NEW"}

    Error Handling:
    -2011 means the order is not cancellable: it does not exist, already filled, was
    already cancelled — or `cancel_restrictions` did not match its current state, which is
    the safe outcome, not a failure. A 5xx/timeout leaves the cancel UNKNOWN: check with
    `binance_get_order` before assuming the order is still live.
    """
    try:
        query: dict[str, Any] = {"symbol": params.symbol}
        query.update(_single_order_id(params.order_id, params.orig_client_order_id))
        if params.new_client_order_id is not None:
            query["newClientOrderId"] = params.new_client_order_id
        if params.cancel_restrictions is not None:
            query["cancelRestrictions"] = params.cancel_restrictions.value
        client = get_client()
        resp = await client.request("DELETE", "/api/v3/order", auth="signed", params=query)
        order = resp.json()
        lines = [f"# Order cancelled on {order.get('symbol', params.symbol)}", ""]
        lines.extend(_order_detail_lines(order))
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_cancel_all_open_orders",
    annotations=ToolAnnotations(
        title="Binance Cancel All Open Orders",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_cancel_all_open_orders(params: CancelAllOpenOrdersInput) -> str:
    """Cancel EVERY open order on one symbol, including order-list legs.

    Calls `DELETE /api/v3/openOrders` (SIGNED, IP weight 1). This is a blunt instrument:
    it takes no id and cancels whatever is resting on that symbol, OCO/OTO lists included
    (their legs come back as order-list objects with `orderReports[]`).

    **Kill-switch.** Refused with `Error: … trading is disabled …` unless the server runs
    with `BINANCE_ALLOW_TRADING=1`.

    When to Use:
    - Flattening the working orders on one symbol — a stop-out or a strategy reset.
    - When several orders must go and cancelling them one by one would race the market.

    When NOT to Use:
    - When one specific order should go — use `binance_cancel_order` with an id.
    - To see what would be cancelled first — call `binance_get_open_orders` with the same
      symbol; that read is free of consequence and this one is not.

    Returns:
    A confirmation listing every cancelled order, plus a section per cancelled order list
    (orderListId, contingencyType, and each leg from `orderReports[]`).

    Examples:
        params = {"symbol": "BTCUSDT"}

    Error Handling:
    -2011 means there was nothing open on that symbol. A 5xx/timeout leaves the outcome
    UNKNOWN — re-read with `binance_get_open_orders` rather than assuming either way.
    """
    try:
        client = get_client()
        resp = await client.request("DELETE", "/api/v3/openOrders", auth="signed", params={"symbol": params.symbol})
        payload = resp.json()
        rows = payload if isinstance(payload, list) else []
        orders = [row for row in rows if row.get("orderListId") in (None, -1)]
        lists = [row for row in rows if row.get("orderListId") not in (None, -1)]
        lines = [f"# Cancelled all open orders on {params.symbol}", ""]
        if not rows:
            lines.append("_Binance returned nothing — there were no open orders on this symbol._")
            return clip_response("\n".join(lines))
        lines.append(f"Binance cancelled **{len(orders):,}** order(s) and **{len(lists):,}** order list(s).")
        if orders:
            lines.append("")
            lines.extend(_order_table(orders, "Cancelled orders")[1:])
        for entry in lists[:MAX_DISPLAY_ROWS]:
            lines.append("")
            lines.append(f"## Order list {entry.get('orderListId')} ({entry.get('contingencyType', 'N/A')})")
            lines.append(f"- **listStatusType**: {entry.get('listStatusType', 'N/A')}")
            lines.append(f"- **listOrderStatus**: {entry.get('listOrderStatus', 'N/A')}")
            reports = entry.get("orderReports") or []
            if reports:
                lines.append("")
                lines.extend(_order_table(reports, "Legs")[1:])
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_cancel_replace_order",
    annotations=ToolAnnotations(
        title="Binance Cancel and Replace Order",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def binance_cancel_replace_order(params: CancelReplaceOrderInput) -> str:
    """Cancel one order and place its replacement in a single request.

    Calls `POST /api/v3/order/cancelReplace` (SIGNED, IP weight 1, unfilled-order count
    1). Use it to reprice a resting order without the window of exposure that a separate
    cancel-then-place leaves open.

    **Kill-switch.** Refused with `Error: … trading is disabled …` unless the server runs
    with `BINANCE_ALLOW_TRADING=1`.

    The two halves can diverge, and `cancel_replace_mode` decides how:
    - `STOP_ON_FAILURE` — if the cancel fails, the new order is never attempted.
    - `ALLOW_FAILURE` — the new order is attempted regardless of the cancel's outcome, so
      you can end up with both orders live, or neither.

    **HTTP 409 is the partial-success case**: the cancel succeeded and the new order
    failed. It is returned as `Error (409): Partial success …` — read it as "the old order
    is gone, the replacement is NOT live" and re-place deliberately.

    When to Use:
    - Repricing or resizing a resting limit order.
    - Rolling a stop as the market moves.

    When NOT to Use:
    - For a fresh order with nothing to cancel — use `binance_place_order`.
    - To only pull an order — use `binance_cancel_order`.
    - On an order-list leg — cancel the list with `binance_cancel_order_list`
      (order_lists.py) and place a new list.

    Returns:
    `cancelResult` and `newOrderResult` (SUCCESS / FAILURE / NOT_ATTEMPTED) plus the two
    response objects Binance returned, rendered separately so it is unambiguous which
    order is live.

    Examples:
        params = {"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT",
                  "time_in_force": "GTC", "quantity": "0.001", "price": "19500.00",
                  "cancel_replace_mode": "STOP_ON_FAILURE", "cancel_order_id": 123456789}
        params = {"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT",
                  "time_in_force": "GTC", "quantity": "0.002", "price": "19000.00",
                  "cancel_replace_mode": "ALLOW_FAILURE",
                  "cancel_orig_client_order_id": "my-entry-001",
                  "cancel_restrictions": "ONLY_NEW"}

    Error Handling:
    HTTP 409 = cancel succeeded, replacement failed (see above). -2021/-2022 wrap the
    failing half in `{code, msg, data}`. -2011 means the order to cancel was not
    cancellable (filled, gone, or `cancel_restrictions` did not match). A 5xx/timeout
    leaves BOTH halves UNKNOWN: read `binance_get_open_orders` for the symbol before
    sending anything else.
    """
    try:
        query = params.to_cancel_replace_params()
        client = get_client()
        resp = await client.request("POST", "/api/v3/order/cancelReplace", auth="signed", params=query)
        data = resp.json()
        lines = [
            f"# Cancel-replace on {params.symbol}",
            "",
            f"- **cancelResult**: {data.get('cancelResult', 'N/A')}",
            f"- **newOrderResult**: {data.get('newOrderResult', 'N/A')}",
        ]
        cancel_response = data.get("cancelResponse")
        if isinstance(cancel_response, dict):
            lines.append("")
            lines.append("## Cancelled order")
            lines.append("")
            detail = _order_detail_lines(cancel_response)
            lines.extend(detail if detail else [f"```json\n{to_json(cancel_response)}\n```"])
        new_response = data.get("newOrderResponse")
        if isinstance(new_response, dict):
            lines.append("")
            lines.append("## New order")
            lines.append("")
            detail = _order_detail_lines(new_response)
            if detail:
                lines.extend(detail)
                fills = new_response.get("fills")
                if isinstance(fills, list):
                    lines.extend(_fills_lines(fills))
            else:
                lines.append(f"```json\n{to_json(new_response)}\n```")
        if data.get("newOrderResult") != "SUCCESS":
            lines.append("")
            lines.append(
                "_The replacement is NOT live. When cancelResult is SUCCESS the old order is gone either "
                "way — decide deliberately before placing anything else._"
            )
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_open_orders",
    annotations=ToolAnnotations(
        title="Binance Open Orders",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_open_orders(params: GetOpenOrdersInput) -> str:
    """List the orders currently resting on the book.

    Calls `GET /api/v3/openOrders` (SIGNED). **Weight 6 with `symbol`, 80 without** — the
    no-symbol form scans every pair and costs more than a percent of the 6000/min IP
    budget in one call. Pass a symbol whenever you know it.

    When to Use:
    - To see what is working right now, before placing or cancelling anything.
    - After a 5xx/timeout on a placement, as a symbol-wide check when you have no id.

    When NOT to Use:
    - For one known order — `binance_get_order` costs weight 4 and is precise.
    - For orders that are no longer open — `binance_get_all_orders` covers history.

    Returns:
    A markdown table (time, symbol, orderId, side, type, status, price, origQty,
    executedQty, cumQuote) capped at 50 rows, or the raw array with
    `response_format="json"`.

    Examples:
        params = {"symbol": "BTCUSDT"}
        params = {}

    Error Handling:
    -2015 means the key lacks permission or this IP is not allowlisted. An empty list is
    a valid answer: nothing is resting.
    """
    try:
        query: dict[str, Any] = {}
        if params.symbol is not None:
            query["symbol"] = params.symbol
        client = get_client()
        resp = await client.request("GET", "/api/v3/openOrders", auth="signed", params=query)
        payload = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(payload))
        orders = payload if isinstance(payload, list) else []
        title = f"Open orders on {params.symbol}" if params.symbol else "Open orders (all symbols)"
        return clip_response("\n".join(_order_table(orders, title)))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_all_orders",
    annotations=ToolAnnotations(
        title="Binance All Orders",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_all_orders(params: GetAllOrdersInput) -> str:
    """List a symbol's orders — open, filled, cancelled and expired alike.

    Calls `GET /api/v3/allOrders` (SIGNED, IP weight 20). Two ways to narrow it:
    `order_id` as a cursor (returns orders with orderId >= it) or a `start_time`/
    `end_time` window. **The window may not exceed 24 hours** — that cap is checked here,
    before the call, so you get a clear message instead of Binance's -1127. Walk a longer
    span in 24 h slices, or page by `order_id`.

    When to Use:
    - Reconstructing what happened on a symbol in a given day.
    - Paging order history forward with an `order_id` cursor.

    When NOT to Use:
    - For what is open right now — `binance_get_open_orders` (weight 6 with a symbol).
    - For the actual fills, fees and trade ids — `binance_get_my_trades`
      (trade_history.py); an order row only carries aggregates.

    Pagination:
    `limit` is 1-1000 (Binance default 500) and at most 50 rows are rendered; use
    `response_format="json"` or narrow the window for the rest. `order_id` pages forward:
    pass the last orderId you saw, plus one.

    Windows:
    `start_time`/`end_time` accept epoch ms or ISO-8601 and must span 24 h or less
    together. Omit both to get the most recent `limit` orders. Orders with no fill are
    archived after 90 days and stop being returned.

    Examples:
        params = {"symbol": "BTCUSDT"}
        params = {"symbol": "BTCUSDT", "start_time": "2026-09-22T00:00:00Z",
                  "end_time": "2026-09-22T23:59:59Z", "limit": 1000}
        params = {"symbol": "BTCUSDT", "order_id": 123456789}

    Error Handling:
    A window wider than 24 h is rejected locally. -1127 from Binance means the same thing
    reached it anyway; -1121 is an unknown symbol. -2015 means the key lacks permission or
    this IP is not allowlisted.
    """
    try:
        start_ms = _to_ms(params.start_time, "start_time")
        end_ms = _to_ms(params.end_time, "end_time")
        if start_ms is not None and end_ms is not None:
            if end_ms <= start_ms:
                return "Error: end_time must be after start_time."
            if end_ms - start_ms > _ORDER_WINDOW_MS:
                hours = (end_ms - start_ms) / (60 * 60 * 1000)
                return (
                    f"Error: the start_time/end_time window spans ~{hours:.1f} hours; Binance's allOrders "
                    "endpoint caps it at 24 hours. Slice the range into 24 h windows, or page with order_id."
                )
        query: dict[str, Any] = {"symbol": params.symbol, "limit": params.limit}
        if params.order_id is not None:
            query["orderId"] = params.order_id
        if start_ms is not None:
            query["startTime"] = start_ms
        if end_ms is not None:
            query["endTime"] = end_ms
        client = get_client()
        resp = await client.request("GET", "/api/v3/allOrders", auth="signed", params=query)
        payload = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(payload))
        orders = payload if isinstance(payload, list) else []
        return clip_response("\n".join(_order_table(orders, f"Orders on {params.symbol}")))
    except Exception as exc:
        return handle_api_error(exc)
