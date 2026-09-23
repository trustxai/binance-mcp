"""Spot account tools: balances, commission rates, order rate limits, prevented
matches (self-trade-prevention rejections) and allocations (SOR fills).

Scope per `.memory/research/02-endpoint-inventory.md` §B: five SIGNED, read-only,
USER_DATA endpoints under `/api/v3`.

Design note on `binance_get_spot_account`'s `omit_zero_balances`: Binance's own
`omitZeroBalances` query param drops zero-balance rows *server-side*, which means
a caller can never learn how many were dropped. This tool instead always fetches
the full `balances` array and applies the same default-True filtering itself,
purely client-side — that is what lets it report a "zero-balance assets hidden"
count. `omit_zero_balances=True` is otherwise behaviourally identical to passing
Binance's flag.

`_to_ms` (shared by `binance_get_allocations`) accepts either an `int` already in
epoch milliseconds, a numeric string of at least 12 digits (also epoch ms — a
shorter numeric string is ambiguous with something else and rejected), or an
ISO-8601 string; anything else returns a readable `Error: ...` string instead of
raising, so the caller can return it directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from binance_mcp.client import get_client
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import ResponseFormat, clip_response, epoch_to_human, fmt_num, to_json
from binance_mcp.server import mcp

# Trading-pair symbols (BTCUSDT); bare asset codes use a separate, looser pattern.
_SYMBOL_PATTERN = r"^[A-Z0-9]{2,20}$"
# One-letter assets exist on Binance (e.g. W, S) — the {2,20} bound belongs to
# trading-pair symbols, not bare assets.
_ASSET_PATTERN = r"^[A-Z0-9]{1,20}$"

# Context-window guard on top of the API `limit`: never render more rows than this,
# even when the API returned (or we fetched) more.
MAX_DISPLAY_ROWS = 50

# GET /api/v3/myAllocations enforces startTime..endTime <= 24 h (S2 L4769).
_ALLOCATIONS_WINDOW_MS = 24 * 60 * 60 * 1000


def _to_ms(value: int | str | None) -> int | str | None:
    """Convert a tool-surface timestamp to epoch milliseconds for Binance.

    Accepts `None` (returned unchanged), an `int` (already epoch ms), a numeric
    string of at least 12 digits (also epoch ms), or an ISO-8601 string. Returns a
    readable `Error: ...` string — instead of raising — when the value cannot be
    parsed either way, so the caller can propagate it directly as the tool result.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = value.strip()
    if text.isdigit() and len(text) >= 12:
        return int(text)
    try:
        iso = f"{text[:-1]}+00:00" if text.endswith("Z") else text
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return (
            f"Error: could not parse timestamp {value!r} — pass epoch milliseconds (an int, or a "
            "numeric string of at least 12 digits) or an ISO-8601 string, e.g. 2024-01-01T00:00:00Z."
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _is_zero_balance(row: dict[str, Any]) -> bool:
    try:
        free = Decimal(str(row.get("free", "0") or "0"))
        locked = Decimal(str(row.get("locked", "0") or "0"))
    except InvalidOperation:
        return False
    return free == 0 and locked == 0


def _format_balance(row: dict[str, Any]) -> str:
    return f"- **{row.get('asset', '?')}**: free {fmt_num(row.get('free'))}, locked {fmt_num(row.get('locked'))}"


def _format_commission_block(label: str, block: dict[str, Any] | None) -> str:
    if not block:
        return f"- **{label} commission**: N/A"
    return (
        f"- **{label} commission**: maker {fmt_num(block.get('maker'))}, taker {fmt_num(block.get('taker'))}, "
        f"buyer {fmt_num(block.get('buyer'))}, seller {fmt_num(block.get('seller'))}"
    )


def _format_rate_limit(row: dict[str, Any]) -> str:
    return (
        f"- **{row.get('rateLimitType', '?')}** per {row.get('intervalNum', '?')} {row.get('interval', '?')}: "
        f"{row.get('count', '?')} / {row.get('limit', '?')} used"
    )


def _format_prevented_match(row: dict[str, Any]) -> str:
    return (
        f"- match `{row.get('preventedMatchId', '?')}` (trade group `{row.get('tradeGroupId', '?')}`) — "
        f"taker order `{row.get('takerOrderId', '?')}` vs maker order `{row.get('makerOrderId', '?')}` "
        f"({row.get('makerSymbol', '?')}), price {fmt_num(row.get('price'))}, "
        f"maker qty prevented {fmt_num(row.get('makerPreventedQuantity'))}, "
        f"STP mode {row.get('selfTradePreventionMode', '?')}, at {epoch_to_human(row.get('transactTime'))}"
    )


def _format_allocation(row: dict[str, Any]) -> str:
    side = "buy" if row.get("isBuyer") else "sell"
    return (
        f"- alloc `{row.get('allocationId', '?')}` — order `{row.get('orderId', '?')}` "
        f"(list `{row.get('orderListId', '-1')}`) {row.get('symbol', '?')}: {side} "
        f"qty {fmt_num(row.get('qty'))} @ {fmt_num(row.get('price'))} = {fmt_num(row.get('quoteQty'))}, "
        f"commission {fmt_num(row.get('commission'))} {row.get('commissionAsset', '')}, "
        f"maker {bool(row.get('isMaker'))}, allocator {bool(row.get('isAllocator'))}, "
        f"at {epoch_to_human(row.get('time'))}"
    )


class _SpotAccountInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    omit_zero_balances: bool = Field(
        default=True,
        description=(
            "Hide zero-balance assets from the rendered list (applied client-side; see the module "
            "docstring for why). Default True, matching Binance's own default-off-but-recommended usage."
        ),
    )
    asset: str | None = Field(
        default=None,
        min_length=1,
        pattern=_ASSET_PATTERN,
        description=(
            "Show only this asset's balance, e.g. USDT (returned even if zero, overriding "
            "omit_zero_balances). Case-insensitive; normalized to uppercase."
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: markdown (default) or json."
    )

    @field_validator("asset", mode="before")
    @classmethod
    def _upper_asset(cls, value: str | None) -> str | None:
        return value.upper() if isinstance(value, str) else value


class _CommissionRatesInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str = Field(..., pattern=_SYMBOL_PATTERN, description="Trading pair symbol, e.g. BTCUSDT.")
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: markdown (default) or json."
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def _upper_symbol(cls, value: str) -> str:
        return value.upper() if isinstance(value, str) else value


class _OrderRateLimitsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: markdown (default) or json."
    )


class _PreventedMatchesInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str = Field(..., pattern=_SYMBOL_PATTERN, description="Trading pair symbol, e.g. BTCUSDT.")
    prevented_match_id: int | None = Field(
        default=None,
        ge=1,
        description="Look up a specific preventedMatchId. Mutually exclusive with order_id.",
    )
    order_id: int | None = Field(
        default=None,
        ge=1,
        description="Look up every prevented match for this orderId. Mutually exclusive with prevented_match_id.",
    )
    from_prevented_match_id: int | None = Field(
        default=None,
        ge=1,
        description="Page forward from this preventedMatchId (inclusive). Only valid together with order_id.",
    )
    limit: int = Field(
        default=500,
        ge=1,
        le=1000,
        description="Max rows when querying by order_id (Binance default 500, max 1000). Ignored otherwise.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: markdown (default) or json."
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def _upper_symbol(cls, value: str) -> str:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_combo(self) -> _PreventedMatchesInput:
        if self.prevented_match_id is None and self.order_id is None:
            raise ValueError("Provide exactly one of prevented_match_id or order_id.")
        if self.prevented_match_id is not None and self.order_id is not None:
            raise ValueError("prevented_match_id and order_id are mutually exclusive — provide exactly one.")
        if self.from_prevented_match_id is not None and self.order_id is None:
            raise ValueError("from_prevented_match_id is only valid together with order_id.")
        return self


class _AllocationsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str = Field(..., pattern=_SYMBOL_PATTERN, description="Trading pair symbol, e.g. BTCUSDT.")
    start_time: int | str | None = Field(
        default=None,
        description="Window start — epoch ms or ISO-8601. Combine with end_time; the span must be <= 24 h.",
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end — epoch ms or ISO-8601. Combine with start_time; the span must be <= 24 h.",
    )
    from_allocation_id: int | None = Field(
        default=None, ge=0, description="Resume from this allocationId (inclusive) — id-cursor pagination."
    )
    order_id: int | None = Field(default=None, ge=1, description="Filter to allocations for this orderId.")
    limit: int = Field(default=500, ge=1, le=1000, description="Max rows (Binance default 500, max 1000).")
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: markdown (default) or json."
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def _upper_symbol(cls, value: str) -> str:
        return value.upper() if isinstance(value, str) else value


@mcp.tool(
    name="binance_get_spot_account",
    annotations=ToolAnnotations(
        title="Binance Spot Account",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_spot_account(params: _SpotAccountInput) -> str:
    """Get spot account state: trade/withdraw/deposit flags, commission rates, balances.

    Calls `GET /api/v3/account` (SIGNED, USER_DATA). IP weight 20 per call.

    When to Use:
    - To check current spot balances (free + locked) for one or all assets.
    - To confirm whether the account can currently trade, withdraw, or deposit,
      and what its maker/taker/buyer/seller commission rates are.

    When NOT to Use:
    - For a specific symbol's commission (with any special/discount overrides) —
      use `binance_get_commission_rates`.
    - For non-spot wallets (Funding, Earn, ...) — use the wallet-account tools.

    Returns:
    A markdown block (or JSON with `response_format="json"`) with canTrade/
    canWithdraw/canDeposit, account type, commissionRates (maker/taker/buyer/
    seller via fmt_num), permissions, last update time, and the balances list
    (free/locked via fmt_num). When `omit_zero_balances` is True (the default)
    and no `asset` filter is given, a count of hidden zero-balance assets is
    shown. Display is capped at `MAX_DISPLAY_ROWS` (50) with a truncation note;
    JSON mode instead adds `truncated`/`balancesShown`/`balancesMatched` fields
    so truncation is machine-readable too.

    Examples:
    params = {}
    params = {"omit_zero_balances": False}
    params = {"asset": "USDT"}

    Error Handling:
    -2015 means the key lacks Reading permission or this machine's IP is not on
    the key's allowlist. An empty balances list after filtering means the asset
    filter matched nothing, or every balance was zero and got hidden.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/api/v3/account", auth="signed")
        data = resp.json()
        balances: list[dict[str, Any]] = data.get("balances") or []
        hidden_zero = 0
        if params.asset:
            balances = [b for b in balances if str(b.get("asset", "")).upper() == params.asset]
        elif params.omit_zero_balances:
            kept = []
            for row in balances:
                if _is_zero_balance(row):
                    hidden_zero += 1
                else:
                    kept.append(row)
            balances = kept
        balances_matched = len(balances)
        truncated = balances_matched > MAX_DISPLAY_ROWS
        truncated_note = None
        if truncated:
            truncated_note = (
                f"_[display truncated: {balances_matched} balances matched, showing the first "
                f"{MAX_DISPLAY_ROWS} — narrow with `asset` to see a specific one]_"
            )
            balances = balances[:MAX_DISPLAY_ROWS]
        if params.response_format is ResponseFormat.JSON:
            return clip_response(
                to_json(
                    {
                        **data,
                        "balances": balances,
                        "hiddenZeroBalances": hidden_zero,
                        "truncated": truncated,
                        "balancesShown": len(balances),
                        "balancesMatched": balances_matched,
                    }
                )
            )
        commission = data.get("commissionRates") or {}
        hidden_note = f", {hidden_zero} zero-balance asset(s) hidden" if hidden_zero else ""
        lines = [
            "# Binance Spot Account",
            "",
            f"- **can trade / withdraw / deposit**: {bool(data.get('canTrade'))} / "
            f"{bool(data.get('canWithdraw'))} / {bool(data.get('canDeposit'))}",
            f"- **account type**: {data.get('accountType', '?')}",
            f"- **commission rates** — maker {fmt_num(commission.get('maker'))}, "
            f"taker {fmt_num(commission.get('taker'))}, buyer {fmt_num(commission.get('buyer'))}, "
            f"seller {fmt_num(commission.get('seller'))}",
            f"- **permissions**: {', '.join(data.get('permissions') or []) or '?'}",
            f"- **updated**: {epoch_to_human(data.get('updateTime'))}",
            "",
            f"## Balances — {len(balances)} shown{hidden_note}",
            "",
        ]
        if balances:
            lines.extend(_format_balance(row) for row in balances)
        else:
            lines.append("_No balances match._")
        if truncated_note:
            lines.append("")
            lines.append(truncated_note)
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_commission_rates",
    annotations=ToolAnnotations(
        title="Binance Commission Rates",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_commission_rates(params: _CommissionRatesInput) -> str:
    """Get the account's standard/special/tax commission rates for one symbol.

    Calls `GET /api/v3/account/commission` (SIGNED, USER_DATA). IP weight 20 per call.

    When to Use:
    - To see the exact maker/taker/buyer/seller commission that will apply on a
      given symbol, including any special-tier override, tax commission, or a
      BNB-style fee discount.

    When NOT to Use:
    - For the account's default (non-symbol-specific) commission rates — use
      `binance_get_spot_account`, which echoes `commissionRates` too.

    Returns:
    A markdown block (or JSON) with the symbol, standard/special/tax commission
    blocks (maker/taker/buyer/seller via fmt_num), and the discount block
    (enabled-for-account, enabled-for-symbol, discount asset, rate).

    Examples:
    params = {"symbol": "BTCUSDT"}

    Error Handling:
    -1121 means an invalid or unknown symbol — check `binance_get_exchange_info`.
    -2015 means the key lacks Reading permission or the IP is not allowlisted.
    """
    try:
        client = get_client()
        query = {"symbol": params.symbol}
        resp = await client.request("GET", "/api/v3/account/commission", params=query, auth="signed")
        data = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        discount = data.get("discount") or {}
        lines = [
            f"# Commission Rates — {data.get('symbol', params.symbol)}",
            "",
            _format_commission_block("standard", data.get("standardCommission")),
            _format_commission_block("special", data.get("specialCommission")),
            _format_commission_block("tax", data.get("taxCommission")),
            f"- **discount**: account {bool(discount.get('enabledForAccount'))}, "
            f"symbol {bool(discount.get('enabledForSymbol'))}, "
            f"asset {discount.get('discountAsset', 'N/A')}, rate {fmt_num(discount.get('discount'))}",
        ]
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_order_rate_limits",
    annotations=ToolAnnotations(
        title="Binance Order Rate Limits",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_order_rate_limits(params: _OrderRateLimitsInput) -> str:
    """Get the account's current order-rate-limit usage (per-second/day order counts).

    Calls `GET /api/v3/rateLimit/order` (SIGNED, USER_DATA). **IP weight 40** per call
    — noticeably heavier than the other tools in this module; do not poll this tightly.

    When to Use:
    - Before a burst of order placements, to see how much of the ORDERS rate
      limit (per interval, e.g. 10s/1d) has already been used.
    - To debug a -1015 "Too many orders" rejection.

    When NOT to Use:
    - For the exchange-wide REQUEST_WEIGHT/RAW_REQUESTS limits — those come back
      in every response's rate-limit headers, not from this endpoint.

    Returns:
    A markdown list (or JSON) of `{rateLimitType, interval, intervalNum, limit,
    count}` entries — one per configured ORDERS rate-limit window.

    Examples:
    params = {}

    Error Handling:
    -2015 means the key lacks Reading permission or the IP is not allowlisted.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/api/v3/rateLimit/order", auth="signed")
        data = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        lines = ["# Order Rate Limits", ""]
        if data:
            lines.extend(_format_rate_limit(row) for row in data)
        else:
            lines.append("_No rate limit data returned._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_prevented_matches",
    annotations=ToolAnnotations(
        title="Binance Prevented Matches",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_prevented_matches(params: _PreventedMatchesInput) -> str:
    """List orders rejected by Self-Trade Prevention (STP) for a symbol.

    Calls `GET /api/v3/myPreventedMatches` (SIGNED, USER_DATA). IP weight 2 when
    queried by `prevented_match_id`, 20 when queried by `order_id`.

    When to Use:
    - To see which of your own orders were prevented from matching against each
      other (STP), including the price and quantity that was blocked.
    - To audit STP behavior for a specific order via `order_id`.

    When NOT to Use:
    - For orders that DID execute — use `binance_get_my_trades` (trade history).

    Returns:
    A markdown list (or JSON) of prevented matches: preventedMatchId,
    tradeGroupId, taker/maker order ids, maker symbol, price, maker quantity
    prevented (via fmt_num), the self-trade-prevention mode, and the
    transaction time. Display is capped at `MAX_DISPLAY_ROWS` (50); JSON mode
    returns `{count, truncated, displayLimit, items}` rather than a bare array,
    so truncation stays valid JSON.

    Pagination:
    Only valid together with `order_id`: `from_prevented_match_id` is an
    inclusive cursor — pass the last-seen `preventedMatchId` (or one past it)
    to page forward, and `limit` (only sent when `from_prevented_match_id` is
    set; Binance default 500, max 1000) caps how many rows come back per call.
    Display is additionally capped at `MAX_DISPLAY_ROWS` (50) regardless of
    `limit`.

    Examples:
    params = {"symbol": "BTCUSDT", "prevented_match_id": 1}
    params = {"symbol": "BTCUSDT", "order_id": 12345, "from_prevented_match_id": 5}

    Error Handling:
    Exactly one of `prevented_match_id` or `order_id` is required — validated
    locally before the call. `from_prevented_match_id` requires `order_id`.
    -1121 means an invalid symbol; -2013/-2011 mean the order id does not exist.
    """
    try:
        client = get_client()
        raw_query: dict[str, Any] = {
            "symbol": params.symbol,
            "preventedMatchId": params.prevented_match_id,
            "orderId": params.order_id,
            "fromPreventedMatchId": params.from_prevented_match_id,
            "limit": params.limit if params.from_prevented_match_id is not None else None,
        }
        query = {key: value for key, value in raw_query.items() if value is not None}
        resp = await client.request("GET", "/api/v3/myPreventedMatches", params=query, auth="signed")
        rows: list[dict[str, Any]] = resp.json()
        total_count = len(rows)
        truncated = total_count > MAX_DISPLAY_ROWS
        truncated_note = None
        if truncated:
            truncated_note = (
                f"_[display truncated: {total_count} rows returned, showing the first {MAX_DISPLAY_ROWS} — "
                "narrow with `order_id`/`from_prevented_match_id` to see the rest]_"
            )
            rows = rows[:MAX_DISPLAY_ROWS]
        if params.response_format is ResponseFormat.JSON:
            return clip_response(
                to_json({"count": total_count, "truncated": truncated, "displayLimit": MAX_DISPLAY_ROWS, "items": rows})
            )
        lines = [f"# Prevented Matches — {params.symbol}", ""]
        if rows:
            lines.extend(_format_prevented_match(row) for row in rows)
        else:
            lines.append("_No prevented matches found._")
        if truncated_note:
            lines.append("")
            lines.append(truncated_note)
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_allocations",
    annotations=ToolAnnotations(
        title="Binance Allocations (SOR Fills)",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_allocations(params: _AllocationsInput) -> str:
    """List Smart Order Routing (SOR) allocations — the per-symbol fills that make
    up an SOR order's execution.

    Calls `GET /api/v3/myAllocations` (SIGNED, USER_DATA). IP weight 20 per call.

    When to Use:
    - To see how an SOR order (`binance_place_sor_order`) was actually filled
      across symbols, with per-allocation price/qty/commission.
    - To page through allocations for a specific order via `order_id`.

    When NOT to Use:
    - For regular (non-SOR) trade fills — use `binance_get_my_trades`.

    Returns:
    A markdown list (or JSON) of allocations: allocationId, orderId/orderListId,
    symbol, side (isBuyer), qty/price/quoteQty (via fmt_num), commission +
    commissionAsset, maker/allocator flags, and time. Display is capped at
    `MAX_DISPLAY_ROWS` (50); JSON mode returns `{count, truncated, displayLimit,
    items}` rather than a bare array, so truncation stays valid JSON.

    Windows:
    `start_time`/`end_time` (epoch ms or ISO-8601) accept a span of at most 24 h
    — enforced locally with a readable `Error:` before the call reaches Binance,
    since Binance would otherwise answer -1127.

    Examples:
    params = {"symbol": "BTCUSDT"}
    params = {"symbol": "BTCUSDT", "start_time": "2024-01-01T00:00:00Z", "end_time": "2024-01-01T12:00:00Z"}
    params = {"symbol": "BTCUSDT", "order_id": 12345}

    Error Handling:
    -1127 means the requested window is too wide (should not happen — the 24 h
    cap is enforced before the request). -1121 means an invalid symbol.
    """
    try:
        client = get_client()
        start_ms = _to_ms(params.start_time)
        if isinstance(start_ms, str):
            return start_ms
        end_ms = _to_ms(params.end_time)
        if isinstance(end_ms, str):
            return end_ms
        if start_ms is not None and end_ms is not None:
            if end_ms < start_ms:
                return "Error: end_time is before start_time."
            if end_ms - start_ms > _ALLOCATIONS_WINDOW_MS:
                return (
                    "Error: startTime..endTime spans more than 24 h, which Binance's myAllocations "
                    "endpoint does not allow — narrow the window."
                )
        raw_query: dict[str, Any] = {
            "symbol": params.symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "fromAllocationId": params.from_allocation_id,
            "orderId": params.order_id,
            "limit": params.limit,
        }
        query = {key: value for key, value in raw_query.items() if value is not None}
        resp = await client.request("GET", "/api/v3/myAllocations", params=query, auth="signed")
        rows: list[dict[str, Any]] = resp.json()
        total_count = len(rows)
        truncated = total_count > MAX_DISPLAY_ROWS
        truncated_note = None
        if truncated:
            truncated_note = (
                f"_[display truncated: {total_count} rows returned, showing the first {MAX_DISPLAY_ROWS} — "
                "narrow the window or lower `limit` to see the rest]_"
            )
            rows = rows[:MAX_DISPLAY_ROWS]
        if params.response_format is ResponseFormat.JSON:
            return clip_response(
                to_json({"count": total_count, "truncated": truncated, "displayLimit": MAX_DISPLAY_ROWS, "items": rows})
            )
        lines = [f"# Allocations — {params.symbol}", ""]
        if rows:
            lines.extend(_format_allocation(row) for row in rows)
        else:
            lines.append("_No allocations found._")
        if truncated_note:
            lines.append("")
            lines.append(truncated_note)
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)
