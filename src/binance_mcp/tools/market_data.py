"""Public Binance Spot market data — `/api/v3`, no API key required (`auth="none"`).

Every tool here reads market data only: order book, trades, klines, tickers, and the
exchange's own symbol/filter metadata. None of them touch the account or place orders.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from re import compile as re_compile
from typing import Any

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from binance_mcp.client import get_client
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import ResponseFormat, clip_response, epoch_to_human, fmt_num, to_json
from binance_mcp.server import mcp

# -- context-window guards ----------------------------------------------------
# Applied on top of the API's own `limit`, per module convention (50 for symbol/ticker
# lists that can carry many fields each, 100 for flatter rows like trades/klines/prices).
MAX_EXCHANGE_INFO_SYMBOLS = 50
MAX_DEPTH_LEVELS = 50
MAX_TRADES_DISPLAY = 100
MAX_KLINES_DISPLAY = 100
MAX_TICKER_DISPLAY = 50
MAX_PRICE_DISPLAY = 100

_SYMBOL_PATTERN = re_compile(r"^[A-Z0-9]{2,20}$")
_RELEVANT_FILTERS = {"LOT_SIZE", "PRICE_FILTER", "NOTIONAL", "MARKET_LOT_SIZE"}


# -- shared helpers -------------------------------------------------------------


def _normalize_symbol(value: str) -> str:
    """Uppercase and validate a Binance symbol (e.g. `btcusdt` -> `BTCUSDT`)."""
    upper = value.strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(upper):
        raise ValueError(f"Invalid symbol {value!r} — expected 2-20 uppercase letters/digits (e.g. BTCUSDT).")
    return upper


def _to_ms(value: Any) -> int | None:
    """Accept an epoch-ms int or an ISO-8601 string on the tool surface; always send ms."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"Invalid timestamp {value!r} — use epoch milliseconds or ISO-8601 (e.g. 2024-01-01T00:00:00Z)."
            ) from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    raise ValueError(f"Invalid timestamp {value!r} — expected an int (epoch ms) or an ISO-8601 string.")


class SymbolStatus(StrEnum):
    """`GET /api/v3/exchangeInfo` symbol lifecycle states."""

    TRADING = "TRADING"
    HALT = "HALT"
    BREAK = "BREAK"


class TickerType(StrEnum):
    """`type` selector on the ticker family of endpoints."""

    FULL = "FULL"
    MINI = "MINI"


def _format_ticker_stats(t: dict[str, Any]) -> str:
    """Shared renderer for `/ticker/24hr`, `/ticker/tradingDay`, and `/ticker` (rolling).

    All three share the same FULL/MINI field shape (price change stats + open/high/low/
    close/volume + a trade count); MINI omits weightedAvgPrice/bid/ask/counts.
    """
    lines = [
        f"## {t.get('symbol')}",
        f"- last: {fmt_num(t.get('lastPrice'))}; change: {fmt_num(t.get('priceChange'))} "
        f"({fmt_num(t.get('priceChangePercent'))}%)",
        f"- open: {fmt_num(t.get('openPrice'))}; high: {fmt_num(t.get('highPrice'))}; "
        f"low: {fmt_num(t.get('lowPrice'))}; volume: {fmt_num(t.get('volume'))}",
    ]
    if "weightedAvgPrice" in t:
        lines.append(
            f"- weighted avg: {fmt_num(t.get('weightedAvgPrice'))}; "
            f"bid: {fmt_num(t.get('bidPrice'))}; ask: {fmt_num(t.get('askPrice'))}"
        )
        lines.append(
            f"- window: {epoch_to_human(t.get('openTime'))} → {epoch_to_human(t.get('closeTime'))}; "
            f"trades: {t.get('count')}"
        )
    return "\n".join(lines)


def _format_exchange_symbol(sym: dict[str, Any]) -> str:
    """Render one `exchangeInfo` symbol: status, assets, order types, and the filters
    an order must respect (LOT_SIZE / PRICE_FILTER / NOTIONAL / MARKET_LOT_SIZE)."""
    filters = {f["filterType"]: f for f in sym.get("filters", []) if f.get("filterType") in _RELEVANT_FILTERS}
    lines = [
        f"## {sym.get('symbol')}",
        f"- status: {sym.get('status')}; base: {sym.get('baseAsset')}; quote: {sym.get('quoteAsset')}",
        f"- order types: {', '.join(sym.get('orderTypes', []))}",
    ]
    if (f := filters.get("LOT_SIZE")) is not None:
        lines.append(
            f"- LOT_SIZE: min {fmt_num(f.get('minQty'))}, max {fmt_num(f.get('maxQty'))}, "
            f"step {fmt_num(f.get('stepSize'))}"
        )
    if (f := filters.get("MARKET_LOT_SIZE")) is not None:
        lines.append(
            f"- MARKET_LOT_SIZE: min {fmt_num(f.get('minQty'))}, max {fmt_num(f.get('maxQty'))}, "
            f"step {fmt_num(f.get('stepSize'))}"
        )
    if (f := filters.get("PRICE_FILTER")) is not None:
        lines.append(
            f"- PRICE_FILTER: min {fmt_num(f.get('minPrice'))}, max {fmt_num(f.get('maxPrice'))}, "
            f"tick {fmt_num(f.get('tickSize'))}"
        )
    if (f := filters.get("NOTIONAL")) is not None:
        lines.append(f"- NOTIONAL: min {fmt_num(f.get('minNotional'))}, applyToMarket {f.get('applyToMarket')}")
    return "\n".join(lines)


# -- binance_get_exchange_info ---------------------------------------------------


class ExchangeInfoInput(BaseModel):
    """Input for `binance_get_exchange_info`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str | None = Field(
        default=None,
        description="Single symbol to look up, e.g. BTCUSDT. Mutually exclusive with `symbols`, "
        "`permissions`, and `symbol_status`.",
    )
    symbols: list[str] | None = Field(
        default=None,
        description="Multiple symbols to look up. Mutually exclusive with `symbol`, `permissions`, "
        "and `symbol_status`.",
    )
    permissions: list[str] | None = Field(
        default=None,
        description="Filter by trading permission set (e.g. SPOT, MARGIN). Cannot combine with `symbol`/`symbols`.",
    )
    symbol_status: SymbolStatus | None = Field(
        default=None,
        description="Filter by symbol status (TRADING/HALT/BREAK). Cannot combine with `symbol`/`symbols`.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str | None) -> str | None:
        return _normalize_symbol(v) if v is not None else v

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, v: list[str] | None) -> list[str] | None:
        return [_normalize_symbol(s) for s in v] if v is not None else v

    @model_validator(mode="after")
    def _check_combo(self) -> ExchangeInfoInput:
        if self.symbol is not None and self.symbols is not None:
            raise ValueError("Pass either `symbol` or `symbols`, not both.")
        narrowing = self.symbol is not None or self.symbols is not None
        filtering = self.permissions is not None or self.symbol_status is not None
        if narrowing and filtering:
            raise ValueError("`symbol`/`symbols` cannot combine with `permissions`/`symbol_status`.")
        return self


@mcp.tool(
    name="binance_get_exchange_info",
    annotations=ToolAnnotations(
        title="Binance Exchange Info",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_exchange_info(params: ExchangeInfoInput) -> str:
    """Look up trading rules, symbol status, and order filters for spot symbols.

    Calls `GET /api/v3/exchangeInfo` (weight 20). Without `symbol`/`symbols`/`permissions`/
    `symbol_status` this returns Binance's full symbol universe (3707+ symbols); the
    response is always capped at 50 symbols here — pass `symbol`/`symbols` to narrow it.

    When to Use:
    - Before placing an order, to read the LOT_SIZE/PRICE_FILTER/NOTIONAL/MARKET_LOT_SIZE
      filter values a quantity/price must respect (see `binance_place_order`).
    - To check whether a symbol is currently TRADING, HALTed, or in BREAK.
    - To discover which symbols share a base/quote asset (with `permissions`/`symbol_status`).

    When NOT to Use:
    - For live prices — use `binance_get_ticker_price` or `binance_get_avg_price`.
    - For account-specific trading permissions — use `binance_get_spot_account`.

    Returns:
    Markdown: one block per symbol (status, base/quote, order types, filter values), capped
    at 50 symbols with a note if more matched. JSON: the same data, `count` vs `shown`.

    Examples:
    params = {"symbol": "BTCUSDT"}
    params = {"symbols": ["BTCUSDT", "ETHUSDT"]}
    params = {"permissions": ["SPOT"], "symbol_status": "TRADING"}

    Error Handling:
    An unknown symbol raises Binance `-1121 Invalid symbol`; combining `symbol`/`symbols`
    with `permissions`/`symbol_status` is rejected locally before the call is made.
    """
    try:
        client = get_client()
        query: dict[str, Any] = {}
        if params.symbol is not None:
            query["symbol"] = params.symbol
        if params.symbols is not None:
            query["symbols"] = params.symbols
        if params.permissions is not None:
            query["permissions"] = params.permissions
        if params.symbol_status is not None:
            query["symbolStatus"] = params.symbol_status.value
        resp = await client.request("GET", "/api/v3/exchangeInfo", params=query, auth="none")
        data = resp.json()
        symbols: list[dict[str, Any]] = data.get("symbols", [])
        shown = symbols[:MAX_EXCHANGE_INFO_SYMBOLS]

        if params.response_format is ResponseFormat.JSON:
            payload: dict[str, Any] = {
                "timezone": data.get("timezone"),
                "serverTime": data.get("serverTime"),
                "count": len(symbols),
                "shown": len(shown),
                "symbols": shown,
            }
            if len(symbols) > MAX_EXCHANGE_INFO_SYMBOLS:
                payload["note"] = (
                    f"{len(symbols)} symbols total; showing first {MAX_EXCHANGE_INFO_SYMBOLS} — "
                    "pass `symbol`/`symbols` to narrow."
                )
            return clip_response(to_json(payload))

        lines = [
            "# Binance exchange info",
            "",
            f"Timezone: {data.get('timezone')}; server time: {epoch_to_human(data.get('serverTime'))}.",
            f"Total symbols matched: {len(symbols):,}.",
            "",
        ]
        if not symbols:
            lines.append("_No symbols matched._")
        for sym in shown:
            lines.append(_format_exchange_symbol(sym))
        if len(symbols) > MAX_EXCHANGE_INFO_SYMBOLS:
            lines.append(
                f"_...{len(symbols) - MAX_EXCHANGE_INFO_SYMBOLS} more symbols not shown — "
                "pass `symbol`/`symbols` to narrow._"
            )
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


# -- binance_get_order_book -------------------------------------------------------


class OrderBookInput(BaseModel):
    """Input for `binance_get_order_book`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str = Field(description="Trading pair symbol, e.g. BTCUSDT.")
    limit: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Order book depth per side. Capped at 500 here (Binance's own max is 5000) "
        "to keep responses small.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        return _normalize_symbol(v)


@mcp.tool(
    name="binance_get_order_book",
    annotations=ToolAnnotations(
        title="Binance Order Book",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_order_book(params: OrderBookInput) -> str:
    """Fetch the current order book (bids/asks) for a symbol.

    Calls `GET /api/v3/depth`. Weight tiers by `limit`: 1-100 → 5, 101-500 → 25 (this
    tool's cap is 500; Binance itself allows up to 5000 at weight 250, unavailable here).

    When to Use:
    - To see live liquidity/spread before sizing an order.
    - To validate a limit price against the current best bid/ask.

    When NOT to Use:
    - For the last traded price only — use `binance_get_book_ticker` (cheaper, weight 2/4).
    - For historical trades — use `binance_get_recent_trades` / `binance_get_agg_trades`.

    Returns:
    Markdown: top 50 levels per side as price/qty tables, with the book's `lastUpdateId`.
    JSON: the full requested depth (up to `limit`), uncapped.

    Examples:
    params = {"symbol": "BTCUSDT", "limit": 20}

    Error Handling:
    An unknown symbol raises Binance `-1121 Invalid symbol`.
    """
    try:
        client = get_client()
        resp = await client.request(
            "GET", "/api/v3/depth", params={"symbol": params.symbol, "limit": params.limit}, auth="none"
        )
        data = resp.json()
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        if params.response_format is ResponseFormat.JSON:
            return clip_response(
                to_json(
                    {
                        "symbol": params.symbol,
                        "lastUpdateId": data.get("lastUpdateId"),
                        "bids": bids,
                        "asks": asks,
                    }
                )
            )

        shown_bids = bids[:MAX_DEPTH_LEVELS]
        shown_asks = asks[:MAX_DEPTH_LEVELS]
        lines = [
            f"# Order book — {params.symbol}",
            "",
            f"lastUpdateId: {data.get('lastUpdateId')}",
            "",
            f"**Bids** (top {len(shown_bids)} of {len(bids)}):",
            "| price | qty |",
            "|---|---|",
        ]
        for price, qty in shown_bids:
            lines.append(f"| {fmt_num(price)} | {fmt_num(qty)} |")
        lines.extend(
            [
                "",
                f"**Asks** (top {len(shown_asks)} of {len(asks)}):",
                "| price | qty |",
                "|---|---|",
            ]
        )
        for price, qty in shown_asks:
            lines.append(f"| {fmt_num(price)} | {fmt_num(qty)} |")
        if len(bids) > MAX_DEPTH_LEVELS or len(asks) > MAX_DEPTH_LEVELS:
            lines.append("")
            lines.append(f"_Showing top {MAX_DEPTH_LEVELS} levels/side — use response_format=json for the rest._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


# -- binance_get_recent_trades -----------------------------------------------------


class RecentTradesInput(BaseModel):
    """Input for `binance_get_recent_trades`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str = Field(description="Trading pair symbol, e.g. BTCUSDT.")
    limit: int = Field(default=500, ge=1, le=1000, description="Number of trades to fetch (Binance max 1000).")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        return _normalize_symbol(v)


@mcp.tool(
    name="binance_get_recent_trades",
    annotations=ToolAnnotations(
        title="Binance Recent Trades",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_recent_trades(params: RecentTradesInput) -> str:
    """Fetch the most recent public trades for a symbol.

    Calls `GET /api/v3/trades` (weight 25). Always returns the latest trades — there is
    no way to page backward here (use `binance_get_agg_trades` with `from_id` for that).

    When to Use:
    - To see the last executed prices/sizes and maker/taker mix for a symbol.

    When NOT to Use:
    - To page through trade history by id — use `binance_get_agg_trades`.
    - For your OWN trades — use `binance_get_my_trades` (signed).

    Returns:
    Markdown: a table of up to 100 trades (id, time, price, qty, side). JSON: the full
    requested page (up to `limit`), uncapped.

    Examples:
    params = {"symbol": "BTCUSDT", "limit": 50}

    Error Handling:
    An unknown symbol raises Binance `-1121 Invalid symbol`.
    """
    try:
        client = get_client()
        resp = await client.request(
            "GET", "/api/v3/trades", params={"symbol": params.symbol, "limit": params.limit}, auth="none"
        )
        trades: list[dict[str, Any]] = resp.json()

        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json({"symbol": params.symbol, "count": len(trades), "trades": trades}))

        shown = trades[:MAX_TRADES_DISPLAY]
        lines = [
            f"# Recent trades — {params.symbol}",
            "",
            f"Showing {len(shown)} of {len(trades)} trade(s).",
            "",
            "| id | time | price | qty | side |",
            "|---|---|---|---|---|",
        ]
        for t in shown:
            side = "maker-buy" if t.get("isBuyerMaker") else "taker-buy"
            lines.append(
                f"| {t.get('id')} | {epoch_to_human(t.get('time'))} | {fmt_num(t.get('price'))} | "
                f"{fmt_num(t.get('qty'))} | {side} |"
            )
        if not trades:
            lines.append("_No trades._")
        elif len(trades) > MAX_TRADES_DISPLAY:
            lines.append("")
            lines.append(f"_...{len(trades) - MAX_TRADES_DISPLAY} more not shown — use response_format=json._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


# -- binance_get_agg_trades --------------------------------------------------------


class AggTradesInput(BaseModel):
    """Input for `binance_get_agg_trades`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str = Field(description="Trading pair symbol, e.g. BTCUSDT.")
    from_id: int | None = Field(
        default=None, description="Aggregate trade id to fetch from (inclusive). Cannot combine with the times."
    )
    start_time: int | None = Field(
        default=None, description="Window start (inclusive), epoch ms or ISO-8601. Cannot combine with `from_id`."
    )
    end_time: int | None = Field(
        default=None, description="Window end (inclusive), epoch ms or ISO-8601. Cannot combine with `from_id`."
    )
    limit: int = Field(
        default=500, ge=1, le=1000, description="Number of aggregate trades to fetch (Binance max 1000)."
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        return _normalize_symbol(v)

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _validate_times(cls, v: Any) -> int | None:
        return _to_ms(v)

    @model_validator(mode="after")
    def _check_combo(self) -> AggTradesInput:
        if self.from_id is not None and (self.start_time is not None or self.end_time is not None):
            raise ValueError("`from_id` cannot combine with `start_time`/`end_time`.")
        return self


@mcp.tool(
    name="binance_get_agg_trades",
    annotations=ToolAnnotations(
        title="Binance Aggregate Trades",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_agg_trades(params: AggTradesInput) -> str:
    """Fetch compressed/aggregate trades (same price, same taker order, same timestamp merged).

    Calls `GET /api/v3/aggTrades` (weight 4). Filter with `from_id` for a stable cursor
    walk, or `start_time`/`end_time` for a time window — **Binance rejects a start/end
    window wider than 1 hour on this endpoint**; slice a longer range into ≤ 1h calls.

    When to Use:
    - To page through historical trades by id (`from_id`), which `binance_get_recent_trades`
      cannot do.
    - To reconstruct a short time window of trade flow cheaply (weight 4 vs 25).

    When NOT to Use:
    - For the very latest trades with no filter — `binance_get_recent_trades` is simpler.

    Returns:
    Markdown: a table of up to 100 aggregate trades (id, time, price, qty, first/last
    trade ids, side). JSON: the full requested page (up to `limit`), uncapped.

    Windows:
    `start_time`/`end_time` together must not span more than 1 hour (per Binance's own
    docs); omit both, or use `from_id`, for a wider walk.

    Examples:
    params = {"symbol": "BTCUSDT", "from_id": 123456}
    params = {"symbol": "BTCUSDT", "start_time": "2024-01-01T00:00:00Z", "end_time": "2024-01-01T00:45:00Z"}

    Error Handling:
    A window wider than 1 hour raises Binance `-1127 More than 1 hours between startTime
    and endTime`; combining `from_id` with the time window is rejected locally.
    """
    try:
        client = get_client()
        query: dict[str, Any] = {"symbol": params.symbol, "limit": params.limit}
        if params.from_id is not None:
            query["fromId"] = params.from_id
        if params.start_time is not None:
            query["startTime"] = params.start_time
        if params.end_time is not None:
            query["endTime"] = params.end_time
        resp = await client.request("GET", "/api/v3/aggTrades", params=query, auth="none")
        trades: list[dict[str, Any]] = resp.json()

        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json({"symbol": params.symbol, "count": len(trades), "trades": trades}))

        shown = trades[:MAX_TRADES_DISPLAY]
        lines = [
            f"# Aggregate trades — {params.symbol}",
            "",
            f"Showing {len(shown)} of {len(trades)} trade(s).",
            "",
            "| agg id | time | price | qty | trade ids | side |",
            "|---|---|---|---|---|---|",
        ]
        for t in shown:
            side = "maker-buy" if t.get("m") else "taker-buy"
            lines.append(
                f"| {t.get('a')} | {epoch_to_human(t.get('T'))} | {fmt_num(t.get('p'))} | {fmt_num(t.get('q'))} | "
                f"{t.get('f')}-{t.get('l')} | {side} |"
            )
        if not trades:
            lines.append("_No trades._")
        elif len(trades) > MAX_TRADES_DISPLAY:
            lines.append("")
            lines.append(f"_...{len(trades) - MAX_TRADES_DISPLAY} more not shown — use response_format=json._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


# -- binance_get_klines / binance_get_ui_klines -------------------------------------


class KlineInterval(StrEnum):
    """The 16 candle intervals Binance supports."""

    S1 = "1s"
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"
    MO1 = "1M"


class KlinesInput(BaseModel):
    """Input for `binance_get_klines`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str = Field(description="Trading pair symbol, e.g. BTCUSDT.")
    interval: KlineInterval = Field(description="Candle interval.")
    start_time: int | None = Field(default=None, description="Window start, epoch ms or ISO-8601.")
    end_time: int | None = Field(
        default=None, description="Window end, epoch ms or ISO-8601. Always UTC, even with `time_zone`."
    )
    time_zone: str | None = Field(
        default=None, description="Timezone offset for candle bucketing, e.g. '+08:00' or '8' (default UTC)."
    )
    limit: int = Field(default=500, ge=1, le=1000, description="Number of candles (Binance max 1000).")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        return _normalize_symbol(v)

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _validate_times(cls, v: Any) -> int | None:
        return _to_ms(v)


def _build_kline_query(params: KlinesInput) -> dict[str, Any]:
    query: dict[str, Any] = {"symbol": params.symbol, "interval": params.interval.value, "limit": params.limit}
    if params.start_time is not None:
        query["startTime"] = params.start_time
    if params.end_time is not None:
        query["endTime"] = params.end_time
    if params.time_zone is not None:
        query["timeZone"] = params.time_zone
    return query


def _render_klines(params: KlinesInput, raw: list[list[Any]], title: str) -> str:
    if params.response_format is ResponseFormat.JSON:
        return clip_response(
            to_json({"symbol": params.symbol, "interval": params.interval.value, "count": len(raw), "klines": raw})
        )
    shown = raw[:MAX_KLINES_DISPLAY]
    lines = [
        f"# {title} — {params.symbol} ({params.interval.value})",
        "",
        f"Showing {len(shown)} of {len(raw)} candle(s).",
        "",
        "| open time | open | high | low | close | volume | close time |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in shown:
        lines.append(
            f"| {epoch_to_human(k[0])} | {fmt_num(k[1])} | {fmt_num(k[2])} | {fmt_num(k[3])} | "
            f"{fmt_num(k[4])} | {fmt_num(k[5])} | {epoch_to_human(k[6])} |"
        )
    if not raw:
        lines.append("_No candles._")
    elif len(raw) > MAX_KLINES_DISPLAY:
        lines.append("")
        lines.append(f"_...{len(raw) - MAX_KLINES_DISPLAY} more not shown — use response_format=json._")
    return clip_response("\n".join(lines))


@mcp.tool(
    name="binance_get_klines",
    annotations=ToolAnnotations(
        title="Binance Klines",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_klines(params: KlinesInput) -> str:
    """Fetch OHLCV candlestick data for a symbol.

    Calls `GET /api/v3/klines` (weight 2).

    When to Use:
    - For price history / technical analysis over a chosen interval and window.

    When NOT to Use:
    - For presentation-smoothed candles matching Binance's own chart UI — use
      `binance_get_ui_klines` instead.
    - For the single latest price — use `binance_get_ticker_price`.

    Returns:
    Markdown: a table of up to 100 candles (open time, OHLC, volume, close time). JSON:
    the raw array-of-arrays Binance returns (up to `limit`), uncapped.

    Pagination:
    Walk forward with `start_time` set to the previous page's last `close time` + 1ms;
    `start_time`/`end_time` are always interpreted in UTC even when `time_zone` is set.

    Examples:
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 200}

    Error Handling:
    An unknown symbol raises Binance `-1121 Invalid symbol`; an invalid interval is
    rejected locally by the `KlineInterval` enum.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/api/v3/klines", params=_build_kline_query(params), auth="none")
        raw: list[list[Any]] = resp.json()
        return _render_klines(params, raw, "Klines")
    except Exception as exc:
        return handle_api_error(exc)


class UiKlinesInput(KlinesInput):
    """Input for `binance_get_ui_klines` (same shape as `KlinesInput`)."""


@mcp.tool(
    name="binance_get_ui_klines",
    annotations=ToolAnnotations(
        title="Binance UI Klines",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_ui_klines(params: UiKlinesInput) -> str:
    """Fetch presentation-adjusted candlestick data, matching Binance's own chart UI.

    Calls `GET /api/v3/uiKlines` (weight 2). Same array shape and parameters as
    `binance_get_klines`; Binance smooths/adjusts these for display purposes.

    When to Use:
    - When the numbers need to match what a user sees on binance.com/binance app charts.

    When NOT to Use:
    - For raw exchange candles used in calculations — use `binance_get_klines`.

    Returns:
    Same shape as `binance_get_klines`: markdown table (capped at 100 rows) or JSON.

    Examples:
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 200}

    Error Handling:
    Same as `binance_get_klines`.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/api/v3/uiKlines", params=_build_kline_query(params), auth="none")
        raw: list[list[Any]] = resp.json()
        return _render_klines(params, raw, "UI Klines")
    except Exception as exc:
        return handle_api_error(exc)


# -- binance_get_avg_price ----------------------------------------------------------


class AvgPriceInput(BaseModel):
    """Input for `binance_get_avg_price`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str = Field(description="Trading pair symbol, e.g. BTCUSDT.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        return _normalize_symbol(v)


@mcp.tool(
    name="binance_get_avg_price",
    annotations=ToolAnnotations(
        title="Binance Average Price",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_avg_price(params: AvgPriceInput) -> str:
    """Fetch the current average price over Binance's configured window (typically 5 min).

    Calls `GET /api/v3/avgPrice` (weight 2).

    When to Use:
    - As a smoothed reference price, e.g. for MARKET order sanity checks.

    When NOT to Use:
    - For the latest tick price — use `binance_get_ticker_price`.

    Returns:
    Markdown: the price, the averaging window in minutes, and the close time. JSON: the
    raw `{mins, price, closeTime}` object.

    Examples:
    params = {"symbol": "BTCUSDT"}

    Error Handling:
    An unknown symbol raises Binance `-1121 Invalid symbol`.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/api/v3/avgPrice", params={"symbol": params.symbol}, auth="none")
        data = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        return clip_response(
            f"# Average price — {params.symbol}\n\n"
            f"- **price**: {fmt_num(data.get('price'))} (over the last {data.get('mins')} minute(s))\n"
            f"- **close time**: {epoch_to_human(data.get('closeTime'))}"
        )
    except Exception as exc:
        return handle_api_error(exc)


# -- binance_get_ticker_24h ----------------------------------------------------------


class Ticker24hInput(BaseModel):
    """Input for `binance_get_ticker_24h`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str | None = Field(default=None, description="Single symbol. Mutually exclusive with `symbols`.")
    symbols: list[str] | None = Field(
        default=None,
        description="Multiple symbols. Mutually exclusive with `symbol`. Omit both for ALL symbols — "
        "weight 80, use sparingly.",
    )
    type: TickerType = Field(default=TickerType.FULL, description="FULL (all fields) or MINI (fewer fields).")
    symbol_status: SymbolStatus | None = Field(
        default=None, description="Filter by symbol status (only meaningful without `symbol`/`symbols`)."
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str | None) -> str | None:
        return _normalize_symbol(v) if v is not None else v

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, v: list[str] | None) -> list[str] | None:
        return [_normalize_symbol(s) for s in v] if v is not None else v

    @model_validator(mode="after")
    def _check_combo(self) -> Ticker24hInput:
        if self.symbol is not None and self.symbols is not None:
            raise ValueError("Pass either `symbol` or `symbols`, not both.")
        return self


@mcp.tool(
    name="binance_get_ticker_24h",
    annotations=ToolAnnotations(
        title="Binance 24h Ticker",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_ticker_24h(params: Ticker24hInput) -> str:
    """Fetch 24-hour rolling price change statistics.

    Calls `GET /api/v3/ticker/24hr`. Weight: `symbol` → 2; `symbols` → 2 for 1-20, 40 for
    21-100, 80 for 101+; **no symbol at all (all 3700+ symbols) → weight 80 — use sparingly.**

    When to Use:
    - For a market snapshot: price change %, high/low, volume over the last 24h.

    When NOT to Use:
    - For a fixed calendar-day window — use `binance_get_trading_day_ticker`.
    - For a custom rolling window — use `binance_get_rolling_ticker`.

    Returns:
    Markdown: up to 50 symbols as stat blocks. JSON: `count`/`shown` plus the tickers.

    Examples:
    params = {"symbol": "BTCUSDT"}
    params = {"symbols": ["BTCUSDT", "ETHUSDT"], "type": "MINI"}

    Error Handling:
    An unknown symbol raises Binance `-1121 Invalid symbol`.
    """
    try:
        client = get_client()
        query: dict[str, Any] = {"type": params.type.value}
        if params.symbol is not None:
            query["symbol"] = params.symbol
        if params.symbols is not None:
            query["symbols"] = params.symbols
        if params.symbol_status is not None:
            query["symbolStatus"] = params.symbol_status.value
        resp = await client.request("GET", "/api/v3/ticker/24hr", params=query, auth="none")
        data = resp.json()
        items: list[dict[str, Any]] = data if isinstance(data, list) else [data]

        if params.response_format is ResponseFormat.JSON:
            shown_json = items[:MAX_TICKER_DISPLAY]
            payload: dict[str, Any] = {"count": len(items), "shown": len(shown_json), "tickers": shown_json}
            if len(items) > MAX_TICKER_DISPLAY:
                payload["note"] = (
                    f"{len(items)} tickers total; showing first {MAX_TICKER_DISPLAY} — narrow with `symbol`/`symbols`."
                )
            return clip_response(to_json(payload))

        shown = items[:MAX_TICKER_DISPLAY]
        lines = [f"# 24h ticker ({params.type.value})", "", f"Showing {len(shown)} of {len(items)} symbol(s).", ""]
        for t in shown:
            lines.append(_format_ticker_stats(t))
        if not items:
            lines.append("_No tickers matched._")
        elif len(items) > MAX_TICKER_DISPLAY:
            lines.append(f"_...{len(items) - MAX_TICKER_DISPLAY} more not shown — narrow with `symbol`/`symbols`._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


# -- binance_get_ticker_price --------------------------------------------------------


class TickerPriceInput(BaseModel):
    """Input for `binance_get_ticker_price`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str | None = Field(default=None, description="Single symbol. Mutually exclusive with `symbols`.")
    symbols: list[str] | None = Field(
        default=None,
        description="Multiple symbols. Mutually exclusive with `symbol`. Omit both for ALL prices (weight 4).",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str | None) -> str | None:
        return _normalize_symbol(v) if v is not None else v

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, v: list[str] | None) -> list[str] | None:
        return [_normalize_symbol(s) for s in v] if v is not None else v

    @model_validator(mode="after")
    def _check_combo(self) -> TickerPriceInput:
        if self.symbol is not None and self.symbols is not None:
            raise ValueError("Pass either `symbol` or `symbols`, not both.")
        return self


@mcp.tool(
    name="binance_get_ticker_price",
    annotations=ToolAnnotations(
        title="Binance Ticker Price",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_ticker_price(params: TickerPriceInput) -> str:
    """Fetch the latest price for one, several, or all symbols.

    Calls `GET /api/v3/ticker/price`. Weight: `symbol` → 2; omitted or `symbols` → 4
    (fetching ALL prices is a flat weight-4 call — cheap even for the whole market).

    When to Use:
    - For the current tick price with the least overhead of any ticker endpoint.

    When NOT to Use:
    - For bid/ask spread — use `binance_get_book_ticker`.
    - For 24h stats (change %, volume) — use `binance_get_ticker_24h`.

    Returns:
    Markdown: a symbol/price table, capped at 100 rows. JSON: `count`/`shown` plus prices.

    Examples:
    params = {"symbol": "BTCUSDT"}
    params = {"symbols": ["BTCUSDT", "ETHUSDT"]}

    Error Handling:
    An unknown symbol raises Binance `-1121 Invalid symbol`.
    """
    try:
        client = get_client()
        query: dict[str, Any] = {}
        if params.symbol is not None:
            query["symbol"] = params.symbol
        if params.symbols is not None:
            query["symbols"] = params.symbols
        resp = await client.request("GET", "/api/v3/ticker/price", params=query, auth="none")
        data = resp.json()
        items: list[dict[str, Any]] = data if isinstance(data, list) else [data]

        if params.response_format is ResponseFormat.JSON:
            shown_json = items[:MAX_PRICE_DISPLAY]
            payload: dict[str, Any] = {"count": len(items), "shown": len(shown_json), "prices": shown_json}
            if len(items) > MAX_PRICE_DISPLAY:
                payload["note"] = (
                    f"{len(items)} prices total; showing first {MAX_PRICE_DISPLAY} — narrow with `symbol`/`symbols`."
                )
            return clip_response(to_json(payload))

        shown = items[:MAX_PRICE_DISPLAY]
        lines = [
            "# Latest price",
            "",
            f"Showing {len(shown)} of {len(items)} symbol(s).",
            "",
            "| symbol | price |",
            "|---|---|",
        ]
        for t in shown:
            lines.append(f"| {t.get('symbol')} | {fmt_num(t.get('price'))} |")
        if not items:
            lines.append("_No prices matched._")
        elif len(items) > MAX_PRICE_DISPLAY:
            lines.append("")
            lines.append(f"_...{len(items) - MAX_PRICE_DISPLAY} more not shown — narrow with `symbol`/`symbols`._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


# -- binance_get_book_ticker ----------------------------------------------------------


class BookTickerInput(BaseModel):
    """Input for `binance_get_book_ticker`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str | None = Field(default=None, description="Single symbol. Mutually exclusive with `symbols`.")
    symbols: list[str] | None = Field(
        default=None, description="Multiple symbols. Mutually exclusive with `symbol`. Omit both for ALL symbols."
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str | None) -> str | None:
        return _normalize_symbol(v) if v is not None else v

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, v: list[str] | None) -> list[str] | None:
        return [_normalize_symbol(s) for s in v] if v is not None else v

    @model_validator(mode="after")
    def _check_combo(self) -> BookTickerInput:
        if self.symbol is not None and self.symbols is not None:
            raise ValueError("Pass either `symbol` or `symbols`, not both.")
        return self


@mcp.tool(
    name="binance_get_book_ticker",
    annotations=ToolAnnotations(
        title="Binance Book Ticker",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_book_ticker(params: BookTickerInput) -> str:
    """Fetch the best bid/ask price and quantity for one, several, or all symbols.

    Calls `GET /api/v3/ticker/bookTicker`. Weight: `symbol` → 2; omitted or `symbols` → 4.

    When to Use:
    - For the current spread and top-of-book size without the full depth of
      `binance_get_order_book`.

    When NOT to Use:
    - For multiple price levels — use `binance_get_order_book`.

    Returns:
    Markdown: a table of bid/ask price+qty per symbol, capped at 100 rows. JSON:
    `count`/`shown` plus the tickers.

    Examples:
    params = {"symbol": "BTCUSDT"}

    Error Handling:
    An unknown symbol raises Binance `-1121 Invalid symbol`.
    """
    try:
        client = get_client()
        query: dict[str, Any] = {}
        if params.symbol is not None:
            query["symbol"] = params.symbol
        if params.symbols is not None:
            query["symbols"] = params.symbols
        resp = await client.request("GET", "/api/v3/ticker/bookTicker", params=query, auth="none")
        data = resp.json()
        items: list[dict[str, Any]] = data if isinstance(data, list) else [data]

        if params.response_format is ResponseFormat.JSON:
            shown_json = items[:MAX_PRICE_DISPLAY]
            payload: dict[str, Any] = {"count": len(items), "shown": len(shown_json), "tickers": shown_json}
            if len(items) > MAX_PRICE_DISPLAY:
                payload["note"] = (
                    f"{len(items)} tickers total; showing first {MAX_PRICE_DISPLAY} — narrow with `symbol`/`symbols`."
                )
            return clip_response(to_json(payload))

        shown = items[:MAX_PRICE_DISPLAY]
        lines = [
            "# Book ticker",
            "",
            f"Showing {len(shown)} of {len(items)} symbol(s).",
            "",
            "| symbol | bid price | bid qty | ask price | ask qty |",
            "|---|---|---|---|---|",
        ]
        for t in shown:
            lines.append(
                f"| {t.get('symbol')} | {fmt_num(t.get('bidPrice'))} | {fmt_num(t.get('bidQty'))} | "
                f"{fmt_num(t.get('askPrice'))} | {fmt_num(t.get('askQty'))} |"
            )
        if not items:
            lines.append("_No tickers matched._")
        elif len(items) > MAX_PRICE_DISPLAY:
            lines.append("")
            lines.append(f"_...{len(items) - MAX_PRICE_DISPLAY} more not shown — narrow with `symbol`/`symbols`._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


# -- binance_get_rolling_ticker ------------------------------------------------------


class RollingTickerInput(BaseModel):
    """Input for `binance_get_rolling_ticker`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str | None = Field(default=None, description="Single symbol. One of `symbol`/`symbols` is required.")
    symbols: list[str] | None = Field(
        default=None, description="Multiple symbols (max 100). One of `symbol`/`symbols` is required."
    )
    window_size: str | None = Field(
        default=None,
        pattern=r"^\d+[mhd]$",
        description="Rolling window, e.g. '1h', '4d' (1m-59m, 1h-23h, 1d-7d; default 1d).",
    )
    type: TickerType = Field(default=TickerType.FULL, description="FULL (all fields) or MINI (fewer fields).")
    symbol_status: SymbolStatus | None = Field(default=None, description="Filter by symbol status.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str | None) -> str | None:
        return _normalize_symbol(v) if v is not None else v

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, v: list[str] | None) -> list[str] | None:
        return [_normalize_symbol(s) for s in v] if v is not None else v

    @model_validator(mode="after")
    def _check_combo(self) -> RollingTickerInput:
        if self.symbol is not None and self.symbols is not None:
            raise ValueError("Pass either `symbol` or `symbols`, not both.")
        if self.symbol is None and self.symbols is None:
            raise ValueError("`symbol` or `symbols` is required for the rolling-window ticker.")
        if self.symbols is not None and len(self.symbols) > 100:
            raise ValueError("At most 100 symbols per request.")
        return self


@mcp.tool(
    name="binance_get_rolling_ticker",
    annotations=ToolAnnotations(
        title="Binance Rolling Window Ticker",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_rolling_ticker(params: RollingTickerInput) -> str:
    """Fetch price change statistics over an arbitrary rolling window.

    Calls `GET /api/v3/ticker` (weight 4 per symbol, capped at 200 once >50 symbols
    requested). Unlike `binance_get_ticker_24h`, the window is not fixed at 24 hours.

    When to Use:
    - For a custom window (e.g. 4h, 7d) that the fixed 24h/trading-day tickers don't cover.

    When NOT to Use:
    - For the standard 24h window — `binance_get_ticker_24h` is cheaper for that case.

    Returns:
    Markdown: stat blocks per symbol (up to 50 shown). JSON: `count`/`shown` plus tickers.

    Examples:
    params = {"symbol": "BTCUSDT", "window_size": "4h"}
    params = {"symbols": ["BTCUSDT", "ETHUSDT"], "window_size": "7d"}

    Error Handling:
    `window_size` outside 1m-59m/1h-23h/1d-7d is rejected by Binance; more than 100
    symbols or neither `symbol` nor `symbols` is rejected locally.
    """
    try:
        client = get_client()
        query: dict[str, Any] = {"type": params.type.value}
        if params.symbol is not None:
            query["symbol"] = params.symbol
        if params.symbols is not None:
            query["symbols"] = params.symbols
        if params.window_size is not None:
            query["windowSize"] = params.window_size
        if params.symbol_status is not None:
            query["symbolStatus"] = params.symbol_status.value
        resp = await client.request("GET", "/api/v3/ticker", params=query, auth="none")
        data = resp.json()
        items: list[dict[str, Any]] = data if isinstance(data, list) else [data]

        if params.response_format is ResponseFormat.JSON:
            shown_json = items[:MAX_TICKER_DISPLAY]
            payload: dict[str, Any] = {"count": len(items), "shown": len(shown_json), "tickers": shown_json}
            if len(items) > MAX_TICKER_DISPLAY:
                payload["note"] = (
                    f"{len(items)} tickers total; showing first {MAX_TICKER_DISPLAY} — narrow with `symbol`/`symbols`."
                )
            return clip_response(to_json(payload))

        shown = items[:MAX_TICKER_DISPLAY]
        window = params.window_size or "1d"
        lines = [f"# Rolling ticker ({window})", "", f"Showing {len(shown)} of {len(items)} symbol(s).", ""]
        for t in shown:
            lines.append(_format_ticker_stats(t))
        if not items:
            lines.append("_No tickers matched._")
        elif len(items) > MAX_TICKER_DISPLAY:
            lines.append(f"_...{len(items) - MAX_TICKER_DISPLAY} more not shown — narrow with `symbol`/`symbols`._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


# -- binance_get_trading_day_ticker --------------------------------------------------


class TradingDayTickerInput(BaseModel):
    """Input for `binance_get_trading_day_ticker`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str | None = Field(default=None, description="Single symbol. One of `symbol`/`symbols` is required.")
    symbols: list[str] | None = Field(
        default=None, description="Multiple symbols (max 100). One of `symbol`/`symbols` is required."
    )
    time_zone: str | None = Field(
        default=None, description="Timezone offset for the trading-day boundary, e.g. '+08:00' (default UTC)."
    )
    type: TickerType = Field(default=TickerType.FULL, description="FULL (all fields) or MINI (fewer fields).")
    symbol_status: SymbolStatus | None = Field(default=None, description="Filter by symbol status.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str | None) -> str | None:
        return _normalize_symbol(v) if v is not None else v

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, v: list[str] | None) -> list[str] | None:
        return [_normalize_symbol(s) for s in v] if v is not None else v

    @model_validator(mode="after")
    def _check_combo(self) -> TradingDayTickerInput:
        if self.symbol is not None and self.symbols is not None:
            raise ValueError("Pass either `symbol` or `symbols`, not both.")
        if self.symbol is None and self.symbols is None:
            raise ValueError("`symbol` or `symbols` is required for the trading-day ticker.")
        if self.symbols is not None and len(self.symbols) > 100:
            raise ValueError("At most 100 symbols per request.")
        return self


@mcp.tool(
    name="binance_get_trading_day_ticker",
    annotations=ToolAnnotations(
        title="Binance Trading Day Ticker",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_trading_day_ticker(params: TradingDayTickerInput) -> str:
    """Fetch price change statistics for the current trading day (a fixed calendar window).

    Calls `GET /api/v3/ticker/tradingDay` (weight 4 per symbol, capped at 200 once >50
    symbols requested; max 100 symbols per request).

    When to Use:
    - For "today's" stats aligned to a specific timezone's midnight — e.g. `time_zone="+08:00"`.

    When NOT to Use:
    - For a rolling 24h window instead of a calendar day — use `binance_get_ticker_24h`.

    Returns:
    Markdown: stat blocks per symbol (up to 50 shown). JSON: `count`/`shown` plus tickers.

    Examples:
    params = {"symbol": "BTCUSDT", "time_zone": "+08:00"}

    Error Handling:
    More than 100 symbols or neither `symbol` nor `symbols` is rejected locally.
    """
    try:
        client = get_client()
        query: dict[str, Any] = {"type": params.type.value}
        if params.symbol is not None:
            query["symbol"] = params.symbol
        if params.symbols is not None:
            query["symbols"] = params.symbols
        if params.time_zone is not None:
            query["timeZone"] = params.time_zone
        if params.symbol_status is not None:
            query["symbolStatus"] = params.symbol_status.value
        resp = await client.request("GET", "/api/v3/ticker/tradingDay", params=query, auth="none")
        data = resp.json()
        items: list[dict[str, Any]] = data if isinstance(data, list) else [data]

        if params.response_format is ResponseFormat.JSON:
            shown_json = items[:MAX_TICKER_DISPLAY]
            payload: dict[str, Any] = {"count": len(items), "shown": len(shown_json), "tickers": shown_json}
            if len(items) > MAX_TICKER_DISPLAY:
                payload["note"] = (
                    f"{len(items)} tickers total; showing first {MAX_TICKER_DISPLAY} — narrow with `symbol`/`symbols`."
                )
            return clip_response(to_json(payload))

        shown = items[:MAX_TICKER_DISPLAY]
        lines = ["# Trading day ticker", "", f"Showing {len(shown)} of {len(items)} symbol(s).", ""]
        for t in shown:
            lines.append(_format_ticker_stats(t))
        if not items:
            lines.append("_No tickers matched._")
        elif len(items) > MAX_TICKER_DISPLAY:
            lines.append(f"_...{len(items) - MAX_TICKER_DISPLAY} more not shown — narrow with `symbol`/`symbols`._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)
