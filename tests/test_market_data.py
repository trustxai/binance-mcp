"""Unit tests for the public market-data tools against a fake client.

All endpoints here are `auth="none"` — no credentials required by Binance — but the
`live` marker still gates on `BINANCE_API_KEY`/secret per the shared conftest, so the
live smoke tests below are skipped in CI same as every other module's.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from binance_mcp.formatters import ResponseFormat
from binance_mcp.tools import market_data as md


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Records every call (method, path, kwargs); optionally raises."""

    def __init__(self, payload: Any = None, exc: Exception | None = None) -> None:
        self._payload = payload
        self._exc = exc
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, path, kwargs))
        if self._exc is not None:
            raise self._exc
        return _FakeResponse(self._payload)


def _patch_client(monkeypatch: pytest.MonkeyPatch, fake: _FakeClient) -> None:
    monkeypatch.setattr(md, "get_client", lambda: fake)


def _status_error(status: int, body: dict[str, Any]) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.binance.com/api/v3/ticker/price")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


# -- binance_get_exchange_info -------------------------------------------------------

EXCHANGE_INFO_PAYLOAD: dict[str, Any] = {
    "timezone": "UTC",
    "serverTime": 1700000000000,
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "orderTypes": ["LIMIT", "MARKET"],
            "filters": [
                {
                    "filterType": "LOT_SIZE",
                    "minQty": "0.00001000",
                    "maxQty": "9000.00000000",
                    "stepSize": "0.00001000",
                },
                {
                    "filterType": "PRICE_FILTER",
                    "minPrice": "0.01000000",
                    "maxPrice": "1000000.00000000",
                    "tickSize": "0.01000000",
                },
                {"filterType": "NOTIONAL", "minNotional": "5.00000000", "applyToMarket": True},
                {
                    "filterType": "MARKET_LOT_SIZE",
                    "minQty": "0.00000000",
                    "maxQty": "100.00000000",
                    "stepSize": "0.00000000",
                },
            ],
        }
    ],
}


async def test_exchange_info_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=EXCHANGE_INFO_PAYLOAD)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_exchange_info(md.ExchangeInfoInput(symbol="btcusdt"))

    assert "## BTCUSDT" in result
    assert "LOT_SIZE: min 0.00001, max 9000, step 0.00001" in result
    assert "NOTIONAL: min 5" in result
    assert fake.calls == [("GET", "/api/v3/exchangeInfo", {"params": {"symbol": "BTCUSDT"}, "auth": "none"})]


async def test_exchange_info_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=EXCHANGE_INFO_PAYLOAD)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_exchange_info(
        md.ExchangeInfoInput(symbol="BTCUSDT", response_format=ResponseFormat.JSON)
    )

    payload = json.loads(result)
    assert payload["count"] == 1
    assert payload["symbols"][0]["symbol"] == "BTCUSDT"


async def test_exchange_info_caps_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    symbols = [
        {
            "symbol": f"SYM{i}USDT",
            "status": "TRADING",
            "baseAsset": f"SYM{i}",
            "quoteAsset": "USDT",
            "orderTypes": ["LIMIT"],
            "filters": [],
        }
        for i in range(55)
    ]
    fake = _FakeClient(payload={"timezone": "UTC", "serverTime": 1, "symbols": symbols})
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_exchange_info(md.ExchangeInfoInput())

    assert "Total symbols matched: 55" in result
    assert "...5 more symbols not shown" in result


def test_exchange_info_rejects_symbol_and_symbols() -> None:
    with pytest.raises(ValidationError):
        md.ExchangeInfoInput(symbol="BTCUSDT", symbols=["ETHUSDT"])


def test_exchange_info_rejects_symbol_with_permissions() -> None:
    with pytest.raises(ValidationError):
        md.ExchangeInfoInput(symbol="BTCUSDT", permissions=["SPOT"])


def test_exchange_info_rejects_invalid_symbol() -> None:
    with pytest.raises(ValidationError):
        md.ExchangeInfoInput(symbol="B")


async def test_exchange_info_falls_back_to_legacy_min_notional_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some symbols still carry the legacy MIN_NOTIONAL filter instead of NOTIONAL."""
    payload = {
        "timezone": "UTC",
        "serverTime": 1700000000000,
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "orderTypes": ["LIMIT"],
                "filters": [{"filterType": "MIN_NOTIONAL", "minNotional": "10.00000000", "applyToMarket": True}],
            }
        ],
    }
    fake = _FakeClient(payload=payload)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_exchange_info(md.ExchangeInfoInput(symbol="BTCUSDT"))

    assert "NOTIONAL: min 10, applyToMarket True" in result


# -- binance_get_order_book ----------------------------------------------------------

ORDER_BOOK_PAYLOAD: dict[str, Any] = {
    "lastUpdateId": 123456,
    "bids": [["50000.00", "1.5"], ["49999.00", "2.0"]],
    "asks": [["50001.00", "1.0"], ["50002.00", "0.5"]],
}


async def test_order_book_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=ORDER_BOOK_PAYLOAD)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_order_book(md.OrderBookInput(symbol="btcusdt", limit=10))

    assert "lastUpdateId: 123456" in result
    assert "| 50000 | 1.5 |" in result
    assert "| 50001 | 1 |" in result
    assert fake.calls == [("GET", "/api/v3/depth", {"params": {"symbol": "BTCUSDT", "limit": 10}, "auth": "none"})]


async def test_order_book_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=ORDER_BOOK_PAYLOAD)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_order_book(md.OrderBookInput(symbol="BTCUSDT", response_format=ResponseFormat.JSON))

    payload = json.loads(result)
    assert payload["lastUpdateId"] == 123456
    assert payload["bids"] == ORDER_BOOK_PAYLOAD["bids"]


async def test_order_book_caps_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    bids = [[str(100 - i), "1"] for i in range(60)]
    asks = [[str(101 + i), "1"] for i in range(60)]
    fake = _FakeClient(payload={"lastUpdateId": 1, "bids": bids, "asks": asks})
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_order_book(md.OrderBookInput(symbol="BTCUSDT", limit=500))

    assert "top 50 of 60" in result
    assert "Showing top 50 levels/side" in result


def test_order_book_rejects_limit_over_500() -> None:
    with pytest.raises(ValidationError):
        md.OrderBookInput(symbol="BTCUSDT", limit=501)


# -- binance_get_recent_trades --------------------------------------------------------

TRADES_PAYLOAD: list[dict[str, Any]] = [
    {"id": 1, "price": "50000.00", "qty": "0.01", "time": 1700000000000, "isBuyerMaker": True},
    {"id": 2, "price": "50001.00", "qty": "0.02", "time": 1700000001000, "isBuyerMaker": False},
]


async def test_recent_trades_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=TRADES_PAYLOAD)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_recent_trades(md.RecentTradesInput(symbol="btcusdt", limit=2))

    assert "Showing 2 of 2 trade(s)" in result
    # id 1 isBuyerMaker=True -> the taker (aggressor) sold; id 2 isBuyerMaker=False -> bought.
    assert "| 1 | 2023-11-14 22:13:20 UTC | 50000 | 0.01 | sell |" in result
    assert "| 2 | 2023-11-14 22:13:21 UTC | 50001 | 0.02 | buy |" in result
    assert fake.calls == [("GET", "/api/v3/trades", {"params": {"symbol": "BTCUSDT", "limit": 2}, "auth": "none"})]


async def test_recent_trades_caps_display(monkeypatch: pytest.MonkeyPatch) -> None:
    trades = [{"id": i, "price": "1", "qty": "1", "time": 1, "isBuyerMaker": True} for i in range(150)]
    fake = _FakeClient(payload=trades)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_recent_trades(md.RecentTradesInput(symbol="BTCUSDT", limit=150))

    assert "Showing 100 of 150 trade(s)" in result
    assert "...50 more not shown" in result


def test_recent_trades_rejects_limit_over_1000() -> None:
    with pytest.raises(ValidationError):
        md.RecentTradesInput(symbol="BTCUSDT", limit=1001)


# -- binance_get_agg_trades ------------------------------------------------------------

AGG_TRADES_PAYLOAD: list[dict[str, Any]] = [
    {"a": 1, "p": "50000.00", "q": "0.01", "f": 10, "l": 12, "T": 1700000000000, "m": True},
]


async def test_agg_trades_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=AGG_TRADES_PAYLOAD)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_agg_trades(md.AggTradesInput(symbol="btcusdt", from_id=5))

    assert "10-12" in result
    # m=True -> the taker (aggressor) sold, same convention as binance_get_recent_trades.
    assert "| sell |" in result
    assert fake.calls == [
        (
            "GET",
            "/api/v3/aggTrades",
            {"params": {"symbol": "BTCUSDT", "limit": 500, "fromId": 5}, "auth": "none"},
        )
    ]


async def test_agg_trades_accepts_iso_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=[])
    _patch_client(monkeypatch, fake)
    params = md.AggTradesInput(symbol="BTCUSDT", start_time="2024-01-01T00:00:00Z", end_time="2024-01-01T00:30:00Z")

    assert params.start_time == 1704067200000

    await md.binance_get_agg_trades(params)

    sent_params = fake.calls[0][2]["params"]
    assert sent_params["startTime"] == 1704067200000
    assert sent_params["endTime"] == 1704069000000
    assert "fromId" not in sent_params


def test_agg_trades_rejects_from_id_with_times() -> None:
    with pytest.raises(ValidationError):
        md.AggTradesInput(symbol="BTCUSDT", from_id=1, start_time=1700000000000)


def test_agg_trades_start_time_schema_allows_string() -> None:
    """A `mode="before"` validator normalises to ms but does not narrow the published
    schema — start_time/end_time must still advertise ISO-8601 strings as accepted."""
    schema = md.AggTradesInput.model_json_schema()
    branch_types = {branch.get("type") for branch in schema["properties"]["start_time"]["anyOf"]}
    assert "string" in branch_types
    assert "integer" in branch_types


# -- binance_get_klines / binance_get_ui_klines -----------------------------------------

KLINE_ROW = [
    1700000000000,
    "50000.00",
    "50100.00",
    "49900.00",
    "50050.00",
    "10.5",
    1700003599999,
    "525000",
    100,
    "5",
    "250000",
    "0",
]


async def test_klines_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=[KLINE_ROW])
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_klines(md.KlinesInput(symbol="btcusdt", interval=md.KlineInterval.H1))

    assert "Klines — BTCUSDT (1h)" in result
    assert "50000" in result
    assert fake.calls == [
        (
            "GET",
            "/api/v3/klines",
            {"params": {"symbol": "BTCUSDT", "interval": "1h", "limit": 500}, "auth": "none"},
        )
    ]


async def test_klines_caps_display(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [KLINE_ROW for _ in range(150)]
    fake = _FakeClient(payload=rows)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_klines(md.KlinesInput(symbol="BTCUSDT", interval=md.KlineInterval.M1, limit=150))

    assert "Showing 100 of 150 candle(s)" in result


async def test_klines_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=[KLINE_ROW])
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_klines(
        md.KlinesInput(symbol="BTCUSDT", interval=md.KlineInterval.H1, response_format=ResponseFormat.JSON)
    )

    payload = json.loads(result)
    assert payload["klines"] == [KLINE_ROW]


async def test_klines_passes_start_end_time_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=[])
    _patch_client(monkeypatch, fake)

    await md.binance_get_klines(
        md.KlinesInput(
            symbol="BTCUSDT",
            interval=md.KlineInterval.D1,
            start_time="2024-01-01T00:00:00Z",
            time_zone="+08:00",
        )
    )

    sent_params = fake.calls[0][2]["params"]
    assert sent_params["startTime"] == 1704067200000
    assert sent_params["timeZone"] == "+08:00"
    assert "endTime" not in sent_params


def test_klines_start_time_schema_allows_string() -> None:
    schema = md.KlinesInput.model_json_schema()
    branch_types = {branch.get("type") for branch in schema["properties"]["start_time"]["anyOf"]}
    assert "string" in branch_types
    assert "integer" in branch_types


def test_ui_klines_input_is_subclass_of_klines_input() -> None:
    assert issubclass(md.UiKlinesInput, md.KlinesInput)


async def test_ui_klines_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=[KLINE_ROW])
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_ui_klines(md.UiKlinesInput(symbol="BTCUSDT", interval=md.KlineInterval.D1))

    assert "UI Klines — BTCUSDT (1d)" in result
    assert fake.calls == [
        (
            "GET",
            "/api/v3/uiKlines",
            {"params": {"symbol": "BTCUSDT", "interval": "1d", "limit": 500}, "auth": "none"},
        )
    ]


# -- binance_get_avg_price -------------------------------------------------------------


async def test_avg_price_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload={"mins": 5, "price": "50000.00000000", "closeTime": 1700000000000})
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_avg_price(md.AvgPriceInput(symbol="btcusdt"))

    assert "price**: 50000 (over the last 5 minute(s))" in result
    assert fake.calls == [("GET", "/api/v3/avgPrice", {"params": {"symbol": "BTCUSDT"}, "auth": "none"})]


# -- binance_get_ticker_24h ------------------------------------------------------------

# Only /ticker/24hr FULL carries bidPrice/askPrice — rolling and tradingDay never do,
# even at FULL (see TICKER_STATS_NO_BID_ASK below).
TICKER_STATS_FULL: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "priceChange": "100.00",
    "priceChangePercent": "0.5",
    "weightedAvgPrice": "50050.00",
    "lastPrice": "50100.00",
    "openPrice": "50000.00",
    "highPrice": "50200.00",
    "lowPrice": "49900.00",
    "volume": "1000.00",
    "bidPrice": "50099.00",
    "askPrice": "50101.00",
    "openTime": 1700000000000,
    "closeTime": 1700086400000,
    "count": 12345,
}

# Realistic /ticker/tradingDay or /ticker (rolling) FULL payload: no bidPrice/askPrice.
TICKER_STATS_NO_BID_ASK: dict[str, Any] = {
    k: v for k, v in TICKER_STATS_FULL.items() if k not in ("bidPrice", "askPrice")
}

# Realistic MINI payload (any of the three endpoints): no priceChange/weightedAvgPrice/bid/ask.
TICKER_STATS_MINI: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "openPrice": "50000.00",
    "highPrice": "50200.00",
    "lowPrice": "49900.00",
    "lastPrice": "50100.00",
    "volume": "1000.00",
    "openTime": 1700000000000,
    "closeTime": 1700086400000,
    "count": 12345,
}


async def test_ticker_24h_happy_path_single_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=TICKER_STATS_FULL)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_ticker_24h(md.Ticker24hInput(symbol="btcusdt"))

    assert "## BTCUSDT" in result
    assert "weighted avg" in result
    assert "bid: 50099; ask: 50101" in result
    assert fake.calls == [
        ("GET", "/api/v3/ticker/24hr", {"params": {"type": "FULL", "symbol": "BTCUSDT"}, "auth": "none"})
    ]


async def test_ticker_24h_mini_omits_change_weighted_avg_and_bid_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=TICKER_STATS_MINI)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_ticker_24h(md.Ticker24hInput(symbol="BTCUSDT", type=md.TickerType.MINI))

    assert "change:" not in result
    assert "weighted avg" not in result
    assert "bid:" not in result
    assert "trades: 12345" in result


async def test_ticker_24h_all_symbols_caps_display(monkeypatch: pytest.MonkeyPatch) -> None:
    tickers = [{**TICKER_STATS_FULL, "symbol": f"SYM{i}USDT"} for i in range(60)]
    fake = _FakeClient(payload=tickers)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_ticker_24h(md.Ticker24hInput())

    assert "Showing 50 of 60 symbol(s)" in result
    assert "...10 more not shown" in result
    assert fake.calls == [("GET", "/api/v3/ticker/24hr", {"params": {"type": "FULL"}, "auth": "none"})]


def test_ticker_24h_rejects_symbol_and_symbols() -> None:
    with pytest.raises(ValidationError):
        md.Ticker24hInput(symbol="BTCUSDT", symbols=["ETHUSDT"])


# -- binance_get_ticker_price ----------------------------------------------------------


async def test_ticker_price_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload={"symbol": "BTCUSDT", "price": "50000.00000000"})
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_ticker_price(md.TickerPriceInput(symbol="btcusdt"))

    assert "| BTCUSDT | 50000 |" in result
    assert fake.calls == [("GET", "/api/v3/ticker/price", {"params": {"symbol": "BTCUSDT"}, "auth": "none"})]


async def test_ticker_price_all_symbols_caps_display(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = [{"symbol": f"SYM{i}USDT", "price": "1"} for i in range(150)]
    fake = _FakeClient(payload=prices)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_ticker_price(md.TickerPriceInput())

    assert "Showing 100 of 150 symbol(s)" in result
    assert fake.calls == [("GET", "/api/v3/ticker/price", {"params": {}, "auth": "none"})]


def test_ticker_price_rejects_symbol_and_symbols() -> None:
    with pytest.raises(ValidationError):
        md.TickerPriceInput(symbol="BTCUSDT", symbols=["ETHUSDT"])


# -- binance_get_book_ticker ------------------------------------------------------------


async def test_book_ticker_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        payload={
            "symbol": "BTCUSDT",
            "bidPrice": "50000.00",
            "bidQty": "1.0",
            "askPrice": "50001.00",
            "askQty": "2.0",
        }
    )
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_book_ticker(md.BookTickerInput(symbol="btcusdt"))

    assert "| BTCUSDT | 50000 | 1 | 50001 | 2 |" in result
    assert fake.calls == [("GET", "/api/v3/ticker/bookTicker", {"params": {"symbol": "BTCUSDT"}, "auth": "none"})]


def test_book_ticker_rejects_symbol_and_symbols() -> None:
    with pytest.raises(ValidationError):
        md.BookTickerInput(symbol="BTCUSDT", symbols=["ETHUSDT"])


# -- binance_get_rolling_ticker ----------------------------------------------------------


async def test_rolling_ticker_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=TICKER_STATS_NO_BID_ASK)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_rolling_ticker(md.RollingTickerInput(symbol="btcusdt", window_size="4h"))

    assert "Rolling ticker (4h)" in result
    assert "weighted avg" in result
    # /ticker (rolling) never returns bidPrice/askPrice, even at FULL — must not render N/A.
    assert "bid:" not in result
    assert fake.calls == [
        (
            "GET",
            "/api/v3/ticker",
            {"params": {"type": "FULL", "symbol": "BTCUSDT", "windowSize": "4h"}, "auth": "none"},
        )
    ]


def test_rolling_ticker_requires_symbol_or_symbols() -> None:
    with pytest.raises(ValidationError):
        md.RollingTickerInput()


def test_rolling_ticker_rejects_too_many_symbols() -> None:
    with pytest.raises(ValidationError):
        md.RollingTickerInput(symbols=[f"SYM{i}USDT" for i in range(101)])


def test_rolling_ticker_rejects_bad_window_size() -> None:
    with pytest.raises(ValidationError):
        md.RollingTickerInput(symbol="BTCUSDT", window_size="4x")


@pytest.mark.parametrize("window_size", ["0m", "60m", "0h", "24h", "0d", "8d"])
def test_rolling_ticker_rejects_window_size_out_of_range(window_size: str) -> None:
    with pytest.raises(ValidationError):
        md.RollingTickerInput(symbol="BTCUSDT", window_size=window_size)


@pytest.mark.parametrize("window_size", ["1m", "59m", "1h", "23h", "1d", "7d"])
def test_rolling_ticker_accepts_window_size_in_range(window_size: str) -> None:
    params = md.RollingTickerInput(symbol="BTCUSDT", window_size=window_size)
    assert params.window_size == window_size


# -- binance_get_trading_day_ticker -------------------------------------------------------


async def test_trading_day_ticker_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=TICKER_STATS_NO_BID_ASK)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_trading_day_ticker(md.TradingDayTickerInput(symbol="btcusdt", time_zone="+08:00"))

    assert "Trading day ticker" in result
    assert "weighted avg" in result
    # /ticker/tradingDay never returns bidPrice/askPrice, even at FULL — must not render N/A.
    assert "bid:" not in result
    assert fake.calls == [
        (
            "GET",
            "/api/v3/ticker/tradingDay",
            {"params": {"type": "FULL", "symbol": "BTCUSDT", "timeZone": "+08:00"}, "auth": "none"},
        )
    ]


def test_trading_day_ticker_requires_symbol_or_symbols() -> None:
    with pytest.raises(ValidationError):
        md.TradingDayTickerInput()


def test_trading_day_ticker_rejects_too_many_symbols() -> None:
    with pytest.raises(ValidationError):
        md.TradingDayTickerInput(symbols=[f"SYM{i}USDT" for i in range(101)])


# -- shared error path (one per module, per the wave contract) --------------------------


async def test_error_path_returns_handled_message(monkeypatch: pytest.MonkeyPatch) -> None:
    exc = _status_error(400, {"code": -1121, "msg": "Invalid symbol."})
    fake = _FakeClient(exc=exc)
    _patch_client(monkeypatch, fake)

    result = await md.binance_get_ticker_price(md.TickerPriceInput(symbol="BTCUSDT"))

    assert result.startswith("Error (400)")
    assert "Invalid symbol" in result


# -- live smoke (real network; skipped without BINANCE_API_KEY + secret in .env) --------
# These endpoints are auth="none" and need no credentials to succeed against Binance, but
# the shared conftest gates ALL `live`-marked tests on has_creds like every other module.


@pytest.mark.live
async def test_live_exchange_info() -> None:
    result = await md.binance_get_exchange_info(md.ExchangeInfoInput(symbol="BTCUSDT"))
    assert not result.startswith("Error")
    assert "BTCUSDT" in result


@pytest.mark.live
async def test_live_order_book() -> None:
    result = await md.binance_get_order_book(md.OrderBookInput(symbol="BTCUSDT", limit=5))
    assert not result.startswith("Error")


@pytest.mark.live
async def test_live_klines() -> None:
    result = await md.binance_get_klines(md.KlinesInput(symbol="BTCUSDT", interval=md.KlineInterval.H1, limit=5))
    assert not result.startswith("Error")


@pytest.mark.live
async def test_live_ticker_price() -> None:
    result = await md.binance_get_ticker_price(md.TickerPriceInput(symbol="BTCUSDT"))
    assert not result.startswith("Error")
