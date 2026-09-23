"""Unit tests for the trade-history tools against a fake client.

Everything here is signed USER_DATA except `exchangeInfo`, so the live smoke tests at
the bottom need real credentials (the shared conftest skips them otherwise). The two
`/sapi` calls used by symbol discovery (`getUserAsset`, `dribblet`) do **not** exist on
the spot testnet, so discovery can only ever be smoke-tested against the real account.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from binance_mcp.formatters import ResponseFormat
from binance_mcp.tools import trade_history as th


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _Seq(list[Any]):
    """Marker for a per-path QUEUE of payloads: one entry consumed per call.

    Needed because a `myTrades` payload is itself a list — without the marker there is
    no way to tell "one page" from "a sequence of pages". An `Exception` entry is raised
    instead of returned, which is how the mid-walk failure case is expressed.
    """


class _FakeClient:
    """Records every call (method, path, kwargs); answers from `routes` or `payload`."""

    def __init__(
        self,
        payload: Any = None,
        exc: Exception | None = None,
        routes: dict[str, Any] | None = None,
        used_weight: int | None = None,
    ) -> None:
        self._payload = payload
        self._exc = exc
        self._routes = routes or {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        # The client captures this from X-MBX-USED-WEIGHT-1M after every real call.
        self.last_used_weight_1m: int | None = used_weight

    async def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, path, kwargs))
        if self._exc is not None:
            raise self._exc
        if path in self._routes:
            value = self._routes[path]
            item = value.pop(0) if isinstance(value, _Seq) else value
            if isinstance(item, Exception):
                raise item
            return _FakeResponse(item)
        return _FakeResponse(self._payload)


def _patch_client(monkeypatch: pytest.MonkeyPatch, fake: _FakeClient) -> None:
    monkeypatch.setattr(th, "get_client", lambda: fake)


def _status_error(status: int, body: dict[str, Any]) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.binance.com/api/v3/myTrades")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


@pytest.fixture(autouse=True)
def _clear_symbol_cache() -> Iterator[None]:
    """`exchangeInfo` is cached module-level for the process; isolate every test."""
    th._SYMBOL_CACHE.clear()
    yield
    th._SYMBOL_CACHE.clear()


def _trade(
    trade_id: int,
    *,
    symbol: str = "BTCUSDT",
    is_buyer: bool = True,
    price: str = "100",
    qty: str = "2",
    quote_qty: str = "200",
    commission: str = "0.1",
    commission_asset: str = "BNB",
    is_maker: bool = False,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "id": trade_id,
        "orderId": 900_000 + trade_id,
        "orderListId": -1,
        "price": price,
        "qty": qty,
        "quoteQty": quote_qty,
        "commission": commission,
        "commissionAsset": commission_asset,
        "time": 1_700_000_000_000 + trade_id,
        "isBuyer": is_buyer,
        "isMaker": is_maker,
        "isBestMatch": True,
    }


MY_TRADES_PAYLOAD: list[dict[str, Any]] = [
    _trade(1),
    _trade(2, is_buyer=False, price="105", qty="1", quote_qty="105", commission="0.05", is_maker=True),
]


# -- binance_get_my_trades -----------------------------------------------------------


async def test_my_trades_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=MY_TRADES_PAYLOAD)
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_my_trades(th.MyTradesInput(symbol="btcusdt"))

    assert "# My trades — BTCUSDT" in result
    assert "Showing 2 of 2 fill(s)." in result
    # isBuyer is MY side on myTrades (public trades expose the aggressor via isBuyerMaker).
    assert "| buy | 100 | 2 | 200 | 0.1 BNB | taker |" in result
    assert "| sell | 105 | 1 | 105 | 0.05 BNB | maker |" in result
    assert "- bought: 2 base for 200 quote" in result
    assert "- sold: 1 base for 105 quote" in result
    assert "- fees: 0.15 BNB" in result
    assert fake.calls == [
        ("GET", "/api/v3/myTrades", {"params": {"symbol": "BTCUSDT", "limit": 500}, "auth": "signed"})
    ]


async def test_my_trades_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=MY_TRADES_PAYLOAD)
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_my_trades(th.MyTradesInput(symbol="BTCUSDT", response_format=ResponseFormat.JSON))

    payload = json.loads(result)
    assert payload["symbol"] == "BTCUSDT"
    assert payload["count"] == 2
    assert payload["totals"] == {
        "trades": 2,
        "bought_qty": "2",
        "bought_quote": "200",
        "sold_qty": "1",
        "sold_quote": "105",
        "fees": {"BNB": "0.15"},
    }
    assert payload["trades"][0]["id"] == 1


async def test_my_trades_order_id_and_from_id_combo_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inventory B lists symbol+orderId+fromId as a legal combination."""
    fake = _FakeClient(payload=[])
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_my_trades(th.MyTradesInput(symbol="BTCUSDT", order_id=42, from_id=7, limit=1000))

    assert "_No fills for this symbol with those filters._" in result
    assert fake.calls == [
        (
            "GET",
            "/api/v3/myTrades",
            {"params": {"symbol": "BTCUSDT", "limit": 1000, "orderId": 42, "fromId": 7}, "auth": "signed"},
        )
    ]


async def test_my_trades_accepts_iso_and_digit_string_times(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=[])
    _patch_client(monkeypatch, fake)

    await th.binance_get_my_trades(
        th.MyTradesInput(symbol="BTCUSDT", start_time="2024-01-01T00:00:00Z", end_time="1704110400000")
    )

    assert fake.calls == [
        (
            "GET",
            "/api/v3/myTrades",
            {
                "params": {
                    "symbol": "BTCUSDT",
                    "limit": 500,
                    "startTime": 1_704_067_200_000,
                    "endTime": 1_704_110_400_000,
                },
                "auth": "signed",
            },
        )
    ]


async def test_my_trades_rejects_window_over_24h(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=[])
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_my_trades(
        th.MyTradesInput(symbol="BTCUSDT", start_time="2024-01-01T00:00:00Z", end_time="2024-01-03T00:00:00Z")
    )

    assert result.startswith("Error: myTrades caps start_time..end_time at 24 hours")
    assert "48.0 h" in result
    assert fake.calls == []


async def test_my_trades_rejects_order_id_with_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=[])
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_my_trades(
        th.MyTradesInput(symbol="BTCUSDT", order_id=42, start_time="2024-01-01T00:00:00Z")
    )

    assert result.startswith("Error: `order_id` cannot combine with `start_time`/`end_time`")
    assert fake.calls == []


async def test_my_trades_rejects_from_id_with_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=[])
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_my_trades(
        th.MyTradesInput(symbol="BTCUSDT", from_id=7, end_time="2024-01-01T00:00:00Z")
    )

    assert result.startswith("Error: `from_id` cannot combine with `start_time`/`end_time`")
    assert fake.calls == []


async def test_my_trades_rejects_inverted_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=[])
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_my_trades(
        th.MyTradesInput(symbol="BTCUSDT", start_time="2024-01-02T00:00:00Z", end_time="2024-01-01T00:00:00Z")
    )

    assert result == "Error: `end_time` is before `start_time`."
    assert fake.calls == []


async def test_my_trades_caps_display_and_hints_next_page(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(payload=[_trade(i) for i in range(1, 121)])
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_my_trades(th.MyTradesInput(symbol="BTCUSDT", limit=120))

    assert "Showing 100 of 120 fill(s)." in result
    assert "...20 more not shown — use response_format=json." in result
    assert "continue with `from_id`: **121**" in result


async def test_my_trades_rejects_unparseable_timestamp() -> None:
    with pytest.raises(ValidationError, match="must be epoch ms or ISO-8601"):
        th.MyTradesInput(symbol="BTCUSDT", start_time="yesterday")


async def test_my_trades_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(exc=_status_error(400, {"code": -2015, "msg": "Invalid API-key, IP, or permissions."}))
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_my_trades(th.MyTradesInput(symbol="BTCUSDT"))

    assert result.startswith("Error (400):")
    assert "code -2015" in result
    assert "IP allowlist" in result or "allowlist" in result


# -- binance_discover_traded_symbols -------------------------------------------------

ACCOUNT_PAYLOAD: dict[str, Any] = {
    "balances": [
        {"asset": "BTC", "free": "0.5", "locked": "0"},
        {"asset": "USDT", "free": "120", "locked": "0"},
    ]
}
USER_ASSET_PAYLOAD: list[dict[str, Any]] = [{"asset": "BNB", "free": "1.2", "locked": "0"}]
DRIBBLET_PAYLOAD: dict[str, Any] = {
    "total": 1,
    "userAssetDribblets": [
        {
            "operateTime": 1_700_000_000_000,
            "userAssetDribbletDetails": [{"fromAsset": "SHIB", "amount": "1000000", "transferedAmount": "0.001"}],
        }
    ],
}


def _sym(symbol: str, base: str, quote: str, status: str = "TRADING") -> dict[str, Any]:
    return {"symbol": symbol, "status": status, "baseAsset": base, "quoteAsset": quote}


EXCHANGE_INFO_PAYLOAD: dict[str, Any] = {
    "symbols": [
        _sym("BTCUSDT", "BTC", "USDT"),  # candidate base, whitelisted quote
        _sym("BNBBTC", "BNB", "BTC"),  # candidate base, quote whitelisted (and a candidate)
        _sym("DOGEUSDT", "DOGE", "USDT"),  # only the QUOTE is a candidate -> NOT selected
        _sym("BTCTRY", "BTC", "TRY"),  # candidate base, TRY neither whitelisted nor held
        _sym("SHIBUSDT", "SHIB", "USDT", status="BREAK"),  # delisted: excluded by default
        _sym("ADAETH", "ADA", "ETH"),  # base is not a candidate asset
    ]
}

DISCOVERY_ROUTES: dict[str, Any] = {
    "/api/v3/account": ACCOUNT_PAYLOAD,
    "/sapi/v3/asset/getUserAsset": USER_ASSET_PAYLOAD,
    "/sapi/v1/asset/dribblet": DRIBBLET_PAYLOAD,
    "/api/v3/exchangeInfo": EXCHANGE_INFO_PAYLOAD,
}

DISCOVERY_CALLS: list[tuple[str, str, dict[str, Any]]] = [
    ("GET", "/api/v3/account", {"params": {"omitZeroBalances": True}, "auth": "signed"}),
    ("POST", "/sapi/v3/asset/getUserAsset", {"params": {}, "auth": "signed"}),
    ("GET", "/sapi/v1/asset/dribblet", {"params": {}, "auth": "signed"}),
    ("GET", "/api/v3/exchangeInfo", {"params": {}, "auth": "none"}),
]


async def test_discover_traded_symbols_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes=dict(DISCOVERY_ROUTES))
    _patch_client(monkeypatch, fake)

    result = await th.binance_discover_traded_symbols(th.DiscoverTradedSymbolsInput())

    assert "**4** candidate asset(s) → **2** symbol(s)" in result
    assert "- assets: BNB, BTC, SHIB, USDT" in result
    assert "BNBBTC, BTCUSDT" in result
    assert "DOGEUSDT" not in result  # only the quote is a candidate — the base decides
    assert "BTCTRY" not in result  # TRY is neither whitelisted nor a candidate
    assert "SHIBUSDT" not in result  # BREAK, and include_break is False
    assert "ADAETH" not in result  # base is not a candidate
    assert "**40** IP weight" in result
    assert fake.calls == DISCOVERY_CALLS


async def test_discover_traded_symbols_requires_the_base_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard for the predicate: base ∈ candidates AND quote ∈ (whitelist ∪ candidates).

    Matching on the quote side alone selects every pair quoted in a stablecoin the account
    merely holds — on the real exchangeInfo, USDT is 493 TRADING pairs (9,860 weight).
    """
    routes = dict(DISCOVERY_ROUTES)
    routes["/api/v3/account"] = {"balances": [{"asset": a, "free": "1", "locked": "0"} for a in ("BTC", "ETH", "USDT")]}
    routes["/sapi/v3/asset/getUserAsset"] = []
    routes["/sapi/v1/asset/dribblet"] = {"total": 0, "userAssetDribblets": []}
    routes["/api/v3/exchangeInfo"] = {
        "symbols": [
            _sym("BTCUSDT", "BTC", "USDT"),
            _sym("ETHUSDT", "ETH", "USDT"),
            _sym("ETHBTC", "ETH", "BTC"),
            _sym("SOLUSDT", "SOL", "USDT"),  # quote-only match — excluded
            _sym("XRPBTC", "XRP", "BTC"),  # quote-only match — excluded
        ]
    }
    fake = _FakeClient(routes=routes)
    _patch_client(monkeypatch, fake)

    payload = json.loads(
        await th.binance_discover_traded_symbols(th.DiscoverTradedSymbolsInput(response_format=ResponseFormat.JSON))
    )

    assert payload["symbols"] == ["BTCUSDT", "ETHBTC", "ETHUSDT"]
    assert payload["estimated_weight"] == 60


async def test_discover_traded_symbols_accepts_a_held_quote_outside_the_whitelist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate quote asset counts even when it is not in `quote_assets` — the rule
    that hands a {BTC, TRY, USDT} holder their BTCTRY pair back."""
    fake = _FakeClient(routes=dict(DISCOVERY_ROUTES))
    _patch_client(monkeypatch, fake)

    payload = json.loads(
        await th.binance_discover_traded_symbols(
            th.DiscoverTradedSymbolsInput(extra_assets=["TRY"], response_format=ResponseFormat.JSON)
        )
    )

    assert payload["symbols"] == ["BNBBTC", "BTCTRY", "BTCUSDT"]
    assert "TRY" not in payload["quote_assets"]


async def test_discover_traded_symbols_include_break(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes=dict(DISCOVERY_ROUTES))
    _patch_client(monkeypatch, fake)

    result = await th.binance_discover_traded_symbols(
        th.DiscoverTradedSymbolsInput(include_break=True, response_format=ResponseFormat.JSON)
    )

    payload = json.loads(result)
    assert payload["symbols"] == ["BNBBTC", "BTCUSDT", "SHIBUSDT"]
    assert payload["estimated_weight"] == 60
    assert payload["assets"] == ["BNB", "BTC", "SHIB", "USDT"]
    assert payload["asset_sources"] == {
        "spot balances": 2,
        "user assets": 1,
        "dust conversions": 1,
        "extra_assets": 0,
    }


async def test_discover_traded_symbols_extra_assets_and_quote_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes=dict(DISCOVERY_ROUTES))
    _patch_client(monkeypatch, fake)

    result = await th.binance_discover_traded_symbols(
        th.DiscoverTradedSymbolsInput(extra_assets=["ada"], quote_assets=["eth"], response_format=ResponseFormat.JSON)
    )

    payload = json.loads(result)
    # `extra_assets` makes ADA a candidate base, so ADAETH appears via the whitelist.
    # BTCUSDT and BNBBTC survive a whitelist of just ETH because their quote assets are
    # themselves held: narrowing `quote_assets` prunes pairs quoted in assets the account
    # does NOT hold, not the ones it does.
    assert payload["symbols"] == ["ADAETH", "BNBBTC", "BTCUSDT"]
    assert "ADA" in payload["assets"]
    assert payload["quote_assets"] == ["ETH"]


async def test_discover_traded_symbols_caches_exchange_info(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes=dict(DISCOVERY_ROUTES))
    _patch_client(monkeypatch, fake)

    await th.binance_discover_traded_symbols(th.DiscoverTradedSymbolsInput())
    await th.binance_discover_traded_symbols(th.DiscoverTradedSymbolsInput())

    exchange_info_calls = [call for call in fake.calls if call[1] == "/api/v3/exchangeInfo"]
    assert len(exchange_info_calls) == 1
    assert len(fake.calls) == 7  # 3 signed reads twice + one cached exchangeInfo


async def test_discover_traded_symbols_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(exc=_status_error(404, {"code": -1121, "msg": "Not found."}))
    _patch_client(monkeypatch, fake)

    result = await th.binance_discover_traded_symbols(th.DiscoverTradedSymbolsInput())

    assert result.startswith("Error (404):")
    assert "testnet has no /sapi endpoints" in result


async def test_discover_rejects_invalid_asset() -> None:
    with pytest.raises(ValidationError, match="Invalid asset"):
        th.DiscoverTradedSymbolsInput(extra_assets=["not-an-asset"])


# -- binance_get_all_my_trades -------------------------------------------------------

FULL_PAGE = [_trade(i) for i in range(1, 1001)]
SHORT_PAGE = [_trade(i) for i in range(1001, 1301)]


def _my_trades_call(symbol: str, from_id: int) -> tuple[str, str, dict[str, Any]]:
    return (
        "GET",
        "/api/v3/myTrades",
        {"params": {"symbol": symbol, "fromId": from_id, "limit": 1000}, "auth": "signed"},
    )


async def test_all_my_trades_walks_multiple_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/myTrades": _Seq([FULL_PAGE, SHORT_PAGE])})
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_all_my_trades(th.AllMyTradesInput(symbols=["btcusdt"]))

    # A full page continues at last id + 1; the short page ends the symbol.
    assert fake.calls == [_my_trades_call("BTCUSDT", 0), _my_trades_call("BTCUSDT", 1001)]
    assert "Walked **1** of **1** symbol(s) in 2 call(s) (~40 IP weight)" in result
    assert "Collected **1,300** fill(s)." in result
    assert "...1100 more not shown" in result
    assert "### BTCUSDT — 1300 fill(s)" in result
    assert '"BTCUSDT": 1300' in result
    assert "Stopped early" not in result


async def test_all_my_trades_stops_on_weight_budget_mid_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/myTrades": _Seq([FULL_PAGE, SHORT_PAGE])})
    _patch_client(monkeypatch, fake)

    # 20 weight buys exactly one page: the second is refused BEFORE it is requested.
    result = await th.binance_get_all_my_trades(th.AllMyTradesInput(symbols=["BTCUSDT"], max_weight=20))

    assert fake.calls == [_my_trades_call("BTCUSDT", 0)]
    assert "Stopped early: weight budget exhausted (20 of max_weight 20 spent over 1 call(s))" in result
    assert "Collected **1,000** fill(s)." in result
    # The cursor points at the last id actually fetched, not at the symbol's true end.
    assert '"BTCUSDT": 1000' in result
    assert "Resume with the `cursor` block" in result


async def test_all_my_trades_resume_from_cursor_has_no_gap_and_no_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chained run: the cursor emitted by a budget stop is fed straight back in."""
    first_fake = _FakeClient(routes={"/api/v3/myTrades": _Seq([FULL_PAGE, SHORT_PAGE])})
    _patch_client(monkeypatch, first_fake)
    first = json.loads(
        await th.binance_get_all_my_trades(
            th.AllMyTradesInput(symbols=["BTCUSDT"], max_weight=20, response_format=ResponseFormat.JSON)
        )
    )
    assert first["stopped_early"]
    assert first["cursor"] == {"BTCUSDT": 1000}

    second_fake = _FakeClient(routes={"/api/v3/myTrades": _Seq([SHORT_PAGE])})
    _patch_client(monkeypatch, second_fake)
    second = json.loads(
        await th.binance_get_all_my_trades(
            th.AllMyTradesInput(symbols=["BTCUSDT"], cursor=first["cursor"], response_format=ResponseFormat.JSON)
        )
    )

    assert second_fake.calls == [_my_trades_call("BTCUSDT", 1001)]
    assert second["stopped_early"] is None
    assert second["cursor"] == {"BTCUSDT": 1300}

    first_ids = [t["id"] for t in first["trades"]]
    second_ids = [t["id"] for t in second["trades"]]
    union = first_ids + second_ids
    assert len(union) == len(set(union)) == 1300  # no duplicate
    assert sorted(union) == list(range(1, 1301))  # no gap


async def test_all_my_trades_keeps_untouched_cursor_entries_and_omits_unknown_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient(routes={"/api/v3/myTrades": _Seq([FULL_PAGE])})
    _patch_client(monkeypatch, fake)

    result = json.loads(
        await th.binance_get_all_my_trades(
            th.AllMyTradesInput(
                symbols=["AAAUSDT", "BBBUSDT", "CCCUSDT"],
                cursor={"CCCUSDT": 77},
                max_weight=20,
                response_format=ResponseFormat.JSON,
            )
        )
    )

    assert fake.calls == [_my_trades_call("AAAUSDT", 0)]
    # AAAUSDT advanced; CCCUSDT keeps the position it came in with; BBBUSDT was never
    # reached and has no known position, so it stays absent.
    assert result["cursor"] == {"CCCUSDT": 77, "AAAUSDT": 1000}
    assert result["symbols_done"] == []


async def test_all_my_trades_stops_on_reported_weight_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/myTrades": _Seq([FULL_PAGE])}, used_weight=5500)
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_all_my_trades(th.AllMyTradesInput(symbols=["BTCUSDT"]))

    assert fake.calls == []
    assert "Binance reports 5500 used IP weight in the last minute, over weight_ceiling 5000" in result


async def test_all_my_trades_error_mid_walk_returns_partial_rows_and_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient(
        routes={"/api/v3/myTrades": _Seq([FULL_PAGE, _status_error(418, {"code": -1003, "msg": "Banned."})])}
    )
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_all_my_trades(th.AllMyTradesInput(symbols=["BTCUSDT"]))

    assert fake.calls == [_my_trades_call("BTCUSDT", 0), _my_trades_call("BTCUSDT", 1001)]
    assert "Stopped on an error after 2 call(s)" in result
    assert "Error (418)" in result
    assert "Collected **1,000** fill(s)." in result
    assert '"BTCUSDT": 1000' in result


async def test_all_my_trades_discovers_symbols_when_not_given(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = dict(DISCOVERY_ROUTES)
    routes["/api/v3/myTrades"] = _Seq([[_trade(1, symbol="BNBBTC")], [_trade(2)]])
    fake = _FakeClient(routes=routes)
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_all_my_trades(th.AllMyTradesInput(response_format=ResponseFormat.JSON))

    payload = json.loads(result)
    assert payload["discovered"] is True
    assert payload["symbols"] == ["BNBBTC", "BTCUSDT"]
    assert fake.calls == [
        *DISCOVERY_CALLS,
        _my_trades_call("BNBBTC", 0),
        _my_trades_call("BTCUSDT", 0),
    ]
    assert payload["calls"] == 2
    assert payload["estimated_weight"] == 40
    assert payload["count"] == 2
    assert payload["cursor"] == {"BNBBTC": 1, "BTCUSDT": 2}
    assert payload["totals"]["BTCUSDT"] == {
        "trades": 1,
        "bought_qty": "2",
        "bought_quote": "200",
        "sold_qty": "0",
        "sold_quote": "0",
        "fees": {"BNB": "0.1"},
    }


async def test_all_my_trades_reports_when_discovery_finds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = dict(DISCOVERY_ROUTES)
    routes["/api/v3/account"] = {"balances": []}
    routes["/sapi/v3/asset/getUserAsset"] = []
    routes["/sapi/v1/asset/dribblet"] = {"total": 0, "userAssetDribblets": []}
    fake = _FakeClient(routes=routes)
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_all_my_trades(th.AllMyTradesInput())

    assert result.startswith("Error: no symbols to walk.")
    assert "extra_assets" in result


async def test_all_my_trades_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(exc=_status_error(400, {"code": -1121, "msg": "Invalid symbol."}))
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_all_my_trades(th.AllMyTradesInput(symbols=["NOPEUSDT"]))

    # The walk swallows the error into a partial result rather than losing the cursor.
    assert "Stopped on an error after 1 call(s)" in result
    assert "code -1121" in result
    assert "_No fills found for these symbols._" in result


async def test_all_my_trades_surfaces_a_credential_error_plainly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A RuntimeError with nothing collected is NOT dressed up as a partial walk.

    Missing credentials, the trading kill-switch and a forbidden path all arrive as
    RuntimeErrors; an empty walk report plus an empty cursor would bury the one thing the
    caller has to fix.
    """
    fake = _FakeClient(exc=RuntimeError("No Binance API key configured. Set BINANCE_API_KEY."))
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_all_my_trades(th.AllMyTradesInput(symbols=["BTCUSDT"]))

    assert result == "Error: No Binance API key configured. Set BINANCE_API_KEY."
    assert "All my trades" not in result


async def test_all_my_trades_keeps_partial_rows_when_a_runtime_error_lands_mid_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-raise is gated on having collected nothing — rows already fetched still win."""
    fake = _FakeClient(routes={"/api/v3/myTrades": _Seq([FULL_PAGE, RuntimeError("kill-switch")])})
    _patch_client(monkeypatch, fake)

    result = await th.binance_get_all_my_trades(th.AllMyTradesInput(symbols=["BTCUSDT"]))

    assert "Stopped on an error after 2 call(s)" in result
    assert "Error: kill-switch" in result
    assert '"BTCUSDT": 1000' in result


async def test_all_my_trades_rejects_negative_cursor() -> None:
    with pytest.raises(ValidationError, match="a trade id cannot be negative"):
        th.AllMyTradesInput(symbols=["BTCUSDT"], cursor={"BTCUSDT": -1})


# -- live smoke ----------------------------------------------------------------------


@pytest.mark.live
async def test_live_smoke_my_trades() -> None:
    """One signed myTrades page for a single symbol against the real account.

    Requires BINANCE_API_KEY + secret/PEM (see conftest's live gating). Weight 20; an
    account that never traded BTCUSDT simply returns an empty table, which still proves
    the signature and the permissions.
    """
    result = await th.binance_get_my_trades(th.MyTradesInput(symbol="BTCUSDT", limit=10))

    assert isinstance(result, str)
    assert not result.startswith("Error")
    assert "# My trades — BTCUSDT" in result


@pytest.mark.live
async def test_live_smoke_discover_traded_symbols() -> None:
    """Symbol discovery against the real account (~46 IP weight, no walk).

    Two of its three signed reads are `/sapi` endpoints that do NOT exist on the spot
    testnet, so this can only run against the real account — never under BINANCE_TESTNET=1.
    """
    result = await th.binance_discover_traded_symbols(th.DiscoverTradedSymbolsInput())

    assert isinstance(result, str)
    assert not result.startswith("Error")
    assert "# Traded-symbol candidates" in result


@pytest.mark.live
async def test_live_smoke_all_my_trades_bounded() -> None:
    """The walk against the real account, bounded to ONE page of ONE symbol.

    `max_weight=20` buys exactly one myTrades call, so this costs 20 IP weight of the
    6000/min budget however much history the account has — no discovery, no fan-out. An
    account with more than 1000 BTCUSDT fills stops on the budget and says so, which is
    still a pass: the point is that the walk, the cursor and the rendering work live.
    """
    result = await th.binance_get_all_my_trades(th.AllMyTradesInput(symbols=["BTCUSDT"], max_weight=20))

    assert isinstance(result, str)
    assert not result.startswith("Error")
    assert "# All my trades" in result
    assert "## Cursor" in result
