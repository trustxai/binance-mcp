"""Unit tests for the spot account tools against a fake client.

All five tools call `/api/v3` (not `/sapi`), so — unlike Simple Earn — the spot
testnet *does* serve these endpoints; the `live`-marked smoke tests below still
use a real mainnet account (per house style) and are auto-skipped without
credentials.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from binance_mcp.tools.spot_account import (
    ResponseFormat,
    _AllocationsInput,
    _CommissionRatesInput,
    _OrderRateLimitsInput,
    _PreventedMatchesInput,
    _SpotAccountInput,
    _to_ms,
    binance_get_allocations,
    binance_get_commission_rates,
    binance_get_order_rate_limits,
    binance_get_prevented_matches,
    binance_get_spot_account,
)


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Routes by path; records every call (method, path, kwargs)."""

    def __init__(self, routes: dict[str, Any] | None = None) -> None:
        self._routes = routes or {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, path, kwargs))
        payload = self._routes.get(path)
        if isinstance(payload, Exception):
            raise payload
        return _FakeResponse(payload if payload is not None else {})


def _status_error(path: str, status: int, body: dict[str, Any]) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", f"https://api.binance.com{path}")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


ACCOUNT_DATA = {
    "makerCommission": 15,
    "takerCommission": 15,
    "commissionRates": {
        "maker": "0.00150000",
        "taker": "0.00150000",
        "buyer": "0.00000000",
        "seller": "0.00000000",
    },
    "canTrade": True,
    "canWithdraw": True,
    "canDeposit": True,
    "brokered": False,
    "requireSelfTradePrevention": False,
    "preventSor": False,
    "updateTime": 1698000000000,
    "accountType": "SPOT",
    "balances": [
        {"asset": "BTC", "free": "1.00000000", "locked": "0.00000000"},
        {"asset": "USDT", "free": "0.00000000", "locked": "0.00000000"},
        {"asset": "ETH", "free": "0.00000000", "locked": "0.50000000"},
    ],
    "permissions": ["SPOT"],
    "uid": 12345,
}

COMMISSION_DATA = {
    "symbol": "BTCUSDT",
    "standardCommission": {"maker": "0.00100000", "taker": "0.00100000", "buyer": "0.00000000", "seller": "0.00000000"},
    "specialCommission": None,
    "taxCommission": {"maker": "0.00000000", "taker": "0.00000000", "buyer": "0.00000000", "seller": "0.00000000"},
    "discount": {"enabledForAccount": True, "enabledForSymbol": True, "discountAsset": "BNB", "discount": "0.25000000"},
}

RATE_LIMITS = [
    {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 50, "count": 0},
    {"rateLimitType": "ORDERS", "interval": "DAY", "intervalNum": 1, "limit": 160000, "count": 1},
]

PREVENTED_MATCH_ROW = {
    "symbol": "BTCUSDT",
    "preventedMatchId": 1,
    "takerOrderId": 5,
    "makerSymbol": "BTCUSDT",
    "makerOrderId": 3,
    "tradeGroupId": 1,
    "selfTradePreventionMode": "EXPIRE_MAKER",
    "price": "1.100000",
    "makerPreventedQuantity": "1.300000",
    "transactTime": 1669101687094,
}

ALLOCATION_ROW = {
    "symbol": "BTCUSDT",
    "allocationId": 0,
    "allocationType": "SOR",
    "orderId": 1,
    "orderListId": -1,
    "price": "1.00000000",
    "qty": "5.00000000",
    "quoteQty": "5.00000000",
    "commission": "0.00000000",
    "commissionAsset": "BTC",
    "time": 1687506878118,
    "isBuyer": True,
    "isMaker": False,
    "isAllocator": False,
}


# -- _to_ms -------------------------------------------------------------------


def test_to_ms_none_passthrough() -> None:
    assert _to_ms(None) is None


def test_to_ms_int_passthrough() -> None:
    assert _to_ms(1698000000000) == 1698000000000


def test_to_ms_numeric_string_with_12_plus_digits() -> None:
    assert _to_ms("1698000000000") == 1698000000000


def test_to_ms_iso8601_string() -> None:
    assert _to_ms("2024-01-01T00:00:00Z") == 1704067200000


def test_to_ms_bad_input_returns_readable_error() -> None:
    result = _to_ms("not-a-timestamp")
    assert isinstance(result, str)
    assert result.startswith("Error: could not parse timestamp")


# -- binance_get_spot_account ---------------------------------------------------


async def test_spot_account_happy_path_hides_zero_balances(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/account": ACCOUNT_DATA})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_spot_account(_SpotAccountInput())

    assert "**BTC**: free 1, locked 0" in result
    assert "**ETH**: free 0, locked 0.5" in result
    assert "**USDT**" not in result
    assert "1 zero-balance asset(s) hidden" in result
    assert "can trade / withdraw / deposit**: True / True / True" in result
    assert "maker 0.0015" in result
    call = fake.calls[0]
    assert call[0] == "GET"
    assert call[1] == "/api/v3/account"
    assert call[2] == {"auth": "signed"}


async def test_spot_account_omit_zero_balances_false_shows_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/account": ACCOUNT_DATA})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_spot_account(_SpotAccountInput(omit_zero_balances=False))

    assert "**BTC**" in result
    assert "**USDT**: free 0, locked 0" in result
    assert "**ETH**" in result
    assert "hidden" not in result
    assert "3 shown" in result


async def test_spot_account_asset_filter_overrides_zero_hiding(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/account": ACCOUNT_DATA})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_spot_account(_SpotAccountInput(asset="usdt"))

    assert "**USDT**: free 0, locked 0" in result
    assert "**BTC**" not in result
    assert "hidden" not in result
    assert "1 shown" in result


async def test_spot_account_truncates_display_and_notes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    balances = [{"asset": f"A{i}", "free": "1.00000000", "locked": "0.00000000"} for i in range(60)]
    fake = _FakeClient(routes={"/api/v3/account": {**ACCOUNT_DATA, "balances": balances}})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_spot_account(_SpotAccountInput())

    assert "display truncated: 60 balances matched" in result
    assert "showing the first 50" in result
    assert "**A49**" in result
    assert "**A50**" not in result


async def test_spot_account_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/account": ACCOUNT_DATA})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_spot_account(_SpotAccountInput(response_format=ResponseFormat.JSON))

    assert '"asset": "BTC"' in result
    assert '"hiddenZeroBalances": 1' in result
    assert '"asset": "USDT"' not in result


async def test_spot_account_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/account": _status_error(
                "/api/v3/account", 401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."}
            )
        }
    )
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_spot_account(_SpotAccountInput())

    assert result.startswith("Error (401)")
    assert "allowlist" in result


# -- binance_get_commission_rates -----------------------------------------------


async def test_commission_rates_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/account/commission": COMMISSION_DATA})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_commission_rates(_CommissionRatesInput(symbol="btcusdt"))

    assert "# Commission Rates — BTCUSDT" in result
    assert "standard commission**: maker 0.001, taker 0.001" in result
    assert "special commission**: N/A" in result
    assert "discount**: account True, symbol True, asset BNB, rate 0.25" in result
    call = fake.calls[0]
    assert call[0] == "GET"
    assert call[1] == "/api/v3/account/commission"
    assert call[2] == {"params": {"symbol": "BTCUSDT"}, "auth": "signed"}


async def test_commission_rates_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/account/commission": COMMISSION_DATA})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_commission_rates(
        _CommissionRatesInput(symbol="BTCUSDT", response_format=ResponseFormat.JSON)
    )

    assert '"symbol": "BTCUSDT"' in result


async def test_commission_rates_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/account/commission": _status_error(
                "/api/v3/account/commission", 400, {"code": -1121, "msg": "Invalid symbol."}
            )
        }
    )
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_commission_rates(_CommissionRatesInput(symbol="BTCUSDT"))

    assert result.startswith("Error (400)")


# -- binance_get_order_rate_limits -----------------------------------------------


async def test_order_rate_limits_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/rateLimit/order": RATE_LIMITS})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_order_rate_limits(_OrderRateLimitsInput())

    assert "**ORDERS** per 10 SECOND: 0 / 50 used" in result
    assert "**ORDERS** per 1 DAY: 1 / 160000 used" in result
    call = fake.calls[0]
    assert call[0] == "GET"
    assert call[1] == "/api/v3/rateLimit/order"
    assert call[2] == {"auth": "signed"}


async def test_order_rate_limits_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/rateLimit/order": []})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_order_rate_limits(_OrderRateLimitsInput())

    assert "_No rate limit data returned._" in result


async def test_order_rate_limits_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/rateLimit/order": RATE_LIMITS})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_order_rate_limits(_OrderRateLimitsInput(response_format=ResponseFormat.JSON))

    assert '"rateLimitType": "ORDERS"' in result


async def test_order_rate_limits_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/rateLimit/order": _status_error(
                "/api/v3/rateLimit/order", 401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions."}
            )
        }
    )
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_order_rate_limits(_OrderRateLimitsInput())

    assert result.startswith("Error (401)")


# -- binance_get_prevented_matches -----------------------------------------------


def test_prevented_matches_requires_one_id() -> None:
    with pytest.raises(ValidationError):
        _PreventedMatchesInput(symbol="BTCUSDT")


def test_prevented_matches_rejects_both_ids() -> None:
    with pytest.raises(ValidationError):
        _PreventedMatchesInput(symbol="BTCUSDT", prevented_match_id=1, order_id=2)


def test_prevented_matches_from_id_requires_order_id() -> None:
    with pytest.raises(ValidationError):
        _PreventedMatchesInput(symbol="BTCUSDT", prevented_match_id=1, from_prevented_match_id=5)


async def test_prevented_matches_by_prevented_match_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/myPreventedMatches": [PREVENTED_MATCH_ROW]})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_prevented_matches(_PreventedMatchesInput(symbol="btcusdt", prevented_match_id=1))

    assert "match `1`" in result
    assert "taker order `5`" in result
    assert "maker order `3`" in result
    assert "price 1.1" in result
    assert "maker qty prevented 1.3" in result
    assert "STP mode EXPIRE_MAKER" in result
    call = fake.calls[0]
    assert call[0] == "GET"
    assert call[1] == "/api/v3/myPreventedMatches"
    assert call[2] == {"params": {"symbol": "BTCUSDT", "preventedMatchId": 1}, "auth": "signed"}


async def test_prevented_matches_by_order_id_with_limit_and_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/myPreventedMatches": []})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_prevented_matches(
        _PreventedMatchesInput(symbol="BTCUSDT", order_id=42, from_prevented_match_id=5, limit=10)
    )

    assert "_No prevented matches found._" in result
    call = fake.calls[0]
    assert call[2] == {
        "params": {"symbol": "BTCUSDT", "orderId": 42, "fromPreventedMatchId": 5, "limit": 10},
        "auth": "signed",
    }


async def test_prevented_matches_truncates_display_and_notes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{**PREVENTED_MATCH_ROW, "preventedMatchId": i} for i in range(60)]
    fake = _FakeClient(routes={"/api/v3/myPreventedMatches": rows})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_prevented_matches(_PreventedMatchesInput(symbol="BTCUSDT", order_id=1, limit=1000))

    assert "display truncated: 60 rows returned" in result
    assert "showing the first 50" in result
    assert "match `49`" in result
    assert "match `50`" not in result


async def test_prevented_matches_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/myPreventedMatches": [PREVENTED_MATCH_ROW]})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_prevented_matches(
        _PreventedMatchesInput(symbol="BTCUSDT", prevented_match_id=1, response_format=ResponseFormat.JSON)
    )

    assert '"preventedMatchId": 1' in result


async def test_prevented_matches_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/myPreventedMatches": _status_error(
                "/api/v3/myPreventedMatches", 400, {"code": -1121, "msg": "Invalid symbol."}
            )
        }
    )
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_prevented_matches(_PreventedMatchesInput(symbol="BTCUSDT", prevented_match_id=1))

    assert result.startswith("Error (400)")


# -- binance_get_allocations ------------------------------------------------------


async def test_allocations_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/myAllocations": [ALLOCATION_ROW]})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_allocations(_AllocationsInput(symbol="btcusdt"))

    assert "alloc `0`" in result
    assert "order `1`" in result
    assert "buy qty 5 @ 1 = 5" in result
    assert "commission 0 BTC" in result
    assert "maker False, allocator False" in result
    call = fake.calls[0]
    assert call[0] == "GET"
    assert call[1] == "/api/v3/myAllocations"
    assert call[2] == {"params": {"symbol": "BTCUSDT", "limit": 500}, "auth": "signed"}


async def test_allocations_window_within_24h_converts_iso_to_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/myAllocations": []})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    await binance_get_allocations(
        _AllocationsInput(symbol="BTCUSDT", start_time="2024-01-01T00:00:00Z", end_time="2024-01-01T12:00:00Z")
    )

    call = fake.calls[0]
    assert call[2]["params"]["startTime"] == 1704067200000
    assert call[2]["params"]["endTime"] == 1704110400000


async def test_allocations_window_over_24h_rejected_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/myAllocations": []})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_allocations(
        _AllocationsInput(symbol="BTCUSDT", start_time="2024-01-01T00:00:00Z", end_time="2024-01-02T00:00:01Z")
    )

    assert result.startswith("Error: startTime..endTime spans more than 24 h")
    assert fake.calls == []


async def test_allocations_end_before_start_rejected_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/myAllocations": []})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_allocations(
        _AllocationsInput(symbol="BTCUSDT", start_time="2024-01-02T00:00:00Z", end_time="2024-01-01T00:00:00Z")
    )

    assert result == "Error: end_time is before start_time."
    assert fake.calls == []


async def test_allocations_bad_timestamp_rejected_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/myAllocations": []})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_allocations(_AllocationsInput(symbol="BTCUSDT", start_time="not-a-timestamp"))

    assert result.startswith("Error: could not parse timestamp")
    assert fake.calls == []


async def test_allocations_optional_filters_dropped_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/myAllocations": []})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    await binance_get_allocations(_AllocationsInput(symbol="BTCUSDT", from_allocation_id=10, order_id=99, limit=100))

    call = fake.calls[0]
    assert call[2]["params"] == {"symbol": "BTCUSDT", "fromAllocationId": 10, "orderId": 99, "limit": 100}
    assert "startTime" not in call[2]["params"]
    assert "endTime" not in call[2]["params"]


async def test_allocations_truncates_display_and_notes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{**ALLOCATION_ROW, "allocationId": i} for i in range(60)]
    fake = _FakeClient(routes={"/api/v3/myAllocations": rows})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_allocations(_AllocationsInput(symbol="BTCUSDT", limit=1000))

    assert "display truncated: 60 rows returned" in result
    assert "showing the first 50" in result
    assert "alloc `49`" in result
    assert "alloc `50`" not in result


async def test_allocations_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/myAllocations": [ALLOCATION_ROW]})
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_allocations(_AllocationsInput(symbol="BTCUSDT", response_format=ResponseFormat.JSON))

    assert '"allocationId": 0' in result


async def test_allocations_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/myAllocations": _status_error(
                "/api/v3/myAllocations", 400, {"code": -1121, "msg": "Invalid symbol."}
            )
        }
    )
    monkeypatch.setattr("binance_mcp.tools.spot_account.get_client", lambda: fake)

    result = await binance_get_allocations(_AllocationsInput(symbol="BTCUSDT"))

    assert result.startswith("Error (400)")


# -- live smoke (real mainnet account, read-only; auto-skipped without creds) --


@pytest.mark.live
async def test_spot_account_live_smoke() -> None:
    result = await binance_get_spot_account(_SpotAccountInput())
    assert not result.startswith("Error")


@pytest.mark.live
async def test_commission_rates_live_smoke() -> None:
    result = await binance_get_commission_rates(_CommissionRatesInput(symbol="BTCUSDT"))
    assert not result.startswith("Error")


@pytest.mark.live
async def test_order_rate_limits_live_smoke() -> None:
    result = await binance_get_order_rate_limits(_OrderRateLimitsInput())
    assert not result.startswith("Error")


@pytest.mark.live
async def test_prevented_matches_live_smoke() -> None:
    result = await binance_get_prevented_matches(_PreventedMatchesInput(symbol="BTCUSDT", prevented_match_id=1))
    assert not result.startswith("Error")


@pytest.mark.live
async def test_allocations_live_smoke() -> None:
    result = await binance_get_allocations(_AllocationsInput(symbol="BTCUSDT"))
    assert not result.startswith("Error")
