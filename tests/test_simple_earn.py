"""Unit tests for the Simple Earn read-only tools against a fake client.

Simple Earn (`/sapi/v1/simple-earn/*`) does not exist on the spot testnet — the
testnet only serves `/api` endpoints (02-endpoint-inventory.md "Testnet") — so
there is no way to smoke-test this module against the testnet; the `live`-marked
tests below (real mainnet account, read-only, auto-skipped without credentials)
are the only live coverage path for these three tools.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from binance_mcp.tools.simple_earn import (
    ResponseFormat,
    _EarnAccountInput,
    _EarnFlexiblePositionsInput,
    _EarnLockedPositionsInput,
    binance_get_earn_account,
    binance_get_earn_flexible_positions,
    binance_get_earn_locked_positions,
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


FLEXIBLE_ROW = {
    "totalAmount": "75.46000000",
    "tierAnnualPercentageRate": {"0-5BTC": 0.05, "5-10BTC": 0.03},
    "latestAnnualPercentageRate": "0.05000000",
    "yesterdayAirdropPercentageRate": "0.05000000",
    "asset": "BTC",
    "airDropAsset": "BETH",
    "canRedeem": True,
    "collateralAmount": "0.75460000",
    "productId": "BTC001",
    "yesterdayRealTimeRewards": "0.00075460",
    "cumulativeBonusRewards": "0.00075460",
    "cumulativeRealTimeRewards": "0.00075460",
    "cumulativeTotalRewards": "0.00150920",
    "autoSubscribe": True,
}

LOCKED_ROW = {
    "positionId": "123123",
    "projectId": "Axs*90",
    "asset": "AXS",
    "amount": "122.09202928",
    "purchaseTime": "1646182276000",
    "duration": "90",
    "accrualDays": "1",
    "rewardAsset": "AXS",
    "APY": "0.023",
    "isRenewable": True,
    "isAutoRenew": True,
    "redeemDate": "1732182276000",
}

ACCOUNT_SUMMARY = {
    "totalAmountInBTC": "0.01067982",
    "totalAmountInUSDT": "77.13289230",
    "totalFlexibleAmountInBTC": "0.00000000",
    "totalFlexibleAmountInUSDT": "0.00000000",
    "totalLockedInBTC": "0.01067982",
    "totalLockedInUSDT": "77.13289230",
}


# -- flexible positions -------------------------------------------------------


async def test_flexible_positions_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/simple-earn/flexible/position": {"rows": [FLEXIBLE_ROW], "total": 1}})
    monkeypatch.setattr("binance_mcp.tools.simple_earn.get_client", lambda: fake)

    result = await binance_get_earn_flexible_positions(_EarnFlexiblePositionsInput(asset="btc"))

    assert "**BTC**" in result
    assert "product `BTC001`" in result
    assert "5%" in result  # latest APR
    assert "0-5BTC 5%" in result  # tiered APR
    assert "3%" in result  # second tier
    assert "redeemable: True" in result
    assert "airdrop asset: BETH" in result
    call = fake.calls[0]
    assert call[1] == "/sapi/v1/simple-earn/flexible/position"
    assert call[2] == {"params": {"asset": "BTC", "productId": None, "current": 1, "size": 10}, "auth": "signed"}


async def test_flexible_positions_pagination_offset_maps_to_current(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/simple-earn/flexible/position": {"rows": [], "total": 0}})
    monkeypatch.setattr("binance_mcp.tools.simple_earn.get_client", lambda: fake)

    await binance_get_earn_flexible_positions(_EarnFlexiblePositionsInput(limit=20, offset=40))

    call = fake.calls[0]
    assert call[2]["params"]["current"] == 3  # 40 // 20 + 1
    assert call[2]["params"]["size"] == 20
    assert call[2]["params"]["productId"] is None


async def test_flexible_positions_product_id_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/simple-earn/flexible/position": {"rows": [], "total": 0}})
    monkeypatch.setattr("binance_mcp.tools.simple_earn.get_client", lambda: fake)

    result = await binance_get_earn_flexible_positions(_EarnFlexiblePositionsInput(product_id="BTC001"))

    assert "_No items._" in result
    call = fake.calls[0]
    assert call[2]["params"]["productId"] == "BTC001"
    assert call[2]["params"]["asset"] is None


async def test_flexible_positions_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/simple-earn/flexible/position": {"rows": [FLEXIBLE_ROW], "total": 1}})
    monkeypatch.setattr("binance_mcp.tools.simple_earn.get_client", lambda: fake)

    result = await binance_get_earn_flexible_positions(_EarnFlexiblePositionsInput(response_format=ResponseFormat.JSON))

    assert '"asset": "BTC"' in result
    assert '"total": 1' in result


async def test_flexible_positions_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/simple-earn/flexible/position": _status_error(
                "/sapi/v1/simple-earn/flexible/position",
                401,
                {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."},
            )
        }
    )
    monkeypatch.setattr("binance_mcp.tools.simple_earn.get_client", lambda: fake)

    result = await binance_get_earn_flexible_positions(_EarnFlexiblePositionsInput())

    assert result.startswith("Error (401)")
    assert "allowlist" in result


# -- locked positions -----------------------------------------------------------


async def test_locked_positions_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/simple-earn/locked/position": {"rows": [LOCKED_ROW], "total": 1}})
    monkeypatch.setattr("binance_mcp.tools.simple_earn.get_client", lambda: fake)

    result = await binance_get_earn_locked_positions(_EarnLockedPositionsInput(asset="axs"))

    assert "**AXS**" in result
    assert "position `123123`" in result
    assert "project `Axs*90`" in result
    assert "APY 2.3%" in result
    assert "duration 90d" in result
    assert "renewable: True" in result
    call = fake.calls[0]
    assert call[1] == "/sapi/v1/simple-earn/locked/position"
    assert call[2] == {
        "params": {"asset": "AXS", "positionId": None, "projectId": None, "current": 1, "size": 10},
        "auth": "signed",
    }


async def test_locked_positions_filters_and_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/simple-earn/locked/position": {"rows": [], "total": 0}})
    monkeypatch.setattr("binance_mcp.tools.simple_earn.get_client", lambda: fake)

    await binance_get_earn_locked_positions(
        _EarnLockedPositionsInput(position_id="123123", project_id="Axs*90", limit=50, offset=50)
    )

    call = fake.calls[0]
    assert call[2]["params"] == {
        "asset": None,
        "positionId": "123123",
        "projectId": "Axs*90",
        "current": 2,  # 50 // 50 + 1
        "size": 50,
    }


async def test_locked_positions_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/simple-earn/locked/position": _status_error(
                "/sapi/v1/simple-earn/locked/position",
                403,
                {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."},
            )
        }
    )
    monkeypatch.setattr("binance_mcp.tools.simple_earn.get_client", lambda: fake)

    result = await binance_get_earn_locked_positions(_EarnLockedPositionsInput())

    assert result.startswith("Error (403)")


# -- account summary --------------------------------------------------------


async def test_earn_account_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/simple-earn/account": ACCOUNT_SUMMARY})
    monkeypatch.setattr("binance_mcp.tools.simple_earn.get_client", lambda: fake)

    result = await binance_get_earn_account(_EarnAccountInput())

    assert "0.01067982 BTC / 77.1328923 USDT" in result
    assert "flexible**: 0 BTC / 0 USDT" in result
    assert "locked**: 0.01067982 BTC / 77.1328923 USDT" in result
    call = fake.calls[0]
    assert call[1] == "/sapi/v1/simple-earn/account"
    assert call[2] == {"auth": "signed"}


async def test_earn_account_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/simple-earn/account": ACCOUNT_SUMMARY})
    monkeypatch.setattr("binance_mcp.tools.simple_earn.get_client", lambda: fake)

    result = await binance_get_earn_account(_EarnAccountInput(response_format=ResponseFormat.JSON))

    assert '"totalAmountInBTC": "0.01067982"' in result


async def test_earn_account_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/simple-earn/account": _status_error(
                "/sapi/v1/simple-earn/account", 401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions."}
            )
        }
    )
    monkeypatch.setattr("binance_mcp.tools.simple_earn.get_client", lambda: fake)

    result = await binance_get_earn_account(_EarnAccountInput())

    assert result.startswith("Error (401)")


# -- live smoke (real mainnet account, read-only; auto-skipped without creds) --


@pytest.mark.live
async def test_flexible_positions_live_smoke() -> None:
    result = await binance_get_earn_flexible_positions(_EarnFlexiblePositionsInput(limit=5))
    assert not result.startswith("Error")


@pytest.mark.live
async def test_locked_positions_live_smoke() -> None:
    result = await binance_get_earn_locked_positions(_EarnLockedPositionsInput(limit=5))
    assert not result.startswith("Error")


@pytest.mark.live
async def test_earn_account_live_smoke() -> None:
    result = await binance_get_earn_account(_EarnAccountInput())
    assert not result.startswith("Error")
