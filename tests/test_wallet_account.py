"""Unit tests for the wallet account + system tools against a fake client."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from binance_mcp.formatters import ResponseFormat
from binance_mcp.tools.wallet_account import (
    AccountInfoInput,
    AccountSnapshotInput,
    AccountStatusInput,
    ApiRestrictionsInput,
    ApiTradingStatusInput,
    DelistScheduleInput,
    SnapshotType,
    SystemStatusInput,
    binance_get_account_info,
    binance_get_account_snapshot,
    binance_get_account_status,
    binance_get_api_restrictions,
    binance_get_api_trading_status,
    binance_get_delist_schedule,
    binance_get_system_status,
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


# -- binance_get_account_status ----------------------------------------------------


async def test_account_status_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/account/status": {"data": "Normal"}})
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_account_status(AccountStatusInput())

    assert "status**: Normal" in result
    assert fake.calls == [("GET", "/sapi/v1/account/status", {"auth": "signed"})]


async def test_account_status_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/account/status": {"data": "Normal"}})
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_account_status(AccountStatusInput(response_format=ResponseFormat.JSON))

    assert '"data": "Normal"' in result


async def test_account_status_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one error-path test for the module — every tool shares handle_api_error."""
    fake = _FakeClient(
        routes={
            "/sapi/v1/account/status": _status_error(
                "/sapi/v1/account/status", 401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions."}
            )
        }
    )
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_account_status(AccountStatusInput())

    assert result.startswith("Error (401)")
    assert "allowlist" in result


# -- binance_get_api_trading_status ------------------------------------------------


async def test_api_trading_status_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/account/apiTradingStatus": {
                "data": {
                    "isLocked": False,
                    "plannedRecoverTime": 0,
                    "triggerCondition": {"GCR": 150, "IFER": 150, "UFR": 300},
                    "updateTime": 1698000000000,
                }
            }
        }
    )
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_api_trading_status(ApiTradingStatusInput())

    assert "locked**: False" in result
    assert "GCR (GTC cancellation ratio): 150" in result
    assert "UFR (unfilled ratio): 300" in result
    assert fake.calls == [("GET", "/sapi/v1/account/apiTradingStatus", {"auth": "signed"})]


# -- binance_get_api_restrictions ---------------------------------------------------

RESTRICTIONS = {
    "ipRestrict": True,
    "createTime": 1698000000000,
    "enableInternalTransfer": True,
    "enableReading": True,
    "enableSpotAndMarginTrading": True,
    "enableWithdrawals": False,
    "permitsUniversalTransfer": False,
    "enableFutures": False,
    "enableMargin": False,
    "enablePortfolioMarginTrading": False,
    "enableVanillaOptions": False,
    "tradingAuthorityExpirationTime": 0,
}


async def test_api_restrictions_full_flag_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/account/apiRestrictions": RESTRICTIONS})
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_api_restrictions(ApiRestrictionsInput())

    assert "withdrawals**: False" in result
    assert "internal transfer**: True" in result
    assert "portfolio margin trading**: False" in result
    assert "vanilla options**: False" in result
    assert "⚠️" not in result
    assert fake.calls == [("GET", "/sapi/v1/account/apiRestrictions", {"auth": "signed"})]


async def test_api_restrictions_flags_risky_key(monkeypatch: pytest.MonkeyPatch) -> None:
    risky = {**RESTRICTIONS, "enableWithdrawals": True, "ipRestrict": False}
    fake = _FakeClient(routes={"/sapi/v1/account/apiRestrictions": risky})
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_api_restrictions(ApiRestrictionsInput())

    assert "should be OFF" in result
    assert "add an IP allowlist" in result


# -- binance_get_account_info --------------------------------------------------------


async def test_account_info_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/account/info": {
                "vipLevel": 0,
                "isMarginEnabled": True,
                "isFutureEnabled": False,
                "isOptionsEnabled": False,
                "isPortfolioMarginRetailEnabled": False,
            }
        }
    )
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_account_info(AccountInfoInput())

    assert "VIP level**: 0" in result
    assert "margin enabled**: True" in result
    assert fake.calls == [("GET", "/sapi/v1/account/info", {"auth": "signed"})]


# -- binance_get_account_snapshot ----------------------------------------------------


async def test_account_snapshot_happy_path_hides_zero_balances(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/accountSnapshot": {
                "code": 200,
                "msg": "",
                "snapshotVos": [
                    {
                        "type": "spot",
                        "updateTime": 1698000000000,
                        "data": {
                            "totalAssetOfBtc": "0.5",
                            "balances": [
                                {"asset": "BTC", "free": "0.5", "locked": "0"},
                                {"asset": "ETH", "free": "0", "locked": "0"},
                            ],
                        },
                    }
                ],
            }
        }
    )
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_account_snapshot(AccountSnapshotInput(type=SnapshotType.SPOT))

    assert "total (BTC)**: 0.5" in result
    assert "| BTC | 0.5 | 0 |" in result
    assert "ETH" not in result  # zero balance hidden
    assert fake.calls == [
        (
            "GET",
            "/sapi/v1/accountSnapshot",
            {
                "auth": "signed",
                "params": {"type": "SPOT", "startTime": None, "endTime": None, "limit": 7},
            },
        )
    ]


async def test_account_snapshot_no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/accountSnapshot": {"code": 200, "msg": "", "snapshotVos": []}})
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_account_snapshot(AccountSnapshotInput(type=SnapshotType.SPOT))

    assert "No snapshots in range" in result


async def test_account_snapshot_window_over_30_days_rejected_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_account_snapshot(
        AccountSnapshotInput(type=SnapshotType.SPOT, start_time=0, end_time=31 * 24 * 60 * 60 * 1000)
    )

    assert result.startswith("Error")
    assert "30 days" in result
    assert fake.calls == []  # rejected before any request was made


async def test_account_snapshot_end_before_start_rejected_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_account_snapshot(
        AccountSnapshotInput(type=SnapshotType.SPOT, start_time=1000, end_time=500)
    )

    assert result.startswith("Error")
    assert fake.calls == []


async def test_account_snapshot_accepts_iso_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/accountSnapshot": {"code": 200, "msg": "", "snapshotVos": []}})
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    await binance_get_account_snapshot(
        AccountSnapshotInput(type=SnapshotType.SPOT, start_time="2026-09-01", end_time="2026-09-10")
    )

    sent_params = fake.calls[0][2]["params"]
    assert sent_params["startTime"] == 1788220800000
    assert sent_params["endTime"] == 1788998400000


# -- binance_get_system_status --------------------------------------------------------


async def test_system_status_normal(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/system/status": {"status": 0, "msg": "normal"}})
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_system_status(SystemStatusInput())

    assert "status**: normal" in result
    assert fake.calls == [("GET", "/sapi/v1/system/status", {"auth": "none"})]


async def test_system_status_maintenance(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/system/status": {"status": 1, "msg": "system maintenance"}})
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_system_status(SystemStatusInput())

    assert "status**: maintenance" in result
    assert "system maintenance" in result


# -- binance_get_delist_schedule -------------------------------------------------------


async def test_delist_schedule_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/spot/delist-schedule": [
                {"delistTime": 1698000000000, "symbols": ["BADCOINUSDT", "BADCOINBTC"]},
            ]
        }
    )
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_delist_schedule(DelistScheduleInput())

    assert "BADCOINUSDT, BADCOINBTC" in result
    assert fake.calls == [("GET", "/sapi/v1/spot/delist-schedule", {"auth": "key"})]


async def test_delist_schedule_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/spot/delist-schedule": []})
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_delist_schedule(DelistScheduleInput())

    assert "No symbols currently scheduled" in result


async def test_delist_schedule_display_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Edge case: more rows than MAX_DISPLAY_ROWS are truncated with a note."""
    rows = [{"delistTime": 1698000000000 + i, "symbols": [f"COIN{i}USDT"]} for i in range(60)]
    fake = _FakeClient(routes={"/sapi/v1/spot/delist-schedule": rows})
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_delist_schedule(DelistScheduleInput())

    assert "COIN0USDT" in result
    assert "10 more row(s) not shown" in result


# -- live smoke -------------------------------------------------------------------------


@pytest.mark.live
async def test_live_smoke_all_read_tools() -> None:
    """Smoke-test every wallet_account read tool against the real configured account.

    Requires BINANCE_API_KEY + secret/PEM in the environment or .env (see conftest's
    live-marker gating). All seven endpoints here live under `/sapi` (or `/sapi`-adjacent
    for system status), none of which exist on the spot testnet — so this module has no
    equivalent `@pytest.mark.trading` test. `binance_get_account_snapshot` alone costs IP
    weight 2400; this test calls it once.
    """
    results = await asyncio.gather(
        binance_get_account_status(AccountStatusInput()),
        binance_get_api_trading_status(ApiTradingStatusInput()),
        binance_get_api_restrictions(ApiRestrictionsInput()),
        binance_get_account_info(AccountInfoInput()),
        binance_get_account_snapshot(AccountSnapshotInput(type=SnapshotType.SPOT)),
        binance_get_system_status(SystemStatusInput()),
        binance_get_delist_schedule(DelistScheduleInput()),
    )
    for result in results:
        assert isinstance(result, str)
        assert result
