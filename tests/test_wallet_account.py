"""Unit tests for the wallet account + system tools against a fake client."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
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
                "params": {"type": "SPOT", "limit": 7},
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


async def test_account_snapshot_only_start_time_defaults_end_to_now_and_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No end_time given: it defaults to now, and the span (start..now) is still checked."""
    fake = _FakeClient()
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_account_snapshot(AccountSnapshotInput(type=SnapshotType.SPOT, start_time=0))

    assert result.startswith("Error")
    assert "30 days" in result
    assert fake.calls == []


async def test_account_snapshot_start_time_older_than_retention_rejected_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A narrow (< 30 day) window that is entirely older than Binance's ~30-day retention."""
    fake = _FakeClient()
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    day_ms = 24 * 60 * 60 * 1000
    start_time = now_ms - 60 * day_ms
    end_time = now_ms - 55 * day_ms  # 5-day span, well under 30 — but both endpoints are ancient

    result = await binance_get_account_snapshot(
        AccountSnapshotInput(type=SnapshotType.SPOT, start_time=start_time, end_time=end_time)
    )

    assert result.startswith("Error")
    assert "30 days" in result
    assert fake.calls == []


async def test_account_snapshot_accepts_iso_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/accountSnapshot": {"code": 200, "msg": "", "snapshotVos": []}})
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    await binance_get_account_snapshot(
        AccountSnapshotInput(type=SnapshotType.SPOT, start_time="2026-09-01", end_time="2026-09-10")
    )

    sent_params = fake.calls[0][2]["params"]
    assert sent_params == {"type": "SPOT", "limit": 7, "startTime": 1788220800000, "endTime": 1788998400000}


async def test_account_snapshot_numeric_string_ge_12_digits_is_epoch_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    """A >=12-digit numeric string is treated as an epoch-ms int, not parsed as ISO."""
    fake = _FakeClient(routes={"/sapi/v1/accountSnapshot": {"code": 200, "msg": "", "snapshotVos": []}})
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    recent_ms = now_ms - 5 * 24 * 60 * 60 * 1000
    assert len(str(recent_ms)) >= 12

    await binance_get_account_snapshot(AccountSnapshotInput(type=SnapshotType.SPOT, start_time=str(recent_ms)))

    assert fake.calls[0][2]["params"]["startTime"] == recent_ms


async def test_account_snapshot_invalid_start_time_returns_named_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_account_snapshot(AccountSnapshotInput(type=SnapshotType.SPOT, start_time="not-a-date"))

    assert result.startswith("Error: start_time must be")
    assert fake.calls == []


async def test_account_snapshot_error_code_in_200_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """accountSnapshot answers HTTP 200 with {code, msg, snapshotVos} on failure — no
    `success` key, so the client's envelope check never fires; the tool must check
    `code` itself."""
    fake = _FakeClient(routes={"/sapi/v1/accountSnapshot": {"code": -1000, "msg": "An unknown error occurred."}})
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_account_snapshot(AccountSnapshotInput(type=SnapshotType.SPOT))

    assert result == "Error: An unknown error occurred. (code -1000)"


async def test_account_snapshot_balances_sorted_desc_and_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    # free = i + 1 (never zero) so all 60 rows survive the zero-balance filter; only the
    # display cap trims them.
    balances = [{"asset": f"COIN{i}", "free": str(i + 1), "locked": "0"} for i in range(60)]
    fake = _FakeClient(
        routes={
            "/sapi/v1/accountSnapshot": {
                "code": 200,
                "msg": "",
                "snapshotVos": [
                    {
                        "type": "spot",
                        "updateTime": 1698000000000,
                        "data": {"totalAssetOfBtc": "1", "balances": balances},
                    }
                ],
            }
        }
    )
    monkeypatch.setattr("binance_mcp.tools.wallet_account.get_client", lambda: fake)

    result = await binance_get_account_snapshot(AccountSnapshotInput(type=SnapshotType.SPOT))

    # largest balance (COIN59, free=60) sorts first; smallest 10 (COIN0..COIN9) are capped out.
    assert "| COIN59 | 60 | 0 |" in result
    assert "COIN0 |" not in result
    assert "10 more asset(s) not shown" in result
    assert result.index("COIN59") < result.index("COIN58")


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
