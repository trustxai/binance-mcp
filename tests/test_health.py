"""Unit tests for the health-check tool against a fake client."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from binance_mcp.config import Settings
from binance_mcp.tools.health import binance_health_check


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Routes by path; records every call (method, path, kwargs)."""

    def __init__(self, routes: dict[str, Any] | None = None, exc: Exception | None = None) -> None:
        self._routes = routes or {}
        self._exc = exc
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.last_used_weight_1m: int | None = 7

    async def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, path, kwargs))
        if self._exc is not None:
            raise self._exc
        payload = self._routes.get(path)
        if isinstance(payload, Exception):
            raise payload
        return _FakeResponse(payload if payload is not None else {})


def _status_error(status: int, body: dict[str, Any]) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.binance.com/sapi/v1/account/apiRestrictions")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


RESTRICTIONS = {
    "ipRestrict": True,
    "createTime": 1698000000000,
    "enableReading": True,
    "enableSpotAndMarginTrading": True,
    "enableWithdrawals": False,
    "permitsUniversalTransfer": False,
    "enableFutures": False,
    "enableMargin": False,
}


async def test_health_public_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/ping": {}, "/api/v3/time": {"serverTime": 1698000000000}})
    monkeypatch.setattr("binance_mcp.tools.health.get_client", lambda: fake)
    monkeypatch.setattr("binance_mcp.tools.health.get_settings", lambda: Settings())

    result = await binance_health_check()

    assert "connectivity**: OK" in result
    assert "2023-10-22" in result  # server time rendered
    assert "none configured" in result
    assert "disabled (read-only" in result
    assert [c[1] for c in fake.calls] == ["/api/v3/ping", "/api/v3/time"]


async def test_health_with_key_reports_permissions(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/ping": {},
            "/api/v3/time": {"serverTime": 1698000000000},
            "/sapi/v1/account/apiRestrictions": RESTRICTIONS,
        }
    )
    monkeypatch.setattr("binance_mcp.tools.health.get_client", lambda: fake)
    monkeypatch.setattr(
        "binance_mcp.tools.health.get_settings",
        lambda: Settings(binance_api_key="k", binance_api_secret="s", binance_allow_trading=True),
    )

    result = await binance_health_check()

    assert "ENABLED — orders/transfers allowed" in result
    assert "withdrawals**: False" in result
    assert "⚠️" not in result
    assert "used weight (1m)**: 7" in result
    signed_call = fake.calls[-1]
    assert signed_call[1] == "/sapi/v1/account/apiRestrictions"
    assert signed_call[2] == {"auth": "signed"}


async def test_health_flags_withdrawals_and_no_ip_restriction(monkeypatch: pytest.MonkeyPatch) -> None:
    risky = {**RESTRICTIONS, "enableWithdrawals": True, "ipRestrict": False}
    fake = _FakeClient(
        routes={"/api/v3/ping": {}, "/api/v3/time": {"serverTime": 1}, "/sapi/v1/account/apiRestrictions": risky}
    )
    monkeypatch.setattr("binance_mcp.tools.health.get_client", lambda: fake)
    monkeypatch.setattr(
        "binance_mcp.tools.health.get_settings", lambda: Settings(binance_api_key="k", binance_api_secret="s")
    )

    result = await binance_health_check()

    assert "should be OFF" in result
    assert "add an IP allowlist" in result


async def test_health_permission_error_is_reported_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/ping": {},
            "/api/v3/time": {"serverTime": 1},
            "/sapi/v1/account/apiRestrictions": _status_error(
                401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."}
            ),
        }
    )
    monkeypatch.setattr("binance_mcp.tools.health.get_client", lambda: fake)
    monkeypatch.setattr(
        "binance_mcp.tools.health.get_settings", lambda: Settings(binance_api_key="k", binance_api_secret="s")
    )

    result = await binance_health_check()

    assert "connectivity**: OK" in result
    assert "Error (401)" in result
    assert "allowlist" in result


async def test_health_testnet_skips_sapi(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/ping": {}, "/api/v3/time": {"serverTime": 1}})
    monkeypatch.setattr("binance_mcp.tools.health.get_client", lambda: fake)
    monkeypatch.setattr(
        "binance_mcp.tools.health.get_settings",
        lambda: Settings(binance_api_key="k", binance_api_secret="s", binance_testnet=True),
    )

    result = await binance_health_check()

    assert "spot testnet" in result
    assert "permissions check skipped" in result
    assert all(c[1] != "/sapi/v1/account/apiRestrictions" for c in fake.calls)


async def test_health_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(exc=httpx.ConnectError("refused"))
    monkeypatch.setattr("binance_mcp.tools.health.get_client", lambda: fake)
    monkeypatch.setattr("binance_mcp.tools.health.get_settings", lambda: Settings())

    result = await binance_health_check()

    assert result.startswith("Error")
    assert "could not connect" in result
