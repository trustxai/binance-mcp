"""Unit tests for the error-to-string mapping."""

from __future__ import annotations

from typing import Any

import httpx

from binance_mcp.client import BinanceEnvelopeError, ForbiddenEndpointError, TradingDisabledError
from binance_mcp.errors import handle_api_error


def _status_error(
    status: int, body: dict[str, Any] | None = None, headers: dict[str, str] | None = None
) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.binance.com/api/v3/thing")
    response = httpx.Response(status, json=body or {}, request=request, headers=headers)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_400_surfaces_code_and_hint() -> None:
    result = handle_api_error(_status_error(400, {"code": -1121, "msg": "Invalid symbol."}))
    assert result.startswith("Error (400)")
    assert "Invalid symbol." in result
    assert "code -1121" in result
    assert "BTCUSDT" in result  # the hint


def test_400_timestamp_drift_hint() -> None:
    result = handle_api_error(
        _status_error(400, {"code": -1021, "msg": "Timestamp for this request is outside of the recvWindow."})
    )
    assert "BINANCE_RECV_WINDOW_MS" in result
    assert "clock" in result


def test_401_permissions_hint() -> None:
    result = handle_api_error(
        _status_error(401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."})
    )
    assert result.startswith("Error (401)")
    assert "allowlist" in result


def test_401_without_code_names_the_key() -> None:
    result = handle_api_error(_status_error(401))
    assert "BINANCE_API_KEY" in result


def test_404_mentions_testnet() -> None:
    result = handle_api_error(_status_error(404, {"code": -1, "msg": "not found"}))
    assert result.startswith("Error (404)")
    assert "testnet" in result


def test_429_retry_after() -> None:
    result = handle_api_error(_status_error(429, {"code": -1003, "msg": "Too many requests."}, {"Retry-After": "12"}))
    assert result.startswith("Error (429)")
    assert "Retry-After: 12s" in result


def test_418_ban() -> None:
    result = handle_api_error(
        _status_error(418, {"code": -1003, "msg": "Way too much request weight used; IP banned."})
    )
    assert result.startswith("Error (418)")
    assert "banned" in result


def test_5xx_execution_unknown() -> None:
    result = handle_api_error(_status_error(503, {"code": -1000, "msg": "An unknown error occurred."}))
    assert result.startswith("Error (503)")
    assert "UNKNOWN" in result


def test_order_rejected_hint() -> None:
    result = handle_api_error(_status_error(400, {"code": -2010, "msg": "Account has insufficient balance."}))
    assert "dry-run" in result


def test_non_json_body() -> None:
    request = httpx.Request("GET", "https://api.binance.com/api/v3/thing")
    response = httpx.Response(403, text="<html>WAF</html>", request=request)
    result = handle_api_error(httpx.HTTPStatusError("boom", request=request, response=response))
    assert result.startswith("Error (403)")
    assert "WAF" in result


def test_timeout_flags_unknown_execution() -> None:
    result = handle_api_error(httpx.TimeoutException("slow"))
    assert "timed out" in result
    assert "UNKNOWN" in result


def test_connect_error() -> None:
    assert "could not connect" in handle_api_error(httpx.ConnectError("refused"))


def test_runtime_error_passthrough() -> None:
    assert handle_api_error(RuntimeError("No Binance API key configured.")) == "Error: No Binance API key configured."


def test_trading_disabled_is_runtime_error() -> None:
    result = handle_api_error(TradingDisabledError("trading is disabled"))
    assert result == "Error: trading is disabled"


def test_envelope_error_is_runtime_error() -> None:
    result = handle_api_error(BinanceEnvelopeError("NOT_FOUND", "no such order"))
    assert result == "Error: Binance rejected the request: no such order (code NOT_FOUND)"


def test_forbidden_endpoint_is_runtime_error() -> None:
    result = handle_api_error(ForbiddenEndpointError("withdrawals never"))
    assert result == "Error: withdrawals never"


def test_unexpected_exception() -> None:
    result = handle_api_error(ValueError("odd"))
    assert "unexpected failure" in result
    assert "ValueError" in result
