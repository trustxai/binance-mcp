"""Unit tests for the error-to-string mapping."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from binance_mcp.client import BinanceEnvelopeError, ForbiddenEndpointError, TradingDisabledError
from binance_mcp.config import get_settings
from binance_mcp.errors import handle_api_error

# Fake credentials only. The conftest autouse fixture strips ambient BINANCE_* vars
# and chdirs away from any .env, so these are the only values Settings can see.
_FAKE_KEY = "FAKEKEY1234567890abcdef"
_FAKE_SECRET = "FAKESECRETzyxwvu0987654321"
_FAKE_PASSPHRASE = "FAKEPASSPHRASE-hunter2"


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


# --- credential exposure -------------------------------------------------------------


def _configure(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()


@pytest.mark.parametrize(
    "header_value",
    [f"b'{_FAKE_KEY} '", f"b'{_FAKE_KEY}\\n'", f"b'FAKEKEY\\n{_FAKE_KEY}'"],
    ids=["trailing-space", "trailing-newline", "internal-newline"],
)
def test_transport_error_never_echoes_header_value(header_value: str) -> None:
    # No credentials configured: the transport branch must hold without redaction.
    result = handle_api_error(httpx.LocalProtocolError(f"Illegal header value {header_value}"))
    assert _FAKE_KEY not in result
    assert "Illegal header value" not in result
    assert "LocalProtocolError" in result
    assert "whitespace" in result
    assert "BINANCE_API_KEY" in result
    assert "UNKNOWN" in result


def test_transport_error_other_subclass_names_type() -> None:
    result = handle_api_error(httpx.RemoteProtocolError(f"peer closed: {_FAKE_KEY}"))
    assert _FAKE_KEY not in result
    assert "RemoteProtocolError" in result


_CREDENTIALS = [
    ("BINANCE_API_KEY", _FAKE_KEY),
    ("BINANCE_API_SECRET", _FAKE_SECRET),
    ("BINANCE_PRIVATE_KEY_PASSPHRASE", _FAKE_PASSPHRASE),
]
_CREDENTIAL_IDS = ["key", "secret", "passphrase"]

_EXCEPTIONS: list[Callable[[str], Exception]] = [ValueError, RuntimeError, KeyError]


@pytest.mark.parametrize(("env_name", "value"), _CREDENTIALS, ids=_CREDENTIAL_IDS)
@pytest.mark.parametrize("make_exc", _EXCEPTIONS, ids=lambda cls: cls.__name__)
def test_configured_credential_is_redacted(
    monkeypatch: pytest.MonkeyPatch, env_name: str, value: str, make_exc: Callable[[str], Exception]
) -> None:
    _configure(monkeypatch, **{env_name: value})
    result = handle_api_error(make_exc(f"something mentioned {value} here"))
    assert value not in result
    assert "something mentioned *** here" in result


def test_runtime_error_redaction_keeps_the_rest_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, BINANCE_API_KEY=_FAKE_KEY, BINANCE_API_SECRET=_FAKE_SECRET)
    result = handle_api_error(RuntimeError(f"key={_FAKE_KEY} secret={_FAKE_SECRET}"))
    assert result == "Error: key=*** secret=***"


def test_whitespace_stripped_form_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, BINANCE_API_KEY=f"  {_FAKE_KEY} \n")
    result = handle_api_error(ValueError(f"bad key {_FAKE_KEY}!"))
    assert _FAKE_KEY not in result
    assert "bad key ***!" in result


def test_status_error_body_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, BINANCE_API_KEY=_FAKE_KEY)
    result = handle_api_error(_status_error(400, {"code": -1100, "msg": f"Illegal characters in {_FAKE_KEY}"}))
    assert _FAKE_KEY not in result
    assert result.startswith("Error (400)")


def test_short_configured_values_do_not_mangle_output(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both values appear in the message, but under 8 chars they are not treated as secrets.
    _configure(monkeypatch, BINANCE_API_KEY="key", BINANCE_API_SECRET="Binance")
    assert handle_api_error(RuntimeError("No Binance API key configured.")) == "Error: No Binance API key configured."


def test_whitespace_only_value_does_not_mangle_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, BINANCE_API_SECRET=" " * 12)
    message = "spaced            out"
    assert handle_api_error(RuntimeError(message)) == f"Error: {message}"


def test_unloadable_settings_returns_text_unredacted(monkeypatch: pytest.MonkeyPatch) -> None:
    # An invalid env var makes get_settings() raise; the tool must still get its message.
    _configure(monkeypatch, BINANCE_RECV_WINDOW_MS="not-a-number")
    assert handle_api_error(RuntimeError("boom")) == "Error: boom"


_TIMEOUT_MESSAGE = (
    "Error: the Binance API request timed out. Execution status is UNKNOWN for a mutating call — check "
    "before retrying. Raise BINANCE_REQUEST_TIMEOUT_SECONDS if this recurs."
)
_CONNECT_MESSAGE = "Error: could not connect to the Binance API. Check network access and BINANCE_API_URL."


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.TimeoutException("slow"), _TIMEOUT_MESSAGE),
        (httpx.ReadTimeout("slow"), _TIMEOUT_MESSAGE),
        (httpx.ConnectTimeout("slow"), _TIMEOUT_MESSAGE),
        (httpx.ConnectError("refused"), _CONNECT_MESSAGE),
    ],
    ids=["timeout", "read-timeout", "connect-timeout", "connect-error"],
)
def test_timeout_and_connect_branches_unchanged(monkeypatch: pytest.MonkeyPatch, exc: Exception, expected: str) -> None:
    # Both are TransportError subclasses: the new transport branch must not shadow them.
    _configure(monkeypatch, BINANCE_API_KEY=_FAKE_KEY)
    assert handle_api_error(exc) == expected
