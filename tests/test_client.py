"""Unit tests for BinanceClient using httpx.MockTransport.

The HMAC vectors are the ones Binance publishes in rest-api.md ("SIGNED request
example (HMAC)"): same secret, same payload, same expected digest — so a signing
regression fails against the official numbers, not against our own function.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from binance_mcp.client import (
    FORBIDDEN_PATHS,
    POST_READ_ALLOWLIST,
    BinanceClient,
    BinanceEnvelopeError,
    ForbiddenEndpointError,
    TradingDisabledError,
    build_query,
    is_fund_moving,
)
from binance_mcp.config import Settings

# --- Binance's published HMAC example (rest-api.md, "HMAC Keys") ---------------
DOC_SECRET = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
DOC_API_KEY = "vmPUZE6mv9SD5VNHk4HlWFsOr6aKE2zvsw0MuIgwCIPy6utIco14y7Ju91duEh8A"
DOC_TIMESTAMP = 1499827319559
DOC_PARAMS: dict[str, Any] = {
    "symbol": "LTCBTC",
    "side": "BUY",
    "type": "LIMIT",
    "timeInForce": "GTC",
    "quantity": 1,
    "price": 0.1,
}
DOC_PAYLOAD = (
    "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559"
)
DOC_SIGNATURE = "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"
DOC_PAYLOAD_NON_ASCII = (
    "symbol=%EF%BC%91%EF%BC%92%EF%BC%93%EF%BC%94%EF%BC%95%EF%BC%96&side=BUY&type=LIMIT&timeInForce=GTC"
    "&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559"
)
DOC_SIGNATURE_NON_ASCII = "e1353ec6b14d888f1164ae9af8228a3dbd508bc82eb867db8ab6046442f33ef3"


def _client_with(handler: Any, **settings_kwargs: Any) -> BinanceClient:
    settings = Settings(binance_api_key=DOC_API_KEY, binance_api_secret=DOC_SECRET, **settings_kwargs)
    return BinanceClient(settings=settings, transport=httpx.MockTransport(handler), clock_ms=lambda: DOC_TIMESTAMP)


def _ok_handler(captured: dict[str, Any]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["query"] = request.url.query.decode()
        captured["headers"] = dict(request.headers)
        captured["method"] = request.method
        return httpx.Response(
            200,
            json={"ok": True},
            headers={
                "X-MBX-USED-WEIGHT-1M": "42",
                "X-MBX-ORDER-COUNT-10S": "3",
                "X-SAPI-USED-IP-WEIGHT-1M": "2400",
                "X-SAPI-USED-UID-WEIGHT-1M": "18000",
            },
        )

    return handler


# --- signing ------------------------------------------------------------------


def test_hmac_signature_matches_binance_doc_vector() -> None:
    client = BinanceClient(settings=Settings(binance_api_key=DOC_API_KEY, binance_api_secret=DOC_SECRET))
    assert client.sign(DOC_PAYLOAD) == DOC_SIGNATURE


def test_hmac_signature_matches_non_ascii_doc_vector() -> None:
    client = BinanceClient(settings=Settings(binance_api_key=DOC_API_KEY, binance_api_secret=DOC_SECRET))
    assert client.sign(DOC_PAYLOAD_NON_ASCII) == DOC_SIGNATURE_NON_ASCII


async def test_signed_request_reproduces_doc_url_exactly() -> None:
    captured: dict[str, Any] = {}
    client = _client_with(_ok_handler(captured), binance_allow_trading=True)

    resp = await client.request("POST", "/api/v3/order", params=DOC_PARAMS, auth="signed")

    assert resp.status_code == 200
    assert captured["method"] == "POST"
    assert captured["query"] == f"{DOC_PAYLOAD}&signature={DOC_SIGNATURE}"
    assert captured["url"].startswith("https://api.binance.com/api/v3/order?")
    assert captured["headers"]["x-mbx-apikey"] == DOC_API_KEY


async def test_signed_request_non_ascii_symbol_is_percent_encoded_before_signing() -> None:
    captured: dict[str, Any] = {}
    client = _client_with(_ok_handler(captured), binance_allow_trading=True)

    await client.request("POST", "/api/v3/order", params={**DOC_PARAMS, "symbol": "１２３４５６"}, auth="signed")

    assert captured["query"] == f"{DOC_PAYLOAD_NON_ASCII}&signature={DOC_SIGNATURE_NON_ASCII}"


def _write_pem(path: Path, key: Any, password: bytes | None = None) -> None:
    encryption: serialization.KeySerializationEncryption = (
        serialization.BestAvailableEncryption(password) if password else serialization.NoEncryption()
    )
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption))


def test_ed25519_signature_is_base64_and_verifies(tmp_path: Path) -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    pem = tmp_path / "ed25519.pem"
    _write_pem(pem, key)
    client = BinanceClient(settings=Settings(binance_api_key="k", binance_private_key_path=str(pem)))

    signature = client.sign(DOC_PAYLOAD)

    key.public_key().verify(base64.b64decode(signature), DOC_PAYLOAD.encode())  # raises if wrong


async def test_ed25519_signature_is_percent_encoded_in_query(tmp_path: Path) -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    pem = tmp_path / "ed25519.pem"
    _write_pem(pem, key)
    captured: dict[str, Any] = {}
    settings = Settings(binance_api_key="k", binance_private_key_path=str(pem))
    client = BinanceClient(
        settings=settings, transport=httpx.MockTransport(_ok_handler(captured)), clock_ms=lambda: DOC_TIMESTAMP
    )

    await client.request("GET", "/api/v3/account", auth="signed")

    query = captured["query"]
    assert query.startswith("recvWindow=5000&timestamp=1499827319559&signature=")
    encoded_sig = query.split("signature=", 1)[1]
    # Base64 characters `+ / =` never appear raw; the decoded value verifies over the signed payload.
    assert not any(ch in encoded_sig for ch in "+/=")
    from urllib.parse import unquote

    signed_payload = query.split("&signature=", 1)[0]
    key.public_key().verify(base64.b64decode(unquote(encoded_sig)), signed_payload.encode())


def test_ed25519_pem_with_passphrase(tmp_path: Path) -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    pem = tmp_path / "ed25519-enc.pem"
    _write_pem(pem, key, password=b"hunter2")
    client = BinanceClient(
        settings=Settings(
            binance_api_key="k", binance_private_key_path=str(pem), binance_private_key_passphrase="hunter2"
        )
    )

    signature = client.sign("a=1")

    key.public_key().verify(base64.b64decode(signature), b"a=1")


def test_rsa_signature_pkcs1v15_sha256_verifies(tmp_path: Path) -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = tmp_path / "rsa.pem"
    _write_pem(pem, key)
    client = BinanceClient(settings=Settings(binance_api_key="k", binance_private_key_path=str(pem)))

    signature = client.sign(DOC_PAYLOAD)

    key.public_key().verify(base64.b64decode(signature), DOC_PAYLOAD.encode(), padding.PKCS1v15(), hashes.SHA256())


def test_pem_path_takes_precedence_over_hmac_secret(tmp_path: Path) -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    pem = tmp_path / "ed25519.pem"
    _write_pem(pem, key)
    client = BinanceClient(
        settings=Settings(binance_api_key="k", binance_api_secret=DOC_SECRET, binance_private_key_path=str(pem))
    )

    assert client.sign(DOC_PAYLOAD) != DOC_SIGNATURE


# --- auth modes -----------------------------------------------------------------


async def test_auth_none_sends_no_key_header() -> None:
    captured: dict[str, Any] = {}
    client = _client_with(_ok_handler(captured))

    await client.request("GET", "/api/v3/ping")

    assert "x-mbx-apikey" not in captured["headers"]
    assert captured["query"] == ""


async def test_auth_key_sends_header_without_signature() -> None:
    captured: dict[str, Any] = {}
    client = _client_with(_ok_handler(captured))

    await client.request("GET", "/api/v3/historicalTrades", params={"symbol": "BTCUSDT", "limit": 5}, auth="key")

    assert captured["headers"]["x-mbx-apikey"] == DOC_API_KEY
    assert captured["query"] == "symbol=BTCUSDT&limit=5"


async def test_auth_none_works_without_any_credentials() -> None:
    captured: dict[str, Any] = {}
    client = BinanceClient(settings=Settings(), transport=httpx.MockTransport(_ok_handler(captured)))

    resp = await client.request("GET", "/api/v3/time")

    assert resp.status_code == 200


async def test_missing_key_raises_for_key_auth() -> None:
    client = BinanceClient(settings=Settings(), transport=httpx.MockTransport(_ok_handler({})))
    with pytest.raises(RuntimeError, match="BINANCE_API_KEY"):
        await client.request("GET", "/api/v3/historicalTrades", auth="key")


async def test_key_without_secret_raises_for_signed_auth() -> None:
    client = BinanceClient(settings=Settings(binance_api_key="k"), transport=httpx.MockTransport(_ok_handler({})))
    with pytest.raises(RuntimeError, match="BINANCE_API_SECRET"):
        await client.request("GET", "/api/v3/account", auth="signed")


async def test_signed_get_is_never_gated_by_kill_switch() -> None:
    captured: dict[str, Any] = {}
    client = _client_with(_ok_handler(captured))  # allow_trading defaults to False

    await client.request("GET", "/api/v3/account", auth="signed")

    assert "signature=" in captured["query"]


# --- kill-switch + forbidden endpoints ----------------------------------------


async def test_fund_moving_post_refused_when_trading_disabled() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={})

    client = _client_with(handler)
    with pytest.raises(TradingDisabledError, match="BINANCE_ALLOW_TRADING"):
        await client.request("POST", "/api/v3/order", params=DOC_PARAMS, auth="signed")
    assert calls == []  # refused before any HTTP traffic


async def test_cancel_refused_when_trading_disabled() -> None:
    client = _client_with(_ok_handler({}))
    with pytest.raises(TradingDisabledError):
        await client.request("DELETE", "/api/v3/order", params={"symbol": "BTCUSDT", "orderId": 1}, auth="signed")


async def test_fund_moving_post_allowed_when_trading_enabled() -> None:
    captured: dict[str, Any] = {}
    client = _client_with(_ok_handler(captured), binance_allow_trading=True)

    resp = await client.request("POST", "/api/v3/order", params=DOC_PARAMS, auth="signed")

    assert resp.status_code == 200


@pytest.mark.parametrize("path", sorted(POST_READ_ALLOWLIST))
async def test_post_read_allowlist_bypasses_kill_switch(path: str) -> None:
    captured: dict[str, Any] = {}
    client = _client_with(_ok_handler(captured))

    resp = await client.request("POST", path, params={"symbol": "BTCUSDT"}, auth="signed")

    assert resp.status_code == 200
    assert "signature=" in captured["query"]


@pytest.mark.parametrize("path", sorted(FORBIDDEN_PATHS))
async def test_withdrawals_are_forbidden_even_with_trading_enabled(path: str) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={})

    client = _client_with(handler, binance_allow_trading=True)
    with pytest.raises(ForbiddenEndpointError, match="never calls it"):
        await client.request("POST", path, params={"coin": "BTC"}, auth="signed")
    assert calls == []


async def test_sapi_envelope_success_false_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "NOT_FOUND", "message": "no such order", "success": False})

    client = _client_with(handler)
    with pytest.raises(BinanceEnvelopeError, match="no such order"):
        await client.request("GET", "/sapi/v1/fiat/orders", params={"transactionType": 0}, auth="signed")


async def test_sapi_envelope_success_true_and_lists_pass() -> None:
    payloads: list[Any] = [{"code": "000000", "message": "success", "data": [], "success": True}, [{"id": 1}], {}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payloads.pop(0))

    client = _client_with(handler)
    for _ in range(3):
        resp = await client.request("GET", "/api/v3/ping")
        assert resp.status_code == 200


def test_is_fund_moving_rules() -> None:
    assert is_fund_moving("GET", "/api/v3/account") is False
    assert is_fund_moving("POST", "/api/v3/order") is True
    assert is_fund_moving("DELETE", "/api/v3/openOrders") is True
    assert is_fund_moving("POST", "/api/v3/order/test") is False
    assert is_fund_moving("POST", "/sapi/v1/asset/transfer") is True
    assert is_fund_moving("POST", "/sapi/v1/convert/acceptQuote") is True
    assert is_fund_moving("POST", "/sapi/v1/convert/getQuote") is False


# --- query encoding -------------------------------------------------------------


def test_build_query_encodes_bools_lists_and_drops_none() -> None:
    query = build_query(
        {"symbols": ["BTCUSDT", "ETHUSDT"], "isIsolated": True, "limit": None, "recvWindow": 5000, "price": 0.1}
    )
    assert query == "symbols=%5B%22BTCUSDT%22%2C%22ETHUSDT%22%5D&isIsolated=true&recvWindow=5000&price=0.1"


def test_build_query_empty() -> None:
    assert build_query(None) == ""
    assert build_query({}) == ""


# --- transport details ----------------------------------------------------------


async def test_testnet_base_url() -> None:
    captured: dict[str, Any] = {}
    client = _client_with(_ok_handler(captured), binance_testnet=True)

    await client.request("GET", "/api/v3/ping")

    assert captured["url"] == "https://testnet.binance.vision/api/v3/ping"


async def test_rate_limit_headers_are_captured() -> None:
    client = _client_with(_ok_handler({}))

    await client.request("GET", "/api/v3/ping")

    assert client.last_used_weight_1m == 42
    assert client.last_order_count_10s == 3
    assert client.last_sapi_ip_weight_1m == 2400
    assert client.last_sapi_uid_weight_1m == 18000


async def test_http_error_raises_status_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})

    client = _client_with(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.request("GET", "/api/v3/ticker/price", params={"symbol": "NOPE"})
