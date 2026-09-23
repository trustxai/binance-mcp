"""Async HTTP client for the Binance Spot + Wallet REST API, with request signing.

Binance has three endpoint security types, exposed here as the `auth` argument:

- ``"none"``   — public market data; no header, no signature.
- ``"key"``    — ``X-MBX-APIKEY`` header only (e.g. historicalTrades, user data streams).
- ``"signed"`` — header + ``timestamp``/``recvWindow`` + ``signature`` over the query
  string (USER_DATA and TRADE endpoints). HMAC keys produce a hex digest; Ed25519/RSA
  keys produce a base64 signature — Binance's own examples are the unit-test vectors.

Every parameter is sent in the query string (Binance accepts that for every method and
it keeps the signed payload identical to what is transmitted).

Two safety rails live HERE, not in the tools, so no wave module can forget them:

- **Trading kill-switch.** A signed non-GET request moves funds or changes account
  state; it is refused with ``TradingDisabledError`` unless ``BINANCE_ALLOW_TRADING=1``.
  The handful of POST endpoints that only *read* (and the order dry-run) are allowlisted.
- **Forbidden endpoints.** Withdrawals are never callable through this server, whatever
  the flags say (``ForbiddenEndpointError``).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlencode

import httpx

from binance_mcp.config import Settings, get_settings

AuthType = Literal["none", "key", "signed"]

# Signed non-GET endpoints that only READ (Binance uses POST for a few queries), plus
# the order dry-run. Everything else that is signed and not GET needs the kill-switch.
POST_READ_ALLOWLIST: frozenset[str] = frozenset(
    {
        "/api/v3/order/test",
        "/api/v3/sor/order/test",
        "/sapi/v1/asset/get-funding-asset",
        "/sapi/v3/asset/getUserAsset",
        "/sapi/v1/asset/dust-btc",
        "/sapi/v1/convert/getQuote",
    }
)

# Never callable through this server: anything that moves funds OUT of Binance (crypto
# withdrawals, fiat rails). Belt and braces: the API key should also have "Enable
# Withdrawals" OFF, but a mis-scoped key must not be enough to move funds out.
FORBIDDEN_PATHS: frozenset[str] = frozenset(
    {
        "/sapi/v1/capital/withdraw/apply",
        "/sapi/v1/capital/withdraw",
        "/wapi/v3/withdraw.html",
        "/sapi/v1/fiat/withdraw",
        "/sapi/v2/fiat/withdraw",
        "/sapi/v1/fiat/deposit",
    }
)


class BinanceEnvelopeError(RuntimeError):
    """A 200 response whose /sapi envelope says `success: false` (fiat, pay, snapshot…)."""

    def __init__(self, code: Any, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"Binance rejected the request: {message} (code {code})")


class TradingDisabledError(RuntimeError):
    """Raised for a fund-moving request while BINANCE_ALLOW_TRADING is off."""


class ForbiddenEndpointError(RuntimeError):
    """Raised for an endpoint this server refuses to call under any configuration."""


class Repeat(list[Any]):
    """Marker for a parameter Binance wants REPEATED (`asset=BTC&asset=ETH`, e.g. dust
    conversion) rather than JSON-encoded like `symbols=["BTCUSDT","ETHUSDT"]`."""


def _encode_value(value: Any) -> str:
    """Render one query value the way Binance expects it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        # e.g. symbols=["BTCUSDT","ETHUSDT"] — compact JSON, no spaces.
        return json.dumps(list(value), separators=(",", ":"))
    return str(value)


def build_query(params: dict[str, Any] | None) -> str:
    """Build the exact query string that is both signed and sent (None values dropped)."""
    if not params:
        return ""
    pairs: list[tuple[str, str]] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, Repeat):
            pairs.extend((key, _encode_value(item)) for item in value if item is not None)
        else:
            pairs.append((key, _encode_value(value)))
    return urlencode(pairs)


def is_fund_moving(method: str, path: str) -> bool:
    """A signed request that is not a GET changes account state, bar the read allowlist."""
    return method.upper() != "GET" and path not in POST_READ_ALLOWLIST


class BinanceClient:
    """Thin async wrapper around httpx for the Binance REST API."""

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        # Injectable transport so tests can use httpx.MockTransport.
        self._transport = transport
        # Injectable clock so tests can pin Binance's published signature vectors.
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._private_key: Any | None = None
        # Diagnostics captured from the last response (health tool reports them).
        # /api budgets are per IP (6000/min); /sapi has its own per-IP (12000/min) and
        # per-UID (180000/min) budgets, reported in separate headers.
        self.last_used_weight_1m: int | None = None
        self.last_order_count_10s: int | None = None
        self.last_sapi_ip_weight_1m: int | None = None
        self.last_sapi_uid_weight_1m: int | None = None

    # -- signing -------------------------------------------------------------

    def _load_private_key(self) -> Any:
        if self._private_key is None:
            from cryptography.hazmat.primitives import serialization

            pem = Path(self._settings.binance_private_key_path).expanduser().read_bytes()
            passphrase = self._settings.binance_private_key_passphrase.encode() or None
            self._private_key = serialization.load_pem_private_key(pem, password=passphrase)
        return self._private_key

    def sign(self, payload: str) -> str:
        """Sign the exact query string: HMAC-SHA256 hex, or Ed25519/RSA base64."""
        settings = self._settings
        if settings.binance_private_key_path:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

            key = self._load_private_key()
            if isinstance(key, ed25519.Ed25519PrivateKey):
                raw = key.sign(payload.encode())
            elif isinstance(key, rsa.RSAPrivateKey):
                raw = key.sign(payload.encode(), padding.PKCS1v15(), hashes.SHA256())
            else:
                raise RuntimeError(
                    f"Unsupported private key type {type(key).__name__} in BINANCE_PRIVATE_KEY_PATH; "
                    "Binance accepts Ed25519 or RSA PEM keys."
                )
            return base64.b64encode(raw).decode()
        return hmac.new(settings.binance_api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    # -- requests ------------------------------------------------------------

    def _require_key(self) -> None:
        if not self._settings.has_api_key:
            raise RuntimeError(
                "No Binance API key configured. Set BINANCE_API_KEY in the environment or .env "
                "(Binance → Account → API Management)."
            )

    def _require_credentials(self) -> None:
        self._require_key()
        if not self._settings.has_credentials:
            raise RuntimeError(
                "BINANCE_API_KEY is set but nothing can sign requests: set BINANCE_API_SECRET (HMAC key) "
                "or BINANCE_PRIVATE_KEY_PATH (Ed25519/RSA PEM)."
            )

    def _guard(self, method: str, path: str, auth: AuthType) -> None:
        if path in FORBIDDEN_PATHS:
            raise ForbiddenEndpointError(
                f"{method.upper()} {path} moves funds out of Binance (withdrawal / fiat rail) — this server "
                "never calls it, by design. Use the Binance app for withdrawals."
            )
        if auth == "signed" and is_fund_moving(method, path) and not self._settings.binance_allow_trading:
            raise TradingDisabledError(
                f"{method.upper()} {path} would move funds or change account state, but trading is disabled. "
                "Set BINANCE_ALLOW_TRADING=1 to enable order placement/cancellation, transfers, convert and "
                "algo orders. Dry-run validation (POST /api/v3/order/test) works without it."
            )

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        auth: AuthType = "none",
        timeout: float | None = None,
    ) -> httpx.Response:
        """Perform a request and return the response.

        Raises `httpx.HTTPStatusError` on non-2xx (tools route it through
        `binance_mcp.errors.handle_api_error`), `RuntimeError` when credentials are
        missing, `TradingDisabledError` / `ForbiddenEndpointError` per the rails above.
        """
        settings = self._settings
        self._guard(method, path, auth)

        headers = {"accept": "application/json"}
        if auth in ("key", "signed"):
            self._require_key()
            headers["X-MBX-APIKEY"] = settings.binance_api_key

        query_params: dict[str, Any] = dict(params or {})
        if auth == "signed":
            self._require_credentials()
            query_params["recvWindow"] = settings.binance_recv_window_ms
            query_params["timestamp"] = self._clock_ms()
        query = build_query(query_params)
        if auth == "signed":
            # HMAC hex is URL-safe as is; Ed25519/RSA base64 carries `+ / =` and MUST be
            # percent-encoded in the query (the signed payload itself is unchanged).
            query = f"{query}&signature={quote(self.sign(query), safe='')}"

        url = f"{settings.base_url.rstrip('/')}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{query}"
        effective_timeout = timeout if timeout is not None else settings.binance_request_timeout_seconds

        async with httpx.AsyncClient(timeout=effective_timeout, transport=self._transport) as client:
            resp = await client.request(method, url, headers=headers)

        self._capture_limits(resp)
        resp.raise_for_status()
        self._check_envelope(resp)
        return resp

    @staticmethod
    def _check_envelope(resp: httpx.Response) -> None:
        """Some /sapi endpoints answer 200 with `{"success": false, "code": …, "message": …}`."""
        if not resp.headers.get("content-type", "").startswith("application/json"):
            return
        try:
            body = resp.json()
        except ValueError:
            return
        if isinstance(body, dict) and "success" in body and not body["success"]:
            raise BinanceEnvelopeError(body.get("code"), str(body.get("message") or body.get("msg") or "unknown error"))

    def _capture_limits(self, resp: httpx.Response) -> None:
        def _int(header: str) -> int | None:
            value = resp.headers.get(header)
            return int(value) if value is not None and value.isdigit() else None

        if (weight := _int("x-mbx-used-weight-1m")) is not None:
            self.last_used_weight_1m = weight
        if (orders := _int("x-mbx-order-count-10s")) is not None:
            self.last_order_count_10s = orders
        if (sapi_ip := _int("x-sapi-used-ip-weight-1m")) is not None:
            self.last_sapi_ip_weight_1m = sapi_ip
        if (sapi_uid := _int("x-sapi-used-uid-weight-1m")) is not None:
            self.last_sapi_uid_weight_1m = sapi_uid


_client: BinanceClient | None = None


def get_client() -> BinanceClient:
    """Lazy module-level singleton (tools monkeypatch this in unit tests)."""
    global _client
    if _client is None:
        _client = BinanceClient()
    return _client
