"""Uniform error-to-string mapping for tool responses.

Binance error bodies are ``{"code": -1121, "msg": "Invalid symbol."}``; the HTTP
status says *which family* (4xx malformed/auth, 429/418 limits, 5xx Binance side) and
the negative ``code`` says *what*. Both are surfaced, plus a hint for the codes that
have a known fix.
"""

from __future__ import annotations

import httpx

# Codes worth a hint. Full list: https://developers.binance.com/docs/binance-spot-api-docs/errors
_CODE_HINTS: dict[int, str] = {
    -1000: "Unknown Binance error — retry once; if it persists check status.binance.com.",
    -1003: "Too many requests — back off; the response headers X-MBX-USED-WEIGHT-1M / Retry-After say how long.",
    -1007: "Binance backend timeout — execution status is UNKNOWN; check order status before retrying.",
    -1015: "Too many orders — the per-account order rate limit was hit (see binance_get_order_rate_limits).",
    -1021: (
        "Timestamp outside recvWindow — this machine's clock drifts from Binance's. Sync the clock (NTP) "
        "or raise BINANCE_RECV_WINDOW_MS (max 60000); binance_health_check reports the drift."
    ),
    -1022: (
        "Invalid signature — the secret does not match the key, or the key type is wrong "
        "(HMAC secret vs Ed25519/RSA PEM). Re-check BINANCE_API_SECRET / BINANCE_PRIVATE_KEY_PATH."
    ),
    -1100: "Illegal characters in a parameter value.",
    -1102: "A mandatory parameter is missing, empty, or malformed.",
    -1104: "Too many parameters — an unexpected parameter was sent.",
    -1111: "Precision is over the maximum defined for this asset (check exchangeInfo filters).",
    -1121: "Invalid symbol — use the exact Binance symbol (e.g. BTCUSDT), see binance_get_exchange_info.",
    -1127: "startTime/endTime span too wide — this endpoint caps the window (24 h for trades/orders); slice it.",
    -1128: "Optional parameter combination is invalid (e.g. startTime/endTime with fromId).",
    -1130: "Illegal parameter value for this endpoint.",
    -2010: (
        "Order rejected — insufficient balance, a symbol filter (LOT_SIZE / PRICE_FILTER / NOTIONAL), "
        "or the symbol is not trading. Validate with the order dry-run first."
    ),
    -2011: "Unknown order — the orderId/origClientOrderId does not exist for that symbol (already filled or cancelled?).",
    -2013: "Order does not exist.",
    -2014: "API-key format invalid.",
    -2015: (
        "Invalid API-key, IP, or permissions for this action — the key lacks the permission "
        "(Reading / Spot & Margin Trading), or this machine's IP is not on the key's allowlist."
    ),
    -2021: "Order would immediately match and take (post-only / GTX rejected).",
    -2022: "Order would immediately trigger.",
    -2026: "Order archived — orders older than 90 days with no fill are no longer queryable.",
    -4001: "Invalid parameter for this wallet endpoint.",
}


def _detail_from_body(resp: httpx.Response) -> tuple[int | None, str]:
    """Extract Binance's (`code`, `msg`) from a response body."""
    try:
        body = resp.json()
    except ValueError:
        return None, resp.text[:300]
    if isinstance(body, dict):
        code = body.get("code")
        msg = body.get("msg") or body.get("message") or ""
        code_int = code if isinstance(code, int) else None
        if code_int is not None and msg:
            return code_int, f"{msg} (code {code_int})"
        if msg:
            return code_int, str(msg)
    return None, resp.text[:300]


def handle_api_error(exc: Exception) -> str:
    """Map an exception to a human-readable `Error ...` string for the LLM.

    Tools never raise: every tool body is wrapped in try/except and returns
    this string on failure.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        code, detail = _detail_from_body(exc.response)
        hint = _CODE_HINTS.get(code, "") if code is not None else ""
        suffix = f" {hint}" if hint else ""
        retry_after = exc.response.headers.get("retry-after")
        if status == 400:
            return f"Error (400): Bad request – {detail}.{suffix or ' Double-check parameter values.'}"
        if status == 401:
            return f"Error (401): Unauthorized – {detail}.{suffix or ' The BINANCE_API_KEY is missing, invalid, or revoked.'}"
        if status == 403:
            return f"Error (403): Forbidden – {detail}. The WAF rejected the request or the key lacks this permission."
        if status == 404:
            return (
                f"Error (404): Not found – {detail}. The endpoint does not exist on this base URL "
                "(the spot testnet has no /sapi endpoints)."
            )
        if status == 409:
            return f"Error (409): Partial success – {detail}. For cancelReplace: the cancel and the new order diverged."
        if status == 418:
            wait = f" Retry-After: {retry_after}s." if retry_after else ""
            return f"Error (418): IP auto-banned for ignoring 429s – {detail}.{wait} Stop all requests until it lifts."
        if status == 429:
            wait = f" Retry-After: {retry_after}s." if retry_after else ""
            return f"Error (429): Rate limited – {detail}.{wait} Back off; do not retry in a tight loop."
        if status >= 500:
            return (
                f"Error ({status}): Binance-side failure – {detail}. Execution status is UNKNOWN: for an order, "
                "check open orders / order status before retrying to avoid a duplicate."
            )
        return f"Error ({status}): {detail}{suffix}"
    if isinstance(exc, httpx.TimeoutException):
        return (
            "Error: the Binance API request timed out. Execution status is UNKNOWN for a mutating call — check "
            "before retrying. Raise BINANCE_REQUEST_TIMEOUT_SECONDS if this recurs."
        )
    if isinstance(exc, httpx.ConnectError):
        return "Error: could not connect to the Binance API. Check network access and BINANCE_API_URL."
    if isinstance(exc, RuntimeError):
        return f"Error: {exc}"
    return f"Error: unexpected failure – {type(exc).__name__}: {exc}"
