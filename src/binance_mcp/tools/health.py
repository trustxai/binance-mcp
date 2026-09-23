"""Health-check tool — the in-repo exemplar of the house tool style.

Wave workers: copy this module's shape (decorator, annotations, docstring sections,
try/except -> handle_api_error, `-> str` return). It exercises all three auth modes
of the client: `none` (ping/time), and `signed` (API-key restrictions).
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from mcp.types import ToolAnnotations

from binance_mcp.client import get_client
from binance_mcp.config import get_settings
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import epoch_to_human
from binance_mcp.server import mcp


def _format_restrictions(data: dict[str, Any]) -> list[str]:
    """Render `GET /sapi/v1/account/apiRestrictions`, flagging risky flags."""
    withdrawals = bool(data.get("enableWithdrawals"))
    lines = [
        f"- **reading**: {bool(data.get('enableReading'))}",
        f"- **spot & margin trading**: {bool(data.get('enableSpotAndMarginTrading'))}",
        f"- **withdrawals**: {withdrawals}"
        + (
            "  ⚠️ should be OFF — this server never withdraws, the key should not be able to either"
            if withdrawals
            else ""
        ),
        f"- **universal transfer**: {bool(data.get('permitsUniversalTransfer'))}",
        f"- **IP restricted**: {bool(data.get('ipRestrict'))}"
        + ("" if data.get("ipRestrict") else "  ⚠️ add an IP allowlist to the key"),
        f"- **futures / margin**: {bool(data.get('enableFutures'))} / {bool(data.get('enableMargin'))}",
        f"- **key created**: {epoch_to_human(data.get('createTime'))}",
    ]
    return lines


@mcp.tool(
    name="binance_health_check",
    annotations=ToolAnnotations(
        title="Binance Health Check",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_health_check() -> str:
    """Verify connectivity, clock drift, and API-key permissions against Binance.

    Calls `GET /api/v3/ping` and `GET /api/v3/time` (public), then — when
    credentials are configured — `GET /sapi/v1/account/apiRestrictions` (signed)
    to report what the key is allowed to do. Also states whether the trading
    kill-switch (BINANCE_ALLOW_TRADING) is on.

    When to Use:
    - As the first call after configuring the server, to confirm the key signs correctly.
    - To debug -1021 (clock drift), -1022 (signature), or -2015 (permissions / IP) errors.

    When NOT to Use:
    - To read balances (use the spot/wallet account tools).

    Returns:
    A markdown block with connectivity, server time vs local drift, the key's
    permission flags (withdrawals should be OFF, IP restriction ON), and the
    kill-switch state — or an `Error ...` string describing the failure.

    Error Handling:
    -2015 means the key's IP allowlist excludes this machine or the key lacks
    Reading; -1022 means the secret / key type is wrong; on the spot testnet the
    /sapi call is skipped because the testnet has no wallet endpoints.
    """
    try:
        settings = get_settings()
        client = get_client()
        await client.request("GET", "/api/v3/ping")
        time_resp = await client.request("GET", "/api/v3/time")
        server_ms = int(time_resp.json().get("serverTime", 0))
        drift_ms = int(time.time() * 1000) - server_ms
        lines = [
            "# Binance health",
            "",
            f"- **base URL**: {settings.base_url}{' (spot testnet)' if settings.binance_testnet else ''}",
            "- **connectivity**: OK",
            f"- **server time**: {epoch_to_human(server_ms)} (local clock drift {drift_ms:+d} ms; "
            f"recvWindow {settings.binance_recv_window_ms} ms)",
            f"- **trading kill-switch**: {'ENABLED — orders/transfers allowed' if settings.binance_allow_trading else 'disabled (read-only; set BINANCE_ALLOW_TRADING=1 to trade)'}",
        ]
        if not settings.has_credentials:
            lines.append("- **credentials**: none configured — public market data only.")
            return "\n".join(lines)
        if settings.binance_testnet:
            lines.append("- **API key**: configured (permissions check skipped — the spot testnet has no /sapi).")
            return "\n".join(lines)
        try:
            resp = await client.request("GET", "/sapi/v1/account/apiRestrictions", auth="signed")
        except httpx.HTTPStatusError as exc:
            lines.append(f"- **API key**: signing or permission problem → {handle_api_error(exc)}")
            return "\n".join(lines)
        lines.append("- **API key permissions**:")
        lines.extend(_format_restrictions(resp.json()))
        if client.last_used_weight_1m is not None:
            lines.append(f"- **used weight (1m)**: {client.last_used_weight_1m}")
        return "\n".join(lines)
    except Exception as exc:
        return handle_api_error(exc)
