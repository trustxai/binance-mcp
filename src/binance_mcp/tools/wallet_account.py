"""Wallet account + system tools (inventory F): status, permissions, snapshots.

Seven read-only tools over `/sapi/v1/account/*`, `/sapi/v1/accountSnapshot`,
`/sapi/v1/system/status` and `/sapi/v1/spot/delist-schedule`. All are SIGNED except
`binance_get_system_status` (NONE — no key needed) and `binance_get_delist_schedule`
(API key only, no signature). None of these exist on the spot testnet (`/sapi` is not
served there).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from binance_mcp.client import get_client
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import ResponseFormat, clip_response, epoch_to_human, fmt_num, to_json
from binance_mcp.server import mcp

# Context-window guard (rule 8): cap displayed rows on top of whatever the API returns.
MAX_DISPLAY_ROWS = 50

# `accountSnapshot` rejects windows of 30 days or more ("Support query within the last
# one month only"); validated client-side before the call so the caller gets a clear
# message instead of Binance's generic one.
_SNAPSHOT_WINDOW_MS = 30 * 24 * 60 * 60 * 1000


def _to_ms(value: int | str | None, field: str) -> int | None:
    """Accept an epoch-ms int or an ISO-8601 string on the tool surface; send ms.

    A numeric string is only treated as an epoch-ms integer when it has at least 12
    digits (a real ms timestamp is 13 digits today); anything shorter is almost
    certainly a malformed date and is parsed as ISO-8601 instead, so a typo surfaces
    a clear error rather than silently becoming a bogus timestamp.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = value.strip()
    if text.isdigit() and len(text) >= 12:
        return int(text)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RuntimeError(
            f"{field} must be an epoch-ms integer (>= 12 digits) or an ISO-8601 date/datetime "
            f"(e.g. '2026-09-01' or '2026-09-01T00:00:00Z'); got {value!r}."
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _decimal(value: Any) -> Decimal:
    """Safe Decimal parse for zero-filtering/sorting; malformed values count as zero."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        return Decimal(0)


def _format_restrictions(data: dict[str, Any]) -> list[str]:
    """Render `GET /sapi/v1/account/apiRestrictions`, flagging risky flags.

    Duplicated from `tools/health.py` (not imported — wave modules stay independent)
    with the identical warning wording, extended to the full flag list Binance returns.
    """
    withdrawals = bool(data.get("enableWithdrawals"))
    return [
        f"- **reading**: {bool(data.get('enableReading'))}",
        f"- **spot & margin trading**: {bool(data.get('enableSpotAndMarginTrading'))}",
        f"- **withdrawals**: {withdrawals}"
        + (
            "  ⚠️ should be OFF — this server never withdraws, the key should not be able to either"
            if withdrawals
            else ""
        ),
        f"- **internal transfer**: {bool(data.get('enableInternalTransfer'))}",
        f"- **universal transfer**: {bool(data.get('permitsUniversalTransfer'))}",
        f"- **IP restricted**: {bool(data.get('ipRestrict'))}"
        + ("" if data.get("ipRestrict") else "  ⚠️ add an IP allowlist to the key"),
        f"- **futures / margin**: {bool(data.get('enableFutures'))} / {bool(data.get('enableMargin'))}",
        f"- **portfolio margin trading**: {bool(data.get('enablePortfolioMarginTrading'))}",
        f"- **vanilla options**: {bool(data.get('enableVanillaOptions'))}",
        f"- **key created**: {epoch_to_human(data.get('createTime'))}",
        f"- **trading authority expires**: {epoch_to_human(data.get('tradingAuthorityExpirationTime'))}",
    ]


class _ReadInput(BaseModel):
    """Shared config for the no-argument (or near-no-argument) read tools below."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: `markdown` (human-readable) or `json` (raw Binance payload).",
    )


class AccountStatusInput(_ReadInput):
    """`GET /sapi/v1/account/status` takes only timestamp+signature — no extra params."""


class ApiTradingStatusInput(_ReadInput):
    """`GET /sapi/v1/account/apiTradingStatus` takes only timestamp+signature."""


class ApiRestrictionsInput(_ReadInput):
    """`GET /sapi/v1/account/apiRestrictions` takes only timestamp+signature."""


class AccountInfoInput(_ReadInput):
    """`GET /sapi/v1/account/info` takes only timestamp+signature."""


class SystemStatusInput(_ReadInput):
    """`GET /sapi/v1/system/status` takes no params at all (not even a key)."""


class DelistScheduleInput(_ReadInput):
    """`GET /sapi/v1/spot/delist-schedule` takes only an optional recvWindow; Binance
    does not support filtering this endpoint by symbol."""


class SnapshotType(StrEnum):
    """Which wallet `binance_get_account_snapshot` reads a daily snapshot of."""

    SPOT = "SPOT"
    MARGIN = "MARGIN"
    FUTURES = "FUTURES"


class AccountSnapshotInput(_ReadInput):
    """Params for `GET /sapi/v1/accountSnapshot`."""

    type: SnapshotType = Field(description="Which wallet to snapshot: SPOT, MARGIN or FUTURES.")
    start_time: int | str | None = Field(
        default=None,
        description=(
            "Window start (inclusive): epoch ms or an ISO-8601 date/datetime. When given together "
            "with end_time, the span must be under 30 days (Binance only retains ~1 month)."
        ),
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end (inclusive): epoch ms or an ISO-8601 date/datetime. Defaults to now.",
    )
    limit: int = Field(
        default=7,
        ge=7,
        le=30,
        description="Number of daily snapshots to return, 7-30 (Binance default 7).",
    )


@mcp.tool(
    name="binance_get_account_status",
    annotations=ToolAnnotations(
        title="Binance Account Status",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_account_status(params: AccountStatusInput) -> str:
    """Report whether the account is in good standing with Binance.

    Calls `GET /sapi/v1/account/status` (SIGNED, IP weight 1). Binance flags accounts
    that trip abuse/AML controls (e.g. excessive order cancellation) here; a healthy
    account reports "Normal".

    When to Use:
    - Before an automated trading run, to confirm the account is not under review.
    - Alongside `binance_get_api_trading_status` when diagnosing rejected orders.

    When NOT to Use:
    - To read API-key permission flags — use `binance_get_api_restrictions`.
    - To read trading-specific locks/triggers — use `binance_get_api_trading_status`.

    Returns:
    A one-line markdown status, or the raw JSON envelope with `response_format="json"`.

    Examples:
        params = {}
        params = {"response_format": "json"}

    Error Handling:
    -2015 means the key lacks Reading permission or this IP is not allowlisted.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/account/status", auth="signed")
        data = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        status = data.get("data", "Unknown")
        return clip_response(f"# Binance account status\n\n- **status**: {status}")
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_api_trading_status",
    annotations=ToolAnnotations(
        title="Binance API Trading Status",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_api_trading_status(params: ApiTradingStatusInput) -> str:
    """Report whether spot trading is locked and what triggered it.

    Calls `GET /sapi/v1/account/apiTradingStatus` (SIGNED, IP weight 1). When Binance's
    abuse-prevention system trips (excessive cancel ratio, etc.) it locks trading for a
    cooldown window; this reports the lock state, the recovery ETA, and which trigger
    fired.

    When to Use:
    - When order placement starts failing for no obvious filter/balance reason, to
      check for a temporary account-wide trading lock.

    When NOT to Use:
    - To check the account's general standing — use `binance_get_account_status`.
    - To check key permission flags — use `binance_get_api_restrictions`.

    Returns:
    A markdown block with `isLocked`, the planned recovery time (rendered UTC), and the
    trigger-condition thresholds (GCR = GTC cancellation ratio, IFER = IOC/FOK
    expiration ratio, UFR = unfilled ratio), or raw JSON with `response_format="json"`.

    Examples:
        params = {}

    Error Handling:
    -2015 means the key lacks Reading permission or this IP is not allowlisted.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/account/apiTradingStatus", auth="signed")
        data = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        payload = data.get("data", {})
        trigger = payload.get("triggerCondition", {})
        lines = [
            "# Binance API trading status",
            "",
            f"- **locked**: {bool(payload.get('isLocked'))}",
            f"- **planned recovery**: {epoch_to_human(payload.get('plannedRecoverTime'))}",
            f"- **updated**: {epoch_to_human(payload.get('updateTime'))}",
            "- **trigger thresholds**:",
            f"  - GCR (GTC cancellation ratio): {trigger.get('GCR', 'N/A')}",
            f"  - IFER (IOC/FOK expiration ratio): {trigger.get('IFER', 'N/A')}",
            f"  - UFR (unfilled ratio): {trigger.get('UFR', 'N/A')}",
        ]
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_api_restrictions",
    annotations=ToolAnnotations(
        title="Binance API Key Restrictions",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_api_restrictions(params: ApiRestrictionsInput) -> str:
    """Report the full permission flag set on the configured API key.

    Calls `GET /sapi/v1/account/apiRestrictions` (SIGNED, IP weight 1) and renders
    every flag Binance returns, with the same warning wording as `binance_health_check`
    for a withdrawals-enabled or no-IP-allowlist key.

    When to Use:
    - To audit a key end-to-end — this is the full flag list.
    - Before enabling BINANCE_ALLOW_TRADING, to confirm trading is permitted and
      withdrawals are off.

    When NOT to Use:
    - For a quick post-startup connectivity+permissions check — use
      `binance_health_check`, which already calls this endpoint as part of a broader
      check.

    Returns:
    A markdown flag list (⚠️ next to anything risky), or raw JSON with
    `response_format="json"`.

    Examples:
        params = {}

    Error Handling:
    -2015 means the key lacks Reading permission or this IP is not allowlisted.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/account/apiRestrictions", auth="signed")
        data = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        lines = ["# Binance API key restrictions", ""]
        lines.extend(_format_restrictions(data))
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_account_info",
    annotations=ToolAnnotations(
        title="Binance Account Info",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_account_info(params: AccountInfoInput) -> str:
    """Report the account's VIP tier and which product lines are enabled.

    Calls `GET /sapi/v1/account/info` (SIGNED, IP weight 1).

    When to Use:
    - To check the account's VIP fee tier, or whether margin/futures/options are
      enabled before routing a request that assumes one of them.

    When NOT to Use:
    - To read balances or trading permissions — use `binance_get_spot_account`
      (spot_account.py) or `binance_get_api_restrictions`.

    Returns:
    A markdown block with `vipLevel` and the isMarginEnabled/isFutureEnabled/
    isOptionsEnabled/isPortfolioMarginRetailEnabled flags, or raw JSON with
    `response_format="json"`.

    Examples:
        params = {}

    Error Handling:
    -2015 means the key lacks Reading permission or this IP is not allowlisted.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/account/info", auth="signed")
        data = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        lines = [
            "# Binance account info",
            "",
            f"- **VIP level**: {data.get('vipLevel', 'N/A')}",
            f"- **margin enabled**: {bool(data.get('isMarginEnabled'))}",
            f"- **futures enabled**: {bool(data.get('isFutureEnabled'))}",
            f"- **options enabled**: {bool(data.get('isOptionsEnabled'))}",
            f"- **portfolio margin (retail) enabled**: {bool(data.get('isPortfolioMarginRetailEnabled'))}",
        ]
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_account_snapshot",
    annotations=ToolAnnotations(
        title="Binance Account Snapshot",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_account_snapshot(params: AccountSnapshotInput) -> str:
    """Return daily balance snapshots for the SPOT, MARGIN or FUTURES wallet.

    Calls `GET /sapi/v1/accountSnapshot` (SIGNED, **IP weight 2400** — a fifth of the
    12000/min `/sapi` IP budget in a single call; call this sparingly, never in a tight
    loop). Binance only retains roughly the last month of snapshots and rejects windows
    of 30 days or more, so `start_time`/`end_time` are validated locally before the
    call to fail fast with a clear message.

    When to Use:
    - To reconstruct a historical balance curve ("what was I holding a week ago").
    - As an occasional, deliberate call — not for polling the current balance.

    When NOT to Use:
    - For the current live balance — use `binance_get_spot_account` (spot_account.py),
      which is far cheaper (IP weight 20) and reflects right now, not yesterday.

    Returns:
    A markdown block per snapshot day (UTC date, totalAssetOfBtc, and a table of
    non-zero balances sorted largest-first, capped at 50 rows per day), or raw JSON
    with `response_format="json"`.

    Windows:
    `start_time`/`end_time` (int ms or ISO-8601) together must span less than 30 days;
    with only `start_time` given, `end_time` defaults to now and the same 30-day span
    check applies. Either way, `start_time` itself must be within the last 30 days —
    Binance does not retain snapshots older than that, regardless of window width.
    Omit both to get the most recent `limit` days. `limit` is 7-30 (Binance default 7).

    Examples:
        params = {"type": "SPOT"}
        params = {"type": "SPOT", "start_time": "2026-09-01", "end_time": "2026-09-10", "limit": 10}

    Error Handling:
    A window of 30 days or more, or a `start_time` more than 30 days ago, is rejected
    locally instead of round-tripping to Binance's "Support query within the last one
    month only". This endpoint answers HTTP 200 with `{code, msg, snapshotVos}` on
    failure (no `success` field, so the client's envelope check does not catch it) — a
    non-200 `code` is surfaced as an `Error:` here. -2015 means the key lacks Reading
    permission or this IP is not allowlisted.
    """
    try:
        start_ms = _to_ms(params.start_time, "start_time")
        end_ms = _to_ms(params.end_time, "end_time")
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        if start_ms is not None:
            effective_end_ms = end_ms if end_ms is not None else now_ms
            if effective_end_ms <= start_ms:
                return "Error: end_time must be after start_time."
            if effective_end_ms - start_ms >= _SNAPSHOT_WINDOW_MS:
                days = (effective_end_ms - start_ms) / (24 * 60 * 60 * 1000)
                return (
                    f"Error: the start_time/end_time window spans ~{days:.1f} days; Binance's "
                    "accountSnapshot only supports windows under 30 days (and only the last "
                    "month of history). Narrow the window."
                )
            if now_ms - start_ms >= _SNAPSHOT_WINDOW_MS:
                days_ago = (now_ms - start_ms) / (24 * 60 * 60 * 1000)
                return (
                    f"Error: start_time is ~{days_ago:.1f} days in the past; Binance's accountSnapshot "
                    "only retains the last 30 days of history, regardless of window width. Use a more "
                    "recent start_time."
                )
        query_params: dict[str, Any] = {"type": params.type.value, "limit": params.limit}
        if start_ms is not None:
            query_params["startTime"] = start_ms
        if end_ms is not None:
            query_params["endTime"] = end_ms
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/accountSnapshot", auth="signed", params=query_params)
        data = resp.json()
        code = data.get("code")
        if code not in (None, 200):
            return f"Error: {data.get('msg')} (code {code})"
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        snapshots = data.get("snapshotVos") or []
        lines = [f"# Binance {params.type.value} account snapshot", ""]
        if not snapshots:
            lines.append("_No snapshots in range._")
            return clip_response("\n".join(lines))
        for snap in snapshots:
            day_data = snap.get("data", {})
            lines.append(f"## {epoch_to_human(snap.get('updateTime'))}")
            total_btc = day_data.get("totalAssetOfBtc")
            if total_btc is not None:
                lines.append(f"- **total (BTC)**: {fmt_num(total_btc)}")
            balances = [
                b
                for b in day_data.get("balances", [])
                if _decimal(b.get("free")) != 0 or _decimal(b.get("locked")) != 0
            ]
            balances.sort(key=lambda b: _decimal(b.get("free")) + _decimal(b.get("locked")), reverse=True)
            if balances:
                lines.append("| asset | free | locked |")
                lines.append("|---|---|---|")
                for bal in balances[:MAX_DISPLAY_ROWS]:
                    lines.append(f"| {bal.get('asset')} | {fmt_num(bal.get('free'))} | {fmt_num(bal.get('locked'))} |")
                if len(balances) > MAX_DISPLAY_ROWS:
                    lines.append(f"_[{len(balances) - MAX_DISPLAY_ROWS} more asset(s) not shown]_")
            else:
                lines.append("_all balances zero_")
            lines.append("")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_system_status",
    annotations=ToolAnnotations(
        title="Binance System Status",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_system_status(params: SystemStatusInput) -> str:
    """Report whether the Binance system is up or under maintenance.

    Calls `GET /sapi/v1/system/status` (NONE — no key or signature needed, IP weight 1).

    When to Use:
    - Before assuming a failure is account-specific — rule out a Binance-wide
      maintenance window first.

    When NOT to Use:
    - To check THIS key's connectivity/permissions — use `binance_health_check`.

    Returns:
    `normal` or `maintenance` plus Binance's message, or raw JSON with
    `response_format="json"`.

    Examples:
        params = {}

    Error Handling:
    This endpoint needs no credentials at all; a failure here means Binance itself is
    unreachable, not a key or signature problem.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/system/status", auth="none")
        data = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        status_code = data.get("status")
        status_label = {0: "normal", 1: "maintenance"}.get(status_code, f"unknown ({status_code})")
        lines = ["# Binance system status", "", f"- **status**: {status_label}"]
        if data.get("msg"):
            lines.append(f"- **message**: {data['msg']}")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_delist_schedule",
    annotations=ToolAnnotations(
        title="Binance Delist Schedule",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_delist_schedule(params: DelistScheduleInput) -> str:
    """List symbols scheduled to be delisted, with their delisting date.

    Calls `GET /sapi/v1/spot/delist-schedule` (API key only — no signature, IP weight
    100). Useful to avoid opening new positions in a symbol about to stop trading.

    When to Use:
    - Before placing a new order, to check the symbol is not on the delist schedule.
    - As a periodic sweep of open positions against upcoming delistings.

    When NOT to Use:
    - To check whether a symbol is trading right now — use `binance_get_exchange_info`
      (market_data.py) and read its `status` field.

    Returns:
    A markdown table of delist date → symbols, capped at 50 rows, or raw JSON with
    `response_format="json"`.

    Examples:
        params = {}

    Error Handling:
    -2015 means the key is missing/invalid or this IP is not allowlisted (this
    endpoint only needs the X-MBX-APIKEY header, not a signature).
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/spot/delist-schedule", auth="key")
        data = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        rows = data if isinstance(data, list) else []
        lines = ["# Binance delist schedule", ""]
        if not rows:
            lines.append("_No symbols currently scheduled for delisting._")
            return clip_response("\n".join(lines))
        lines.append("| delist time | symbols |")
        lines.append("|---|---|")
        for row in rows[:MAX_DISPLAY_ROWS]:
            symbols = row.get("symbols")
            if symbols is None and row.get("symbol") is not None:
                symbols = [row["symbol"]]  # older docs show a singular `symbol` shape
            lines.append(f"| {epoch_to_human(row.get('delistTime'))} | {', '.join(symbols or [])} |")
        if len(rows) > MAX_DISPLAY_ROWS:
            lines.append("")
            lines.append(f"_[{len(rows) - MAX_DISPLAY_ROWS} more row(s) not shown]_")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)
