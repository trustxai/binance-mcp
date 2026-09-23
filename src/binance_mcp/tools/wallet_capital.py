"""Wallet capital tools (inventory D): deposits, withdrawals, addresses, coin config.

Seven SIGNED read-only tools over `/sapi/v1/capital/*`. This module answers the
question the rest of the server cannot: **every deposit and withdrawal since the
account was created** — `binance_get_all_deposits` / `binance_get_all_withdrawals` walk
Binance's 90-day window cap backwards in 89-day slices, paging `offset` inside each
window, until they reach `since` or run out of call budget.

There is deliberately **no withdrawal tool**. `POST /sapi/v1/capital/withdraw/apply`
is on the client's `FORBIDDEN_PATHS` and raises `ForbiddenEndpointError` whatever the
flags say; nothing here can express a withdrawal.

None of these endpoints exist on the spot testnet (`/sapi` is not served there).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

import httpx
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, field_validator

from binance_mcp.client import get_client
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import ResponseFormat, clip_response, epoch_to_human, fmt_num, to_json
from binance_mcp.server import mcp

# Context-window guard (rule 8): markdown rows are capped here on top of the API limit;
# the JSON format still carries the full set (through clip_response).
MAX_DISPLAY_ROWS = 100

# `config/getall` returns every listed coin (~400); cap the markdown rendering.
MAX_COINS = 50

_DAY_MS = 24 * 60 * 60 * 1000

# Binance rejects a deposit/withdraw window of 90 days or more ("must be less than 90
# days"); `withdrawOrderId` narrows that to 7 days. Enforced locally, before the call.
_WINDOW_CAP_MS = 90 * _DAY_MS
_ORDER_ID_WINDOW_CAP_MS = 7 * _DAY_MS

# The walks slice the span into 89-day windows — one day under the cap, so a boundary
# rounding difference on Binance's side can never trip -1127.
_WALK_WINDOW_MS = 89 * _DAY_MS
# Floor for the defensive bisect below: never narrow a window past a single day.
_MIN_WALK_WINDOW_MS = _DAY_MS

# `limit` maxes out at 1000 on both history endpoints; the walks always ask for the max
# and page `offset` by it until a short page proves the window is drained.
_PAGE_LIMIT = 1000

# Binance opened for business in July 2017. There is NO endpoint that reports when an
# account was created: `apiRestrictions.createTime` is the API KEY's creation date, not
# the account's. So "since the beginning" means "since the exchange existed".
BINANCE_LAUNCH = "2017-07-01"

_WALLET_TYPES: dict[int, str] = {0: "spot", 1: "funding"}


class DepositStatus(StrEnum):
    """`status` filter for `GET /sapi/v1/capital/deposit/hisrec` (sent as its code)."""

    PENDING = "pending"  # 0
    SUCCESS = "success"  # 1
    REJECTED = "rejected"  # 2
    CREDITED_CANNOT_WITHDRAW = "credited_cannot_withdraw"  # 6
    WRONG_DEPOSIT = "wrong_deposit"  # 7
    AWAITING_CONFIRMATION = "awaiting_confirmation"  # 8


class WithdrawStatus(StrEnum):
    """`status` filter for `GET /sapi/v1/capital/withdraw/history` (sent as its code)."""

    EMAIL_SENT = "email_sent"  # 0
    CANCELLED = "cancelled"  # 1
    AWAITING_APPROVAL = "awaiting_approval"  # 2
    REJECTED = "rejected"  # 3
    PROCESSING = "processing"  # 4
    FAILURE = "failure"  # 5
    COMPLETED = "completed"  # 6


_DEPOSIT_STATUS_CODES: dict[DepositStatus, int] = {
    DepositStatus.PENDING: 0,
    DepositStatus.SUCCESS: 1,
    DepositStatus.REJECTED: 2,
    DepositStatus.CREDITED_CANNOT_WITHDRAW: 6,
    DepositStatus.WRONG_DEPOSIT: 7,
    DepositStatus.AWAITING_CONFIRMATION: 8,
}
_WITHDRAW_STATUS_CODES: dict[WithdrawStatus, int] = {
    WithdrawStatus.EMAIL_SENT: 0,
    WithdrawStatus.CANCELLED: 1,
    WithdrawStatus.AWAITING_APPROVAL: 2,
    WithdrawStatus.REJECTED: 3,
    WithdrawStatus.PROCESSING: 4,
    WithdrawStatus.FAILURE: 5,
    WithdrawStatus.COMPLETED: 6,
}
_DEPOSIT_STATUS_NAMES: dict[int, str] = {code: name.value for name, code in _DEPOSIT_STATUS_CODES.items()}
_WITHDRAW_STATUS_NAMES: dict[int, str] = {code: name.value for name, code in _WITHDRAW_STATUS_CODES.items()}


# -- small shared helpers -------------------------------------------------------------


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _to_ms(value: int | str | None, field_name: str) -> int | None:
    """Accept an epoch-ms int or an ISO-8601 string on the tool surface; send ms.

    A numeric string is only treated as an epoch-ms integer when it has at least 12
    digits (a real ms timestamp is 13 digits today); anything shorter is almost
    certainly a malformed date and is parsed as ISO-8601 instead, so a typo surfaces a
    clear error rather than silently becoming a bogus timestamp.
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
            f"{field_name} must be an epoch-ms integer (>= 12 digits) or an ISO-8601 date/datetime "
            f"(e.g. '2017-07-01' or '2017-07-01T00:00:00Z'); got {value!r}."
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _to_ms_required(value: int | str, field_name: str) -> int:
    """`_to_ms` for a field that always has a value (narrows the type for mypy)."""
    ms = _to_ms(value, field_name)
    if ms is None:  # pragma: no cover - `value` is never None at the call sites
        raise RuntimeError(f"{field_name} is required.")
    return ms


def _row_ms(value: Any) -> int:
    """Best-effort epoch-ms for sorting. Withdraw rows carry `applyTime` as a
    `"YYYY-MM-DD HH:MM:SS"` STRING on older records, not an epoch — handle both."""
    if value in (None, "", 0, "0"):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(text, fmt).replace(tzinfo=UTC).timestamp() * 1000)
        except ValueError:
            continue
    return 0


def _decimal(value: Any) -> Decimal:
    """Safe Decimal parse for totals; malformed values count as zero."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        return Decimal(0)


def _wallet_label(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return _WALLET_TYPES.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def _status_label(value: Any, names: dict[int, str]) -> str:
    try:
        code = int(value)
    except (TypeError, ValueError):
        return "N/A" if value is None else str(value)
    return f"{names.get(code, 'unknown')} ({code})"


def _resolve_window(
    start_ms: int | None,
    end_ms: int | None,
    *,
    cap_ms: int,
    cap_label: str,
    hint: str,
) -> tuple[dict[str, int], str | None]:
    """Validate (and one-sidedly complete) a start/end window before any request.

    Returns `(params, error)`. Binance's rule is "if both are sent the interval must be
    less than <cap>"; the one-sided cases are resolved the way Binance itself would,
    then validated, so an impossible request fails here instead of round-tripping:

    - both given  -> validated, both sent;
    - start only  -> `endTime` defaults to now on Binance's side, so the span
      `start..now` is validated; only `startTime` is sent (no clock-skew surprise);
    - end only    -> Binance would default `startTime` to 90 days before **now**, which
      for an old `end_time` yields an empty result for no obvious reason. A `startTime`
      of `end - (cap - 1 day)` is sent instead, so the window actually brackets the
      requested end;
    - neither     -> nothing sent; Binance returns its own last-90-days default.
    """
    if start_ms is not None and end_ms is not None:
        if end_ms <= start_ms:
            return {}, "Error: end_time must be after start_time."
        if end_ms - start_ms >= cap_ms:
            days = (end_ms - start_ms) / _DAY_MS
            return {}, (
                f"Error: the start_time/end_time window spans ~{days:.1f} days; Binance requires this "
                f"window to be under {cap_label}. Narrow the window{hint}."
            )
        return {"startTime": start_ms, "endTime": end_ms}, None
    if start_ms is not None:
        span = _now_ms() - start_ms
        if span >= cap_ms:
            days = span / _DAY_MS
            return {}, (
                f"Error: start_time is ~{days:.1f} days ago and end_time defaults to now, so the window "
                f"would span {cap_label} or more, which Binance rejects. Pass end_time as well{hint}."
            )
        return {"startTime": start_ms}, None
    if end_ms is not None:
        return {"startTime": end_ms - (cap_ms - _DAY_MS), "endTime": end_ms}, None
    return {}, None


def _normalise_asset(value: str) -> str:
    """Uppercase and check a bare ASSET code (`BTC`, and the real one-letter assets `W`/`S`)."""
    upper = value.strip().upper()
    if not upper or len(upper) > 20 or not upper.isalnum():
        raise ValueError(f"asset code must be 1-20 alphanumeric characters (e.g. 'BTC'); got {value!r}")
    return upper


def _asset_validator(value: str | None) -> str | None:
    """Optional-field flavour of `_normalise_asset`."""
    return None if value is None else _normalise_asset(value)


def _positive_decimal(value: str | None) -> str | None:
    """Amounts travel as STRINGS, verbatim — never floats (precision matters)."""
    if value is None:
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"amount must be a decimal number sent as a string; got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"amount must be greater than zero; got {value!r}")
    return value


# -- rendering ------------------------------------------------------------------------

_DEPOSIT_HEADER = (
    "| inserted | completed | coin | amount | network | status | wallet | txId |\n|---|---|---|---|---|---|---|---|"
)
_WITHDRAW_HEADER = (
    "| applied | completed | coin | amount | fee | network | status | wallet | txId | withdrawOrderId |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def _deposit_row(row: dict[str, Any]) -> str:
    return (
        f"| {epoch_to_human(row.get('insertTime'))} | {epoch_to_human(row.get('completeTime'))} "
        f"| {row.get('coin') or 'N/A'} | {fmt_num(row.get('amount'))} | {row.get('network') or 'N/A'} "
        f"| {_status_label(row.get('status'), _DEPOSIT_STATUS_NAMES)} | {_wallet_label(row.get('walletType'))} "
        f"| `{row.get('txId') or 'N/A'}` |"
    )


def _withdraw_row(row: dict[str, Any]) -> str:
    return (
        f"| {epoch_to_human(row.get('applyTime'))} | {epoch_to_human(row.get('completeTime'))} "
        f"| {row.get('coin') or 'N/A'} | {fmt_num(row.get('amount'))} | {fmt_num(row.get('transactionFee'))} "
        f"| {row.get('network') or 'N/A'} | {_status_label(row.get('status'), _WITHDRAW_STATUS_NAMES)} "
        f"| {_wallet_label(row.get('walletType'))} | `{row.get('txId') or 'N/A'}` "
        f"| {row.get('withdrawOrderId') or 'N/A'} |"
    )


def _totals_by_coin(rows: list[dict[str, Any]], *, with_fees: bool) -> dict[str, dict[str, Any]]:
    """Per-coin count + summed amount (+ summed transactionFee for withdrawals)."""
    totals: dict[str, dict[str, Any]] = {}
    for row in rows:
        coin = str(row.get("coin") or "N/A")
        bucket = totals.setdefault(coin, {"count": 0, "amount": Decimal(0), "fees": Decimal(0)})
        bucket["count"] += 1
        bucket["amount"] += _decimal(row.get("amount"))
        bucket["fees"] += _decimal(row.get("transactionFee"))
    return {
        coin: (
            {"count": b["count"], "amount": fmt_num(b["amount"]), "fees": fmt_num(b["fees"])}
            if with_fees
            else {"count": b["count"], "amount": fmt_num(b["amount"])}
        )
        for coin, b in sorted(totals.items())
    }


def _totals_table(totals: dict[str, dict[str, Any]], *, with_fees: bool) -> list[str]:
    if not totals:
        return []
    lines = ["## Totals by coin", ""]
    if with_fees:
        lines.append("| coin | count | amount | fees |")
        lines.append("|---|---|---|---|")
        for coin, t in totals.items():
            lines.append(f"| {coin} | {t['count']:,} | {t['amount']} | {t['fees']} |")
    else:
        lines.append("| coin | count | amount |")
        lines.append("|---|---|---|")
        for coin, t in totals.items():
            lines.append(f"| {coin} | {t['count']:,} | {t['amount']} |")
    lines.append("")
    return lines


def _history_response(
    *,
    rows: list[dict[str, Any]],
    title: str,
    fmt: ResponseFormat,
    header: str,
    row_formatter: Callable[[dict[str, Any]], str],
    with_fees: bool,
    extra: dict[str, Any] | None = None,
) -> str:
    """Shared markdown/JSON rendering for a single history page."""
    totals = _totals_by_coin(rows, with_fees=with_fees)
    if fmt is ResponseFormat.JSON:
        payload: dict[str, Any] = {"count": len(rows), "totals": totals, "items": rows}
        if extra:
            payload.update(extra)
        return clip_response(to_json(payload))
    lines = [f"# {title}", "", f"**{len(rows):,}** record(s).", ""]
    if not rows:
        lines.append("_No records in range._")
        return clip_response("\n".join(lines))
    lines.extend(_totals_table(totals, with_fees=with_fees))
    lines.append(header)
    for row in rows[:MAX_DISPLAY_ROWS]:
        lines.append(row_formatter(row))
    if len(rows) > MAX_DISPLAY_ROWS:
        lines.append("")
        lines.append(f'_[{len(rows) - MAX_DISPLAY_ROWS} more row(s) not shown — use response_format="json"]_')
    return clip_response("\n".join(lines))


# -- the walk -------------------------------------------------------------------------


@dataclass
class _WalkResult:
    """Outcome of a windowed walk. `resume_before` is the boundary of the next UNFETCHED
    range — the loop's own `window_end` at exit, never derived from the rows."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    calls: int = 0
    resume_before: int | None = None
    note: str | None = None


def _is_span_error(exc: httpx.HTTPStatusError) -> bool:
    """True only for the "your window is too wide" family (-1127 and its wordings).

    Everything else — auth (-2015/401), rate limits (429/418), server failures —
    is NOT tolerated: the walk stops at once rather than burning budget retrying.
    """
    try:
        body = exc.response.json()
    except ValueError:
        return False
    if not isinstance(body, dict):
        return False
    if body.get("code") == -1127:
        return True
    msg = str(body.get("msg") or "").lower()
    return "time interval" in msg or "90 days" in msg or "interval must be" in msg


def _dedupe_key(row: dict[str, Any], id_key: str) -> str:
    value = row.get(id_key)
    if value in (None, ""):
        return "raw:" + to_json(row)
    return f"{id_key}:{value}"


async def _walk_history(
    *,
    path: str,
    id_key: str,
    since_ms: int,
    until_ms: int,
    max_calls: int,
    extra_params: dict[str, Any],
) -> _WalkResult:
    """Walk `until_ms` → `since_ms` in 89-day windows, newest first, paging `offset`.

    Invariants (the two blocking bugs found in the first wave of walk tools):
    - the budget is checked BEFORE every request, and `calls` counts every request that
      was actually issued — including one that then failed inside a page loop;
    - `resume_before` on an early exit is the loop's own `window_end`, i.e. the boundary
      of the next range that was NOT fetched. A mid-window stop therefore re-fetches
      that whole window on resume, which is safe because rows are deduped by `id_key`.
    """
    client = get_client()
    collected: dict[str, dict[str, Any]] = {}
    calls = 0
    window_end = until_ms

    def _stop(note: str) -> _WalkResult:
        return _WalkResult(rows=list(collected.values()), calls=calls, resume_before=window_end, note=note)

    while window_end > since_ms:
        span = min(_WALK_WINDOW_MS, window_end - since_ms)
        window_start = window_end - span
        offset = 0
        while True:
            if calls >= max_calls:
                return _stop(
                    f"Stopped early: the max_calls budget of {max_calls} request(s) was exhausted before "
                    "reaching `since`."
                )
            params = {
                **extra_params,
                "startTime": window_start,
                "endTime": window_end,
                "offset": offset,
                "limit": _PAGE_LIMIT,
            }
            calls += 1
            try:
                resp = await client.request("GET", path, auth="signed", params=params)
            except httpx.HTTPStatusError as exc:
                if _is_span_error(exc) and span > _MIN_WALK_WINDOW_MS:
                    # Tolerated: halve THIS window (keeping its newer half) and retry.
                    # The older half is picked up by the next outer iteration, so the
                    # walk narrows without ever leaving a gap.
                    span = max(_MIN_WALK_WINDOW_MS, span // 2)
                    window_start = window_end - span
                    offset = 0
                    continue
                return _stop(f"Stopped early: {handle_api_error(exc)}")
            except Exception as exc:
                return _stop(f"Stopped early: {handle_api_error(exc)}")
            page = resp.json()
            if not isinstance(page, list):
                page = []
            for row in page:
                if isinstance(row, dict):
                    collected[_dedupe_key(row, id_key)] = row
            if len(page) < _PAGE_LIMIT:
                break
            offset += _PAGE_LIMIT
        window_end = window_start

    return _WalkResult(rows=list(collected.values()), calls=calls, resume_before=None, note=None)


def _walk_response(
    *,
    result: _WalkResult,
    title: str,
    fmt: ResponseFormat,
    header: str,
    row_formatter: Callable[[dict[str, Any]], str],
    with_fees: bool,
    time_key: str,
    since_ms: int,
    until_ms: int,
) -> str:
    """Render a walk: rows newest-first, per-coin totals, and the resume cursor."""
    rows = sorted(result.rows, key=lambda r: _row_ms(r.get(time_key)), reverse=True)
    totals = _totals_by_coin(rows, with_fees=with_fees)
    resume_note = None
    if result.resume_before is not None:
        resume_note = (
            f"{result.note} Resume with `since={since_ms}`, `resume_before={result.resume_before}` "
            f"({epoch_to_human(result.resume_before)}) to continue backwards from where this stopped."
        )

    if fmt is ResponseFormat.JSON:
        return clip_response(
            to_json(
                {
                    "count": len(rows),
                    "truncated": result.resume_before is not None,
                    "since": since_ms,
                    "until": until_ms,
                    "resume_before": result.resume_before,
                    "calls_made": result.calls,
                    "totals": totals,
                    "items": rows,
                }
            )
        )

    lines = [
        f"# {title}",
        "",
        f"**{len(rows):,}** record(s) between {epoch_to_human(since_ms)} and {epoch_to_human(until_ms)}, "
        f"in **{result.calls}** API call(s).",
        "",
    ]
    if resume_note:
        lines.extend([f"> ⚠️ {resume_note}", ""])
    else:
        lines.extend(["_Complete: the walk reached `since`._", ""])
    if not rows:
        lines.append("_No records in range._")
        return clip_response("\n".join(lines))
    lines.extend(_totals_table(totals, with_fees=with_fees))
    lines.append(header)
    for row in rows[:MAX_DISPLAY_ROWS]:
        lines.append(row_formatter(row))
    if len(rows) > MAX_DISPLAY_ROWS:
        lines.append("")
        lines.append(
            f'_[{len(rows) - MAX_DISPLAY_ROWS} more row(s) not shown — use response_format="json" for the full set]_'
        )
    return clip_response("\n".join(lines))


# -- input models ---------------------------------------------------------------------


class _BaseInput(BaseModel):
    """Shared model config + output format selector."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: `markdown` (human-readable) or `json` (raw Binance payload).",
    )


class DepositHistoryInput(_BaseInput):
    """Params for `GET /sapi/v1/capital/deposit/hisrec`."""

    coin: str | None = Field(
        default=None,
        description="Filter by asset code, e.g. 'BTC' or 'USDT'. Omit for every coin.",
    )
    status: DepositStatus | None = Field(
        default=None,
        description=(
            "Filter by deposit state: pending (0), success (1), rejected (2), "
            "credited_cannot_withdraw (6), wrong_deposit (7), awaiting_confirmation (8)."
        ),
    )
    start_time: int | str | None = Field(
        default=None,
        description=(
            "Window start: epoch ms or ISO-8601. With end_time the span must be under 90 days; "
            "alone, it must be under 90 days ago (end_time defaults to now)."
        ),
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end: epoch ms or ISO-8601. Alone, start_time is filled as end_time - 89 days.",
    )
    offset: int = Field(default=0, ge=0, description="Row offset for paging within the window (Binance default 0).")
    limit: int = Field(default=1000, ge=1, le=1000, description="Rows per call, 1-1000 (Binance default 1000).")
    include_source: bool | None = Field(
        default=None,
        description="Include the `sourceAddress` field on each row (Binance `includeSource`).",
    )
    tx_id: str | None = Field(default=None, min_length=1, description="Filter by on-chain transaction id.")

    @field_validator("coin")
    @classmethod
    def _coin_upper(cls, value: str | None) -> str | None:
        return _asset_validator(value)


class WithdrawHistoryInput(_BaseInput):
    """Params for `GET /sapi/v1/capital/withdraw/history`."""

    coin: str | None = Field(default=None, description="Filter by asset code, e.g. 'BTC'. Omit for every coin.")
    status: WithdrawStatus | None = Field(
        default=None,
        description=(
            "Filter by withdrawal state: email_sent (0), cancelled (1), awaiting_approval (2), "
            "rejected (3), processing (4), failure (5), completed (6)."
        ),
    )
    withdraw_order_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Filter by the client-supplied withdrawal id. With this set Binance narrows the window "
            "cap to **7 days** (and defaults to the last 7 days)."
        ),
    )
    id_list: list[str] | None = Field(
        default=None,
        max_length=45,
        description="Up to 45 Binance withdrawal ids; sent as a comma-separated `idList`.",
    )
    start_time: int | str | None = Field(
        default=None,
        description="Window start: epoch ms or ISO-8601. Span cap is 90 days (7 days with withdraw_order_id).",
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end: epoch ms or ISO-8601. Alone, start_time is filled one day inside the cap.",
    )
    offset: int = Field(default=0, ge=0, description="Row offset for paging within the window (Binance default 0).")
    limit: int = Field(default=1000, ge=1, le=1000, description="Rows per call, 1-1000 (Binance default 1000).")

    @field_validator("coin")
    @classmethod
    def _coin_upper(cls, value: str | None) -> str | None:
        return _asset_validator(value)

    @field_validator("id_list")
    @classmethod
    def _ids_non_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("id_list must contain at least one non-empty id")
        return cleaned


class _WalkInput(_BaseInput):
    """Shared surface of the two `since`-walks."""

    coin: str | None = Field(default=None, description="Restrict the walk to one asset code, e.g. 'BTC'.")
    since: int | str = Field(
        default=BINANCE_LAUNCH,
        description=(
            "How far back to walk: epoch ms or ISO-8601. Defaults to 2017-07-01, Binance's launch — "
            "there is no endpoint that reports an account's creation date, so 'everything' means "
            "'since the exchange existed'."
        ),
    )
    until: int | str | None = Field(
        default=None,
        description="Newest edge of the walk: epoch ms or ISO-8601. Defaults to now.",
    )
    resume_before: int | str | None = Field(
        default=None,
        description=(
            "Continue a previous run: pass the `resume_before` cursor it returned. It replaces "
            "`until`, so the walk picks up exactly at the first range that was not fetched."
        ),
    )

    @field_validator("coin")
    @classmethod
    def _coin_upper(cls, value: str | None) -> str | None:
        return _asset_validator(value)


class AllDepositsInput(_WalkInput):
    """Params for the deposit walk."""

    max_calls: int = Field(
        default=60,
        ge=1,
        le=200,
        description="Hard cap on API calls for this walk (IP weight 1 each). 60 covers ~15 years.",
    )


class AllWithdrawalsInput(_WalkInput):
    """Params for the withdrawal walk."""

    max_calls: int = Field(
        default=10,
        ge=1,
        le=60,
        description=(
            "Hard cap on API calls for this walk. Each call costs **UID weight 18000** of the "
            "180000/min budget, so 10 calls is one full minute of budget. Raise it deliberately."
        ),
    )


class DepositAddressInput(_BaseInput):
    """Params for `GET /sapi/v1/capital/deposit/address`."""

    coin: str = Field(description="Asset code to get a deposit address for, e.g. 'BTC' or 'USDT'.")
    network: str | None = Field(
        default=None,
        min_length=1,
        description="Network to deposit over, e.g. 'BSC', 'ETH', 'TRX'. Omit for the coin's default network.",
    )
    amount: str | None = Field(
        default=None,
        description="Intended deposit amount as a decimal STRING (some networks encode it in the address URL).",
    )

    @field_validator("coin")
    @classmethod
    def _coin_upper(cls, value: str) -> str:
        return _normalise_asset(value)

    @field_validator("amount")
    @classmethod
    def _amount_decimal(cls, value: str | None) -> str | None:
        return _positive_decimal(value)


class DepositAddressesInput(_BaseInput):
    """Params for `GET /sapi/v1/capital/deposit/address/list`."""

    coin: str = Field(description="Asset code to list deposit addresses for, e.g. 'BTC'.")
    network: str | None = Field(
        default=None,
        min_length=1,
        description="Restrict the list to one network, e.g. 'BSC'. Omit for every network.",
    )

    @field_validator("coin")
    @classmethod
    def _coin_upper(cls, value: str) -> str:
        return _normalise_asset(value)


class CoinConfigInput(_BaseInput):
    """Params for `GET /sapi/v1/capital/config/getall`."""

    coin: str | None = Field(
        default=None,
        description="Show only this asset code, e.g. 'BTC'. Filtered client-side — the endpoint takes no filter.",
    )

    @field_validator("coin")
    @classmethod
    def _coin_upper(cls, value: str | None) -> str | None:
        return _asset_validator(value)


# -- tools ----------------------------------------------------------------------------


@mcp.tool(
    name="binance_get_deposit_history",
    annotations=ToolAnnotations(
        title="Binance Deposit History",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_deposit_history(params: DepositHistoryInput) -> str:
    """List crypto deposits into the account for one window (up to 90 days).

    Calls `GET /sapi/v1/capital/deposit/hisrec` (SIGNED, IP weight 1). One call answers
    a single window; Binance caps that window at "less than 90 days" and defaults to
    the last 90 days when no times are given. The cap is enforced here, before the
    request, so an over-wide window fails with a readable message instead of -1127.

    When to Use:
    - "Did my USDT deposit land?" — a recent, bounded lookup.
    - Reconciling one month's deposits, or chasing one `tx_id`.

    When NOT to Use:
    - For the full history since the account opened — use `binance_get_all_deposits`,
      which walks these windows for you and returns a resume cursor.
    - For withdrawals — use `binance_get_withdraw_history`.
    - For fiat (card/bank) deposits — those are not here; use `binance_get_fiat_orders`
      (fiat.py).

    Returns:
    A markdown table (insertTime, completeTime, coin, amount, network, status,
    walletType, txId) plus per-coin totals, capped at 100 displayed rows; or the raw
    Binance array with `response_format="json"`.

    Pagination:
    `limit` is 1-1000 (Binance default 1000) and `offset` pages within the window. A
    full page means there is more: re-call with `offset += limit`.

    Windows:
    `start_time`/`end_time` together must span under 90 days. `start_time` alone must be
    under 90 days ago (Binance defaults `endTime` to now). `end_time` alone gets a
    `startTime` of `end_time - 89 days` so the window actually brackets it.

    Examples:
        params = {"coin": "USDT", "status": "success"}
        params = {"start_time": "2026-08-01", "end_time": "2026-09-01"}
        params = {"tx_id": "0xabc...", "response_format": "json"}

    Error Handling:
    An over-wide window is rejected locally. -2015 means the key lacks Reading
    permission or this IP is not allowlisted. `/sapi` does not exist on the spot
    testnet — a 404 there is expected.
    """
    try:
        start_ms = _to_ms(params.start_time, "start_time")
        end_ms = _to_ms(params.end_time, "end_time")
        window, error = _resolve_window(
            start_ms,
            end_ms,
            cap_ms=_WINDOW_CAP_MS,
            cap_label="90 days",
            hint=", or use `binance_get_all_deposits` which walks the windows for you",
        )
        if error:
            return error
        query: dict[str, Any] = {"offset": params.offset, "limit": params.limit, **window}
        if params.coin is not None:
            query["coin"] = params.coin
        if params.status is not None:
            query["status"] = _DEPOSIT_STATUS_CODES[params.status]
        if params.include_source is not None:
            query["includeSource"] = params.include_source
        if params.tx_id is not None:
            query["txId"] = params.tx_id
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/capital/deposit/hisrec", auth="signed", params=query)
        data = resp.json()
        rows = [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
        return _history_response(
            rows=rows,
            title="Binance deposit history",
            fmt=params.response_format,
            header=_DEPOSIT_HEADER,
            row_formatter=_deposit_row,
            with_fees=False,
            extra={"offset": params.offset, "limit": params.limit},
        )
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_withdraw_history",
    annotations=ToolAnnotations(
        title="Binance Withdrawal History",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_withdraw_history(params: WithdrawHistoryInput) -> str:
    """List crypto withdrawals out of the account for one window (up to 90 days).

    Calls `GET /sapi/v1/capital/withdraw/history` (SIGNED). This is a **read** — this
    server can never submit a withdrawal: `POST /sapi/v1/capital/withdraw/apply` is on
    the client's forbidden-path list and no tool exists for it.

    ⚠️ **Cost: UID weight 18000 per call** (a tenth of the 180000/min per-account
    budget) **and a hard limit of 10 requests per second** on this endpoint — Binance
    reports the per-second usage in `X-SAPI-USED-UID-WEIGHT-1S`. Do not poll it.

    Each row includes `transactionFee`, the network fee Binance charged for that
    withdrawal, so the true cost of moving funds out is readable here.

    When to Use:
    - "Did my withdrawal go through, and what did it cost?" — a recent, bounded lookup.
    - Looking up specific withdrawals by `withdraw_order_id` or `id_list`.

    When NOT to Use:
    - For the full history since the account opened — use
      `binance_get_all_withdrawals`, which budgets these expensive calls for you.
    - For deposits — use `binance_get_deposit_history`.
    - To MAKE a withdrawal — impossible by design; use the Binance app.

    Returns:
    A markdown table (applyTime, completeTime, coin, amount, transactionFee, network,
    status, walletType, txId, withdrawOrderId) plus per-coin totals including fees,
    capped at 100 displayed rows; or the raw Binance array with
    `response_format="json"`.

    Pagination:
    `limit` is 1-1000 (Binance default 1000) and `offset` pages within the window.
    `id_list` accepts at most 45 ids and is sent comma-separated.

    Windows:
    `start_time`/`end_time` together must span under 90 days — **under 7 days when
    `withdraw_order_id` is set** (Binance's own rule; it also defaults to the last 7
    days in that case). Both caps are enforced locally.

    Examples:
        params = {"coin": "BTC", "status": "completed"}
        params = {"start_time": "2026-08-01", "end_time": "2026-09-01"}
        params = {"id_list": ["b6ae22b3aa844210a7041aee7589627c"], "response_format": "json"}

    Error Handling:
    An over-wide window is rejected locally. -2015 means the key lacks Reading
    permission or this IP is not allowlisted. A 429 here means the 10 req/s ceiling was
    hit — back off, do not retry in a loop.
    """
    try:
        start_ms = _to_ms(params.start_time, "start_time")
        end_ms = _to_ms(params.end_time, "end_time")
        with_order_id = params.withdraw_order_id is not None
        window, error = _resolve_window(
            start_ms,
            end_ms,
            cap_ms=_ORDER_ID_WINDOW_CAP_MS if with_order_id else _WINDOW_CAP_MS,
            cap_label="7 days (because withdraw_order_id is set)" if with_order_id else "90 days",
            hint=("" if with_order_id else ", or use `binance_get_all_withdrawals` which walks the windows for you"),
        )
        if error:
            return error
        query: dict[str, Any] = {"offset": params.offset, "limit": params.limit, **window}
        if params.coin is not None:
            query["coin"] = params.coin
        if params.status is not None:
            query["status"] = _WITHDRAW_STATUS_CODES[params.status]
        if params.withdraw_order_id is not None:
            query["withdrawOrderId"] = params.withdraw_order_id
        if params.id_list is not None:
            query["idList"] = ",".join(params.id_list)
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/capital/withdraw/history", auth="signed", params=query)
        data = resp.json()
        rows = [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
        return _history_response(
            rows=rows,
            title="Binance withdrawal history",
            fmt=params.response_format,
            header=_WITHDRAW_HEADER,
            row_formatter=_withdraw_row,
            with_fees=True,
            extra={"offset": params.offset, "limit": params.limit},
        )
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_all_deposits",
    annotations=ToolAnnotations(
        title="Binance All Deposits (windowed walk)",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_all_deposits(params: AllDepositsInput) -> str:
    """Every crypto deposit since `since` — the 90-day cap walked for you.

    Binance only answers 90 days at a time, so this walks `until` → `since` in 89-day
    windows, newest first, paging `offset` by 1000 inside each window until a short page
    proves it is drained. Rows are deduped by their Binance `id`, so overlapping
    windows (and a resumed run) can never double-count.

    `since` defaults to **2017-07-01, Binance's launch**: there is no API that reports
    when an account was created — `apiRestrictions.createTime` is the API KEY's date,
    not the account's — so "everything" means "since the exchange existed". Windows
    before the account opened simply return nothing.

    Cost: `GET /sapi/v1/capital/deposit/hisrec` is IP weight 1, so the default
    `max_calls=60` (≈ 15 years of windows) is cheap. The budget is checked BEFORE every
    request and the reported call count is the real one.

    When to Use:
    - "Show me every deposit I have ever made" — the headline question.
    - Reconstructing cost basis or an audit trail of money coming IN.

    When NOT to Use:
    - For one recent window — `binance_get_deposit_history` is one call.
    - For withdrawals — use `binance_get_all_withdrawals`.
    - For fiat on-ramps (card/bank) — use `binance_get_fiat_orders` /
      `binance_get_fiat_payments` (fiat.py); they are a different rail.

    Returns:
    Rows sorted newest-first with per-coin totals (count + summed amount). Markdown
    displays at most 100 rows; `response_format="json"` returns
    `{count, truncated, since, until, resume_before, calls_made, totals, items}` with
    the full set. `truncated` is true exactly when the walk did not reach `since`.

    Pagination:
    If the walk stops before `since` — budget exhausted, or an error — it returns the
    rows it already has PLUS `resume_before`, the boundary of the next UNFETCHED range.
    Call again with the same `since` and that `resume_before` to continue. A stop in
    the middle of a window re-fetches that window, which the dedupe makes harmless.

    Examples:
        params = {}
        params = {"since": "2024-01-01", "coin": "USDT"}
        params = {"since": "2017-07-01", "resume_before": 1717200000000, "response_format": "json"}

    Error Handling:
    An over-wide-window error from Binance is tolerated: the window is halved and
    retried. Anything else — auth (-2015), rate limits (429/418), a Binance envelope
    failure — stops the walk immediately and is reported alongside the rows already
    collected and the resume cursor, so nothing fetched is ever thrown away.
    """
    try:
        since_ms = _to_ms_required(params.since, "since")
        resume_ms = _to_ms(params.resume_before, "resume_before")
        until_ms = resume_ms if resume_ms is not None else (_to_ms(params.until, "until") or _now_ms())
        if until_ms <= since_ms:
            return "Error: `until` (or `resume_before`) must be after `since`."
        extra: dict[str, Any] = {}
        if params.coin is not None:
            extra["coin"] = params.coin
        result = await _walk_history(
            path="/sapi/v1/capital/deposit/hisrec",
            id_key="id",
            since_ms=since_ms,
            until_ms=until_ms,
            max_calls=params.max_calls,
            extra_params=extra,
        )
        return _walk_response(
            result=result,
            title="Binance deposits — full history walk",
            fmt=params.response_format,
            header=_DEPOSIT_HEADER,
            row_formatter=_deposit_row,
            with_fees=False,
            time_key="insertTime",
            since_ms=since_ms,
            until_ms=until_ms,
        )
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_all_withdrawals",
    annotations=ToolAnnotations(
        title="Binance All Withdrawals (windowed walk)",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_all_withdrawals(params: AllWithdrawalsInput) -> str:
    """Every crypto withdrawal since `since` — the 90-day cap walked for you.

    Same walk as `binance_get_all_deposits` (89-day windows newest-first, `offset`
    paging by 1000 inside each, dedupe by Binance `id`), against
    `GET /sapi/v1/capital/withdraw/history`. Reading only — this server can never
    submit a withdrawal.

    ⚠️ **Each call costs UID weight 18000** of the 180000/min per-account budget, and
    the endpoint separately allows only **10 requests per second**. That is why
    `max_calls` defaults to **10** — one full minute of UID budget — instead of the 60
    the deposit walk uses. A full 2017→today sweep needs ~35 windows, so expect to
    resume across several calls; the cursor makes that exact and gap-free.

    `since` defaults to **2017-07-01, Binance's launch**: no API reports an account's
    creation date (`apiRestrictions.createTime` is the API KEY's date).

    When to Use:
    - "Show me every withdrawal I have ever made", including the fees they cost.
    - Auditing money going OUT, for tax or reconciliation.

    When NOT to Use:
    - For one recent window — `binance_get_withdraw_history` is one call.
    - For deposits — use `binance_get_all_deposits`.
    - To MAKE a withdrawal — impossible by design.

    Returns:
    Rows sorted newest-first with per-coin totals (count, summed amount, summed
    `transactionFee`). Markdown displays at most 100 rows; `response_format="json"`
    returns `{count, truncated, since, until, resume_before, calls_made, totals, items}`
    with the full set. `truncated` is true exactly when the walk did not reach `since`.

    Pagination:
    If the walk stops before `since` — which with `max_calls=10` is the normal case for
    a multi-year sweep — it returns the rows it has PLUS `resume_before`, the boundary
    of the next UNFETCHED range. Call again with the same `since` and that
    `resume_before`; repeat until `truncated` is false.

    Examples:
        params = {}
        params = {"since": "2024-01-01", "coin": "BTC"}
        params = {"since": "2017-07-01", "resume_before": 1717200000000, "max_calls": 10}

    Error Handling:
    An over-wide-window error is tolerated (the window is halved and retried). Auth
    (-2015), rate limits (429/418) and envelope failures stop the walk immediately and
    are reported alongside the rows already collected and the resume cursor.
    """
    try:
        since_ms = _to_ms_required(params.since, "since")
        resume_ms = _to_ms(params.resume_before, "resume_before")
        until_ms = resume_ms if resume_ms is not None else (_to_ms(params.until, "until") or _now_ms())
        if until_ms <= since_ms:
            return "Error: `until` (or `resume_before`) must be after `since`."
        extra: dict[str, Any] = {}
        if params.coin is not None:
            extra["coin"] = params.coin
        result = await _walk_history(
            path="/sapi/v1/capital/withdraw/history",
            id_key="id",
            since_ms=since_ms,
            until_ms=until_ms,
            max_calls=params.max_calls,
            extra_params=extra,
        )
        return _walk_response(
            result=result,
            title="Binance withdrawals — full history walk",
            fmt=params.response_format,
            header=_WITHDRAW_HEADER,
            row_formatter=_withdraw_row,
            with_fees=True,
            time_key="applyTime",
            since_ms=since_ms,
            until_ms=until_ms,
        )
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_deposit_address",
    annotations=ToolAnnotations(
        title="Binance Deposit Address",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_deposit_address(params: DepositAddressInput) -> str:
    """Get the deposit address for one coin on one network.

    Calls `GET /sapi/v1/capital/deposit/address` (SIGNED, IP weight 10). Omitting
    `network` returns the coin's default network — which is NOT always the one you
    want; `binance_get_coin_config` lists every network with its `isDefault` flag.

    ⚠️ Sending a coin to an address on the wrong network loses the funds. Confirm the
    network before using the address, and use the `tag`/memo when one is returned.

    When to Use:
    - Before sending crypto into Binance from an external wallet.

    When NOT to Use:
    - To see every address already issued for a coin — use
      `binance_get_deposit_addresses`.
    - To check whether deposits are even enabled for that coin/network right now —
      use `binance_get_coin_config` first.

    Returns:
    A markdown block with address, coin, tag and Binance's `url` (a block-explorer
    link), or raw JSON with `response_format="json"`.

    Examples:
        params = {"coin": "USDT", "network": "TRX"}
        params = {"coin": "BTC"}

    Error Handling:
    An unknown coin/network pair returns a Binance error naming the parameter. -2015
    means the key lacks Reading permission or this IP is not allowlisted.
    """
    try:
        query: dict[str, Any] = {"coin": params.coin}
        if params.network is not None:
            query["network"] = params.network
        if params.amount is not None:
            query["amount"] = params.amount
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/capital/deposit/address", auth="signed", params=query)
        data = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        lines = [
            f"# Binance deposit address — {data.get('coin') or params.coin}",
            "",
            f"- **address**: `{data.get('address') or 'N/A'}`",
            f"- **tag / memo**: {data.get('tag') or '_none_'}",
            f"- **network**: {params.network or '_coin default_'}",
            f"- **explorer**: {data.get('url') or 'N/A'}",
            "",
            "⚠️ Send only this coin, on this network. A wrong-network transfer is not recoverable.",
        ]
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_deposit_addresses",
    annotations=ToolAnnotations(
        title="Binance Deposit Address List",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_deposit_addresses(params: DepositAddressesInput) -> str:
    """List every deposit address issued for one coin, across networks.

    Calls `GET /sapi/v1/capital/deposit/address/list` (SIGNED, IP weight 10) and marks
    which address is the default for its network.

    When to Use:
    - To recognise an address you have used before, or to audit which addresses belong
      to this account.

    When NOT to Use:
    - To get an address to deposit to right now — `binance_get_deposit_address` returns
      the canonical one for a coin/network pair.

    Returns:
    A markdown table of network, address, tag and `isDefault`, capped at 50 rows, or
    raw JSON with `response_format="json"`.

    Examples:
        params = {"coin": "USDT"}
        params = {"coin": "USDT", "network": "BSC"}

    Error Handling:
    -2015 means the key lacks Reading permission or this IP is not allowlisted.
    """
    try:
        query: dict[str, Any] = {"coin": params.coin}
        if params.network is not None:
            query["network"] = params.network
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/capital/deposit/address/list", auth="signed", params=query)
        data = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        rows = [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
        lines = [f"# Binance deposit addresses — {params.coin}", ""]
        if not rows:
            lines.append("_No addresses issued for this coin yet._")
            return clip_response("\n".join(lines))
        lines.append("| network | address | tag | default |")
        lines.append("|---|---|---|---|")
        for row in rows[:MAX_COINS]:
            lines.append(
                f"| {row.get('network') or 'N/A'} | `{row.get('address') or 'N/A'}` "
                f"| {row.get('tag') or '_none_'} | {bool(row.get('isDefault'))} |"
            )
        if len(rows) > MAX_COINS:
            lines.append("")
            lines.append(f"_[{len(rows) - MAX_COINS} more address(es) not shown]_")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_coin_config",
    annotations=ToolAnnotations(
        title="Binance Coin Configuration",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_coin_config(params: CoinConfigInput) -> str:
    """Per-coin deposit/withdraw switches, networks, fees and minimums.

    Calls `GET /sapi/v1/capital/config/getall` (SIGNED, **IP weight 10**). The endpoint
    takes no filter and returns every listed coin (hundreds), so `coin` is applied
    client-side after the call — the cost is the same either way, which is why it is
    worth calling once and reading several coins out of the JSON.

    This is where you learn, before moving anything: whether deposits/withdrawals are
    enabled at all for a coin, which networks it supports, which is the default, what
    each network charges (`withdrawFee`) and its minimum (`withdrawMin`).

    When to Use:
    - Before depositing — confirm `depositEnable` on the network you plan to use.
    - To compare network fees for the same asset (e.g. USDT on TRX vs ETH).

    When NOT to Use:
    - To get the actual address — use `binance_get_deposit_address`.
    - For trading fees — that is `binance_get_trade_fees` (wallet_asset.py); this is
      the on-chain transfer fee.

    Returns:
    Per coin: `depositAllEnable` / `withdrawAllEnable`, then a per-network table of
    isDefault, depositEnable, withdrawEnable, withdrawFee and withdrawMin. Capped at 50
    coins in markdown (pass `coin` to narrow); `response_format="json"` returns the
    filtered payload in full.

    Examples:
        params = {"coin": "USDT"}
        params = {"response_format": "json"}

    Error Handling:
    An unknown `coin` returns an empty result, not an error — the filter is local.
    -2015 means the key lacks Reading permission or this IP is not allowlisted.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/capital/config/getall", auth="signed")
        data = resp.json()
        coins = [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
        if params.coin is not None:
            coins = [row for row in coins if str(row.get("coin", "")).upper() == params.coin]
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json({"count": len(coins), "items": coins}))
        lines = ["# Binance coin configuration", "", f"**{len(coins):,}** coin(s).", ""]
        if not coins:
            lines.append("_No matching coin._" if params.coin is None else f"_No coin named {params.coin} is listed._")
            return clip_response("\n".join(lines))
        for row in coins[:MAX_COINS]:
            lines.append(f"## {row.get('coin', 'N/A')} — {row.get('name') or ''}".rstrip(" —"))
            lines.append(f"- **deposits enabled (all networks)**: {bool(row.get('depositAllEnable'))}")
            lines.append(f"- **withdrawals enabled (all networks)**: {bool(row.get('withdrawAllEnable'))}")
            networks = [net for net in (row.get("networkList") or []) if isinstance(net, dict)]
            if networks:
                lines.append("")
                lines.append("| network | default | deposit | withdraw | withdrawFee | withdrawMin |")
                lines.append("|---|---|---|---|---|---|")
                for net in networks:
                    lines.append(
                        f"| {net.get('network') or 'N/A'} | {bool(net.get('isDefault'))} "
                        f"| {bool(net.get('depositEnable'))} | {bool(net.get('withdrawEnable'))} "
                        f"| {fmt_num(net.get('withdrawFee'))} | {fmt_num(net.get('withdrawMin'))} |"
                    )
            lines.append("")
        if len(coins) > MAX_COINS:
            lines.append(f'_[{len(coins) - MAX_COINS} more coin(s) not shown — pass `coin` or use "json"]_')
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)
