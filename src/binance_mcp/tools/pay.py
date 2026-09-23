"""Binance Pay (`/sapi/v1/pay/transactions`) — Pay history and an 18-month window walk.

Binance Card spending is NOT available via any public API. `GET /sapi/v1/pay/transactions`
only shows Binance Pay activity (merchant payments, C2C transfers, refunds, crypto box,
payouts, remittances); a Pay payment that happened to be *funded by* the Binance Card
shows up here with `walletType` 4 or 6 ("card") — that is the closest visibility this
server has into card usage (see `.memory/research/03-card-and-gaps.md` §1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from binance_mcp.client import get_client
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import ResponseFormat, clip_response, epoch_to_human, fmt_num, to_json
from binance_mcp.server import mcp

PAY_TRANSACTIONS_PATH = "/sapi/v1/pay/transactions"

# Endpoint's own caps (developers.binance.com/docs/pay/rest-api).
MAX_PAY_WINDOW_MS = 90 * 24 * 60 * 60 * 1000  # startTime..endTime must be <= 90 days
PAGE_LIMIT = 100  # default AND max `limit` for this endpoint

# `binance_get_pay_history` walk knobs.
WALK_WINDOW_MS = 89 * 24 * 60 * 60 * 1000  # stay under the 90-day cap with margin
PAY_HISTORY_LOOKBACK_MONTHS = 18  # "Support for querying orders within the last 18 months"
# Binance rejects a startTime AT the 18-month boundary (400, code 403004). `_months_ago_ms`
# truncates to midnight, so two days guarantee >= 24 h of real margin at any time of day.
PAY_LOOKBACK_MARGIN_MS = 2 * 24 * 60 * 60 * 1000
DEFAULT_MAX_CALLS = 30  # UID weight 3000 each -> 90,000 UID spent at the default

# Context-window guard on top of the API's own `limit`. Only bounds MARKDOWN display —
# JSON output always carries the full row set (only `clip_response`'s byte cap applies).
MAX_DISPLAY_ROWS = 50

WALLET_TYPE_NAMES: dict[int, str] = {1: "funding", 2: "spot", 3: "fiat", 4: "card", 5: "earn", 6: "card"}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _to_ms(value: Any) -> int | None:
    """Accept an epoch-ms int, a >=12-digit epoch-ms numeric string, or an ISO-8601 string.

    Raises ValueError on anything else — including a short digit string, which is
    ambiguous between an epoch and a bare year/id rather than silently misread.
    """
    if value is None or isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    digits = text[1:] if text.startswith("-") else text
    if digits.isdigit() and len(digits) >= 12:
        return int(text)
    iso = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        raise ValueError(f"{value!r} is not an epoch-ms integer (>=12 digits) or an ISO-8601 string") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def _as_ms(value: int | str | None) -> int | None:
    """Narrow an already-validated timestamp field back to `int` for typing purposes.

    The model's `mode="before"` validator has already normalized any raw input to
    `int | None` by construction time; the `str` arm only exists in the *declared*
    field type so the tool's JSON schema advertises ISO-8601 as an accepted input shape.
    """
    if value is None or isinstance(value, int):
        return value
    return _to_ms(value)


def _months_ago_ms(months: int, *, now_ms: int | None = None) -> int:
    """Subtract whole calendar months from `now_ms` (default: now), return an epoch ms."""
    now = datetime.fromtimestamp((now_ms if now_ms is not None else _now_ms()) / 1000, tz=UTC)
    total_months = now.year * 12 + (now.month - 1) - months
    year, month0 = divmod(total_months, 12)
    day = min(now.day, 28)  # dodge day-out-of-range (e.g. Aug 31 - 6 months)
    target = now.replace(year=year, month=month0 + 1, day=day, hour=0, minute=0, second=0, microsecond=0)
    return int(target.timestamp() * 1000)


def _wallet_type_name(code: Any) -> str:
    if code is None:
        return "N/A"
    try:
        return WALLET_TYPE_NAMES.get(int(code), str(code))
    except (TypeError, ValueError):
        return str(code)


def _format_amount(amount: Any) -> str:
    """Render the signed amount: `+` income, `-` expenditure (fmt_num already keeps the minus)."""
    if amount is None:
        return "N/A"
    try:
        dec = Decimal(str(amount))
    except (InvalidOperation, TypeError):
        return str(amount)
    rendered = fmt_num(dec)
    if dec > 0 and not rendered.startswith("+"):
        return f"+{rendered}"
    return rendered


def _counterparty(item: dict[str, Any]) -> str:
    """The other side of the transaction: receiver on an expenditure, payer on income."""
    try:
        is_expenditure = Decimal(str(item.get("amount"))) < 0
    except (InvalidOperation, TypeError):
        is_expenditure = False
    info = item.get("receiverInfo") if is_expenditure else item.get("payerInfo")
    if not isinstance(info, dict):
        info = item.get("payerInfo") if is_expenditure else item.get("receiverInfo")
    if not isinstance(info, dict):
        return "N/A"
    name = info.get("name") or info.get("binanceId") or info.get("accountId")
    return str(name) if name else "N/A"


def _format_pay_transaction(item: dict[str, Any]) -> str:
    when = epoch_to_human(item.get("transactionTime"))
    order_type = item.get("orderType", "N/A")
    amount = _format_amount(item.get("amount"))
    currency = item.get("currency", "")
    wallet = _wallet_type_name(item.get("walletType"))
    counterparty = _counterparty(item)
    tx_id = item.get("transactionId", "N/A")
    return f"- **{when}** — {order_type} {amount} {currency} ({wallet} wallet) with {counterparty} — txId `{tx_id}`"


def _dedupe_by_transaction_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """De-duplicate by transactionId, scoped to this one call's collected rows."""
    seen: set[Any] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        tx_id = item.get("transactionId")
        if tx_id is not None:
            if tx_id in seen:
                continue
            seen.add(tx_id)
        result.append(item)
    return result


class PayTransactionsInput(BaseModel):
    """Params for `GET /sapi/v1/pay/transactions` (UID weight 3000)."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    start_time: int | str | None = Field(
        default=None,
        description="Window start: ms epoch int, a >=12-digit epoch-ms string, or an ISO-8601 string. Give "
        "both start_time and end_time together (span <= 90 days), or neither — a lone bound is rejected.",
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end: ms epoch int, a >=12-digit epoch-ms string, or an ISO-8601 string. When both "
        "start_time and end_time are omitted, Binance returns the most recent 90 days.",
    )
    limit: int = Field(default=20, ge=1, le=PAGE_LIMIT, description="Max rows, 1-100 (this endpoint's own max).")
    wallet_type: int | None = Field(
        default=None,
        ge=1,
        le=6,
        description="Filter results (client-side) to one Binance Pay walletType code: 1 funding, 2 spot, 3 fiat, "
        "4 card, 5 earn, 6 card.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="markdown or json output.")

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _normalize_time(cls, value: Any, info: ValidationInfo) -> int | None:
        try:
            return _to_ms(value)
        except ValueError:
            raise ValueError(f"{info.field_name} must be epoch ms or ISO-8601") from None


class PayHistoryInput(BaseModel):
    """Params for the 18-month `binance_get_pay_history` window walk."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    since: int | str | None = Field(
        default=None,
        description="Lower bound: ms epoch int, a >=12-digit epoch-ms string, or an ISO-8601 string. Defaults "
        "to Binance's 18-month lookback (plus a two-day safety margin). A value older than that lookback is "
        "clamped to it and the response says so (`since_clamped`); Binance keeps no older Pay history.",
    )
    resume_before: int | str | None = Field(
        default=None,
        description="Cursor from a previous truncated call (ms epoch, epoch-ms string, or ISO-8601): only "
        "fetch transactions from before this instant. Overrides the default upper bound of 'now'. Pass it "
        "alongside the same `since` used before.",
    )
    max_calls: int = Field(
        default=DEFAULT_MAX_CALLS,
        ge=1,
        le=200,
        description="Budget of GET /sapi/v1/pay/transactions calls (UID weight 3000 each) to spend before "
        "stopping early and returning a resume_before cursor.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="markdown or json output.")

    @field_validator("since", "resume_before", mode="before")
    @classmethod
    def _normalize_time(cls, value: Any, info: ValidationInfo) -> int | None:
        try:
            return _to_ms(value)
        except ValueError:
            raise ValueError(f"{info.field_name} must be epoch ms or ISO-8601") from None


async def _fetch_pay_window(client: Any, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    resp = await client.request(
        "GET",
        PAY_TRANSACTIONS_PATH,
        params={"startTime": start_ms, "endTime": end_ms, "limit": PAGE_LIMIT},
        auth="signed",
    )
    data = resp.json().get("data") or []
    return list(data)


@dataclass
class PayHistoryWalkResult:
    """Outcome of one `_walk_pay_history` call."""

    transactions: list[dict[str, Any]]
    calls_used: int
    resume_before: int | None
    no_progress: bool
    possibly_incomplete: bool
    # Set when a request failed mid-walk: the rows collected so far are still returned,
    # `resume_before` points at the failed range's upper boundary, and this carries the
    # rendered error (walk rule 7d — never just the error string).
    stop_error: str | None = None


async def _walk_pay_history(client: Any, since_ms: int, until_ms: int, max_calls: int) -> PayHistoryWalkResult:
    """Walk [since_ms, until_ms) newest-first in <=89-day windows, bisecting full pages.

    `resume_before` is always the boundary of the next UNFETCHED range — the LIFO stack
    below is oldest-at-index-0 (newest on top, since a full page pushes the older half
    first so the newer half pops next), so everything strictly newer than the top-of-
    stack boundary has already been collected when the budget runs out — never derived
    from row content. `no_progress` is True when the budget ran out before even the
    newest top-level window could be reduced past its own starting boundary: emitting a
    cursor in that case would just repeat the identical calls forever, so no cursor is
    emitted and the caller should raise `max_calls` instead of resuming.
    `possibly_incomplete` is True when a bisection reached its 1 ms floor and STILL got
    back a full 100-row page — more rows exist in that instant than this endpoint can
    enumerate, and those extra rows were not fetched.
    """
    transactions: list[dict[str, Any]] = []
    calls_used = 0
    possibly_incomplete = False

    top_windows: list[tuple[int, int]] = []
    window_end = until_ms
    while window_end > since_ms:
        window_start = max(window_end - WALK_WINDOW_MS, since_ms)
        top_windows.append((window_start, window_end))
        window_end = window_start

    for window_start, window_end in top_windows:
        if calls_used >= max_calls:
            return PayHistoryWalkResult(transactions, calls_used, window_end, False, possibly_incomplete)

        stack: list[tuple[int, int]] = [(window_start, window_end)]
        while stack:
            if calls_used >= max_calls:
                # Oldest-at-index-0, newest-at-top: everything strictly newer than the
                # top-of-stack boundary has already been fully collected.
                cursor = stack[-1][1]
                if cursor >= until_ms:
                    # Not even the newest window could be reduced once — emitting this
                    # cursor would just repeat the exact same calls forever.
                    return PayHistoryWalkResult(transactions, calls_used, None, True, possibly_incomplete)
                return PayHistoryWalkResult(transactions, calls_used, cursor, False, possibly_incomplete)
            start_ms, end_ms = stack.pop()
            try:
                data = await _fetch_pay_window(client, start_ms, end_ms)
            except Exception as exc:
                if not transactions and isinstance(exc, RuntimeError):
                    # Nothing collected and a config-class failure (no credentials,
                    # kill-switch, envelope): surface it as a plain Error, not a partial walk.
                    raise
                # The failed request still spent its weight; the failed range [start, end]
                # was not fetched, and everything newer than `end_ms` in this top window
                # already was (LIFO: newer halves pop first).
                calls_used += 1
                if end_ms >= until_ms:
                    return PayHistoryWalkResult(
                        transactions, calls_used, None, True, possibly_incomplete, handle_api_error(exc)
                    )
                return PayHistoryWalkResult(
                    transactions, calls_used, end_ms, False, possibly_incomplete, handle_api_error(exc)
                )
            calls_used += 1
            if len(data) >= PAGE_LIMIT:
                if end_ms > start_ms:
                    mid = (start_ms + end_ms) // 2
                    if mid > start_ms:
                        # LIFO stack: push the older half first so the newer half pops next.
                        stack.append((start_ms, mid))
                        stack.append((mid + 1, end_ms))
                        continue
                # Bisection floor (a <=1 ms window) and STILL a full page: cannot split
                # further, so these rows are kept but flagged as possibly incomplete.
                possibly_incomplete = True
            transactions.extend(data)

    return PayHistoryWalkResult(transactions, calls_used, None, False, possibly_incomplete)


def _render_pay_transactions(data: list[dict[str, Any]], *, wallet_type: int | None, fmt: ResponseFormat) -> str:
    title = "Binance Pay Transactions"
    if wallet_type is not None:
        title = f"{title} (walletType={wallet_type} {_wallet_type_name(wallet_type)})"

    if fmt is ResponseFormat.JSON:
        # Full row set — this endpoint has no offset/cursor of its own to fake, and
        # `limit` already bounds what the API itself returned (<= 100).
        return clip_response(to_json({"title": title, "count": len(data), "items": data}))

    display = data[:MAX_DISPLAY_ROWS]
    lines = [f"# {title}", "", f"Showing **{len(display):,}** of **{len(data):,}** row(s)."]
    if len(data) > MAX_DISPLAY_ROWS:
        lines.append(
            f"_More than {MAX_DISPLAY_ROWS} rows in this window — narrow start_time/end_time, or use "
            "binance_get_pay_history to walk a wider range._"
        )
    lines.append("")
    if display:
        lines.extend(_format_pay_transaction(item) for item in display)
    else:
        lines.append("_No transactions in this window._")
    return clip_response("\n".join(lines))


@mcp.tool(
    name="binance_get_pay_transactions",
    annotations=ToolAnnotations(
        title="Binance Pay Transactions",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_pay_transactions(params: PayTransactionsInput) -> str:
    """Fetch Binance Pay transactions (merchant payments, C2C, refunds, payouts) for the account.

    Calls `GET /sapi/v1/pay/transactions` (**UID weight 3000**). Binance Card spending
    is NOT available via API — this only shows Binance Pay activity; a payment funded
    *by* the Binance Card appears here with `walletType` 4 or 6 ("card"), which is the
    closest visibility this server has into card usage.

    When to Use:
    - To review recent (<= 90 day) Binance Pay activity: merchant payments, C2C transfers,
      refunds, crypto box, payouts, remittances.
    - To find which wallet funded a Pay payment — pass `wallet_type` or read the rendered
      wallet name (funding/spot/fiat/card/earn).

    When NOT to Use:
    - For a span over 90 days, or to page past the first 100 rows of a dense window — use
      `binance_get_pay_history`, which walks and bisects the window for you.
    - To see Binance Card POS spend — no endpoint returns that (03-card-and-gaps.md §1);
      this only shows card-*funded* Pay payments, not card terminal purchases.

    Returns:
    Markdown list (or JSON) of transactions: time, orderType, signed amount (`+` income /
    `-` expenditure) and currency, walletType name, the counterparty name, and
    transactionId. Markdown display is capped at 50 rows with a note pointing at a
    narrower window or `binance_get_pay_history`; `response_format="json"` always carries
    every row this call fetched (still `limit`-capped by the API itself, <= 100).

    Pagination/Windows:
    `start_time`/`end_time` accept an ms epoch, a >=12-digit epoch-ms string, or an
    ISO-8601 string. Give both together (span <= 90 days) or neither — Binance's
    behavior for a single bound is undocumented for this endpoint, so a lone bound is
    rejected locally with a clear `Error:` rather than sent on. When both are omitted,
    Binance returns the most recent 90 days. `limit` is <= 100 (this endpoint's own max).

    Examples:
    params = {"wallet_type": 4}  # only card-funded Pay payments
    params = {"start_time": "2026-06-01", "end_time": "2026-08-01"}

    Error Handling:
    -1127 means the startTime/endTime span exceeds Binance's cap (should not happen —
    this tool validates first); -2015 means the key lacks permission or the IP is not
    on the key's allowlist.
    """
    try:
        start_ms = _as_ms(params.start_time)
        end_ms = _as_ms(params.end_time)

        if (start_ms is None) != (end_ms is None):
            if start_ms is not None:
                implied_days = (_now_ms() - start_ms) / 86_400_000
                hint = f"with only start_time given, the window through now would span {implied_days:.1f} day(s)"
            else:
                assert end_ms is not None
                implied_start = end_ms - MAX_PAY_WINDOW_MS
                hint = (
                    "with only end_time given, pass start_time explicitly too (e.g. "
                    f"start_time={implied_start}, exactly 90 days earlier)"
                )
            return f"Error: give both start_time and end_time together (span <= 90 days), or neither — {hint}."

        if start_ms is not None and end_ms is not None:
            span = end_ms - start_ms
            if span < 0:
                return "Error: start_time must be before end_time."
            if span > MAX_PAY_WINDOW_MS:
                return (
                    "Error: the startTime/endTime span for GET /sapi/v1/pay/transactions cannot exceed 90 days "
                    f"(got {span / 86_400_000:.1f} days). Narrow the window, or omit both to get the most recent "
                    "90 days."
                )

        client = get_client()
        query = {
            key: value
            for key, value in {"startTime": start_ms, "endTime": end_ms, "limit": params.limit}.items()
            if value is not None
        }
        resp = await client.request("GET", PAY_TRANSACTIONS_PATH, params=query, auth="signed")
        data: list[dict[str, Any]] = resp.json().get("data") or []
        if params.wallet_type is not None:
            data = [item for item in data if item.get("walletType") == params.wallet_type]

        return _render_pay_transactions(data, wallet_type=params.wallet_type, fmt=params.response_format)
    except Exception as exc:
        return handle_api_error(exc)


def _empty_walk(note: str, fmt: ResponseFormat) -> str:
    """The walk made no request: one uniform shape for both formats."""
    if fmt is ResponseFormat.JSON:
        return clip_response(to_json({"title": "Binance Pay History", "count": 0, "items": [], "note": note}))
    return clip_response(f"# Binance Pay History\n\n_{note}_")


def _render_pay_history(
    transactions: list[dict[str, Any]],
    *,
    calls_used: int,
    resume_before: int | None,
    no_progress: bool,
    possibly_incomplete: bool,
    since_ms: int,
    until_ms: int,
    fmt: ResponseFormat,
    stop_error: str | None = None,
    since_clamped: bool = False,
) -> str:
    if fmt is ResponseFormat.JSON:
        # Full row set — only `clip_response`'s byte cap bounds this, not MAX_DISPLAY_ROWS.
        return clip_response(
            to_json(
                {
                    "title": "Binance Pay History",
                    "since": since_ms,
                    "since_clamped": since_clamped,
                    "until": until_ms,
                    "calls_used": calls_used,
                    "count": len(transactions),
                    "resume_before": resume_before,
                    "no_progress": no_progress,
                    "possibly_incomplete": possibly_incomplete,
                    "stop_error": stop_error,
                    "items": transactions,
                }
            )
        )

    display = transactions[:MAX_DISPLAY_ROWS]
    display_truncated = len(transactions) > MAX_DISPLAY_ROWS

    lines = [
        "# Binance Pay History",
        "",
        f"Window walked: **{epoch_to_human(since_ms)}** -> **{epoch_to_human(until_ms)}** "
        f"({calls_used} call(s) spent).",
        f"Found **{len(transactions):,}** unique transaction(s) (deduped by transactionId within this "
        "call, newest first).",
    ]
    if since_clamped:
        lines.append(
            f"_`since` was clamped to Binance's 18-month lookback ({epoch_to_human(since_ms)}); "
            "older Pay history is not retrievable via the API._"
        )
    if stop_error is not None:
        lines.append(f"⚠️ Stopped early on a request failure: {stop_error}")
        if no_progress:
            lines.append(
                "Nothing was collected before the failure — fix the cause and call again with the same "
                "`since`/`resume_before`."
            )
        elif resume_before is not None:
            lines.append(
                f"The rows above are what was collected before the failure. Once the cause is fixed, resume "
                f"with `since={since_ms}, resume_before={resume_before}` ({epoch_to_human(resume_before)})."
            )
    elif no_progress:
        lines.append(
            "⚠️ The budget could not complete even the newest window — raise `max_calls` and call again "
            "with the same `since`/`resume_before`."
        )
    elif resume_before is not None:
        lines.append(
            f"⚠️ Call budget exhausted before reaching `since` — resume with `since={since_ms}, "
            f"resume_before={resume_before}` ({epoch_to_human(resume_before)})."
        )
    if possibly_incomplete:
        lines.append(
            "⚠️ At least one 1 ms instant still returned a full 100-row page — Binance has more "
            "transactions in that instant than this endpoint can enumerate; those extra rows were not fetched."
        )
    if display_truncated:
        lines.append(
            f"_Showing the {MAX_DISPLAY_ROWS} most recent of {len(transactions):,} — use "
            'response_format="json" for the full set._'
        )
    lines.append("")
    if display:
        lines.extend(_format_pay_transaction(item) for item in display)
    else:
        lines.append("_No transactions in this window._")
    return clip_response("\n".join(lines))


@mcp.tool(
    name="binance_get_pay_history",
    annotations=ToolAnnotations(
        title="Binance Pay History (18-Month Walk)",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_pay_history(params: PayHistoryInput) -> str:
    """Walk up to 18 months of Binance Pay history, past the 90-day / 100-row API caps.

    Repeatedly calls `GET /sapi/v1/pay/transactions` (**UID weight 3000 per call**) in
    <=89-day windows, newest-first, from `since` (default: Binance's 18-month lookback
    plus a two-day margin; an older `since` is clamped to it) up to now, or up to a `resume_before` cursor
    from a previous truncated call. A window that comes back with exactly 100 rows (the
    page limit) is bisected — split at its midpoint and re-walked — so a dense period is
    not silently dropped; if bisection reaches its 1 ms floor and STILL gets a full page,
    those extra rows are dropped and the response says so (`possibly_incomplete`).
    Spends at most `max_calls` requests (default 30 = 90,000 UID) before stopping; the
    returned `resume_before` cursor always marks the boundary of the next unfetched
    range (never derived from which rows happened to come back), and is omitted
    (`no_progress`) on the rare case the budget ran out before even the newest window
    could make any progress — resuming then would just repeat the same calls, so raise
    `max_calls` instead.

    Binance Card spending is NOT available via API; card-funded Pay payments appear here
    with `walletType` 4 or 6 ("card").

    When to Use:
    - To pull a full Pay history for reconciliation without hand-rolling the 90-day
      windowing or the 100-row-per-window cap.
    - To resume a previous run that stopped early — pass the same `since` plus its
      `resume_before` back in.

    When NOT to Use:
    - For a single recent window — `binance_get_pay_transactions` is one call and cheaper.

    Returns:
    Markdown (or JSON) list of transactions deduped by transactionId (scoped to this one
    call — not a persistent cross-call dedupe) sorted newest-first, the number of API
    calls spent, and — when the budget ran out before reaching `since` — a `resume_before`
    cursor to pass back on the next call alongside the same `since`. Markdown display is
    capped at 50 rows; `response_format="json"` always carries the full set collected by
    this call (only `clip_response`'s byte cap applies).

    Pagination/Windows:
    `since` accepts an ms epoch, a >=12-digit epoch-ms string, or an ISO-8601 string,
    default now minus 18 months (+2 days); anything older is clamped to that floor and
    reported as `since_clamped`. Each top-level window is at most 89 days (under
    Binance's 90-day cap); a window may cost more than one call if it has to be
    bisected, so `max_calls` bounds total calls, not windows. `resume_before` is always
    the boundary of the next unfetched range, so a resumed call never re-walks
    already-collected ranges (aside from harmless dedupe-caught edge overlaps).

    Examples:
    params = {}  # last 18 months, up to 30 calls
    params = {"since": "2026-01-01", "resume_before": 1700000000000, "max_calls": 10}

    Error Handling:
    A request failure mid-walk does NOT discard what was already fetched: the rows
    collected so far come back with `resume_before` (the boundary of the next unfetched
    range) and the failure itself in `stop_error` — fix the cause, then call again with
    the same `since` and that `resume_before`. `no_progress: true` with no cursor means
    nothing was collected before the failure — clear the cause and repeat the same call.
    A range that ends before Binance's 18-month lookback is refused locally (nothing in
    it is retrievable). -2015 means the key lacks permission or the IP is not on the
    key's allowlist.
    """
    try:
        since_input = _as_ms(params.since)
        resume_input = _as_ms(params.resume_before)
        until_ms = resume_input if resume_input is not None else _now_ms()
        # Binance rejects a startTime at or beyond its 18-month lookback with a 400
        # (code 403004 "invalid parameter") — measured live 2026-09-23 on the window that
        # began exactly 18 calendar months back at midnight. Keep a margin and clamp any
        # older `since` to the floor instead of erroring.
        lookback_floor_ms = _months_ago_ms(PAY_HISTORY_LOOKBACK_MONTHS) + PAY_LOOKBACK_MARGIN_MS
        if until_ms <= lookback_floor_ms:
            # The whole requested range is older than the lookback: every request would be
            # the same permanent 400, so refuse locally instead of inviting a retry loop.
            note = (
                f"The whole requested range (up to {epoch_to_human(until_ms)}) is older than Binance's "
                f"18-month Pay lookback ({epoch_to_human(lookback_floor_ms)}) — nothing in it is "
                "retrievable via the API."
            )
            return _empty_walk(note, params.response_format)
        since_clamped = since_input is not None and since_input < lookback_floor_ms
        since_ms = lookback_floor_ms if since_input is None or since_clamped else since_input

        if since_ms >= until_ms:
            # Unreachable with a clamped `since` (the floor is strictly below `until` here),
            # so this only ever reports the caller's own bounds.
            note = (
                f"No window to walk: since ({epoch_to_human(since_ms)}) is not before the upper bound "
                f"({epoch_to_human(until_ms)})."
            )
            return _empty_walk(note, params.response_format)

        client = get_client()
        result = await _walk_pay_history(client, since_ms, until_ms, params.max_calls)
        transactions = _dedupe_by_transaction_id(result.transactions)
        transactions.sort(key=lambda item: int(item.get("transactionTime") or 0), reverse=True)

        return _render_pay_history(
            transactions,
            calls_used=result.calls_used,
            resume_before=result.resume_before,
            no_progress=result.no_progress,
            possibly_incomplete=result.possibly_incomplete,
            since_ms=since_ms,
            until_ms=until_ms,
            fmt=params.response_format,
            stop_error=result.stop_error,
            since_clamped=since_clamped,
        )
    except Exception as exc:
        return handle_api_error(exc)
