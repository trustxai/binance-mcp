"""Binance Pay (`/sapi/v1/pay/transactions`) — Pay history and an 18-month window walk.

Binance Card spending is NOT available via any public API. `GET /sapi/v1/pay/transactions`
only shows Binance Pay activity (merchant payments, C2C transfers, refunds, crypto box,
payouts, remittances); a Pay payment that happened to be *funded by* the Binance Card
shows up here with `walletType` 4 or 6 ("card") — that is the closest visibility this
server has into card usage (see `.memory/research/03-card-and-gaps.md` §1).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, field_validator

from binance_mcp.client import get_client
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import ResponseFormat, clip_response, epoch_to_human, fmt_num, paginated_response, to_json
from binance_mcp.server import mcp

PAY_TRANSACTIONS_PATH = "/sapi/v1/pay/transactions"

# Endpoint's own caps (developers.binance.com/docs/pay/rest-api).
MAX_PAY_WINDOW_MS = 90 * 24 * 60 * 60 * 1000  # startTime..endTime must be <= 90 days
PAGE_LIMIT = 100  # default AND max `limit` for this endpoint

# `binance_get_pay_history` walk knobs.
WALK_WINDOW_MS = 89 * 24 * 60 * 60 * 1000  # stay under the 90-day cap with margin
PAY_HISTORY_LOOKBACK_MONTHS = 18  # "Support for querying orders within the last 18 months"
DEFAULT_MAX_CALLS = 30  # UID weight 3000 each -> 90,000 UID spent at the default

# Context-window guard on top of the API's own `limit`.
MAX_DISPLAY_ROWS = 50

WALLET_TYPE_NAMES: dict[int, str] = {1: "funding", 2: "spot", 3: "fiat", 4: "card", 5: "earn", 6: "card"}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _to_ms(value: Any) -> int | None:
    """Accept an epoch-ms int or an ISO-8601 string on the tool surface; normalize to ms."""
    if value is None or isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.lstrip("-").isdigit():
        return int(text)
    iso = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    parsed = datetime.fromisoformat(iso)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def _months_ago_ms(months: int, *, now_ms: int | None = None) -> int:
    """Subtract whole calendar months from `now_ms` (default: now), return an epoch ms."""
    now = datetime.fromtimestamp((now_ms if now_ms is not None else _now_ms()) / 1000, tz=UTC)
    total_months = now.year * 12 + (now.month - 1) - months
    year, month0 = divmod(total_months, 12)
    day = min(now.day, 28)  # dodge day-out-of-range (e.g. Aug 31 - 6 months)
    target = now.replace(year=year, month=month0 + 1, day=day, hour=0, minute=0, second=0, microsecond=0)
    return int(target.timestamp() * 1000)


def _wallet_type_name(code: Any) -> str:
    try:
        return WALLET_TYPE_NAMES.get(int(code), str(code))
    except (TypeError, ValueError):
        return str(code)


def _format_amount(amount: Any) -> str:
    """Render the signed amount: `+` income, `-` expenditure (fmt_num already keeps the minus)."""
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

    start_time: int | None = Field(
        default=None,
        description="Window start, ms epoch or ISO-8601. Paired with end_time the span cannot exceed 90 days.",
    )
    end_time: int | None = Field(
        default=None,
        description="Window end, ms epoch or ISO-8601. When both start_time and end_time are omitted, Binance "
        "returns the most recent 90 days.",
    )
    limit: int = Field(default=PAGE_LIMIT, ge=1, le=PAGE_LIMIT, description="Max rows (Binance default and max: 100).")
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
    def _normalize_time(cls, value: Any) -> int | None:
        return _to_ms(value)


class PayHistoryInput(BaseModel):
    """Params for the 18-month `binance_get_pay_history` window walk."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    since: int | None = Field(
        default=None,
        description="Lower bound, ms epoch or ISO-8601. Defaults to 18 months ago — Binance's documented lookback "
        "for this endpoint.",
    )
    resume_before: int | None = Field(
        default=None,
        description="Cursor from a previous truncated call: only fetch transactions from before this ms epoch "
        "(ms epoch or ISO-8601). Overrides the default upper bound of 'now'.",
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
    def _normalize_time(cls, value: Any) -> int | None:
        return _to_ms(value)


async def _fetch_pay_window(client: Any, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    resp = await client.request(
        "GET",
        PAY_TRANSACTIONS_PATH,
        params={"startTime": start_ms, "endTime": end_ms, "limit": PAGE_LIMIT},
        auth="signed",
    )
    data = resp.json().get("data") or []
    return list(data)


async def _walk_pay_history(
    client: Any, since_ms: int, until_ms: int, max_calls: int
) -> tuple[list[dict[str, Any]], int, int | None]:
    """Walk [since_ms, until_ms) newest-first in <=89-day windows, bisecting full pages.

    Returns (transactions, calls_used, resume_before). `resume_before` is the boundary
    of the newest not-fully-processed top-level window, or None when `since_ms` was
    reached without exhausting `max_calls`.
    """
    transactions: list[dict[str, Any]] = []
    calls_used = 0

    top_windows: list[tuple[int, int]] = []
    window_end = until_ms
    while window_end > since_ms:
        window_start = max(window_end - WALK_WINDOW_MS, since_ms)
        top_windows.append((window_start, window_end))
        window_end = window_start

    for window_start, window_end in top_windows:
        if calls_used >= max_calls:
            return transactions, calls_used, window_end

        stack: list[tuple[int, int]] = [(window_start, window_end)]
        exhausted = False
        while stack:
            if calls_used >= max_calls:
                exhausted = True
                break
            start_ms, end_ms = stack.pop()
            data = await _fetch_pay_window(client, start_ms, end_ms)
            calls_used += 1
            if len(data) >= PAGE_LIMIT and end_ms > start_ms:
                mid = (start_ms + end_ms) // 2
                if mid > start_ms:
                    # LIFO stack: push the older half first so the newer half pops next.
                    stack.append((start_ms, mid))
                    stack.append((mid + 1, end_ms))
                    continue
            transactions.extend(data)

        if exhausted:
            return transactions, calls_used, window_end

    return transactions, calls_used, None


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
    transactionId. Display capped at 50 rows; pass `response_format="json"` for the full
    (still `limit`-capped) set.

    Pagination/Windows:
    `start_time`/`end_time` accept an ms epoch or an ISO-8601 string. When both are
    omitted, Binance returns the most recent 90 days. When both are given, the span
    cannot exceed 90 days — checked locally, returning a clear `Error:` instead of
    letting Binance answer -1127. `limit` is <= 100 (this endpoint's own max).

    Examples:
    params = {"wallet_type": 4}  # only card-funded Pay payments
    params = {"start_time": "2026-06-01", "end_time": "2026-08-01"}

    Error Handling:
    -1127 means the startTime/endTime span exceeds Binance's cap (should not happen —
    this tool validates first); -2015 means the key lacks permission or the IP is not
    on the key's allowlist.
    """
    try:
        if params.start_time is not None and params.end_time is not None:
            span = params.end_time - params.start_time
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
            for key, value in {
                "startTime": params.start_time,
                "endTime": params.end_time,
                "limit": params.limit,
            }.items()
            if value is not None
        }
        resp = await client.request("GET", PAY_TRANSACTIONS_PATH, params=query, auth="signed")
        data: list[dict[str, Any]] = resp.json().get("data") or []
        if params.wallet_type is not None:
            data = [item for item in data if item.get("walletType") == params.wallet_type]

        title = "Binance Pay Transactions"
        if params.wallet_type is not None:
            title = f"{title} (walletType={params.wallet_type} {_wallet_type_name(params.wallet_type)})"

        return paginated_response(
            items=data[:MAX_DISPLAY_ROWS],
            limit=params.limit,
            offset=0,
            fmt=params.response_format,
            item_formatter=_format_pay_transaction,
            title=title,
            total=len(data),
        )
    except Exception as exc:
        return handle_api_error(exc)


def _render_pay_history(
    transactions: list[dict[str, Any]],
    *,
    calls_used: int,
    resume_before: int | None,
    since_ms: int,
    until_ms: int,
    fmt: ResponseFormat,
) -> str:
    display = transactions[:MAX_DISPLAY_ROWS]
    truncated = len(transactions) > MAX_DISPLAY_ROWS

    if fmt is ResponseFormat.JSON:
        return clip_response(
            to_json(
                {
                    "title": "Binance Pay History",
                    "since": since_ms,
                    "until": until_ms,
                    "calls_used": calls_used,
                    "count": len(transactions),
                    "resume_before": resume_before,
                    "truncated": truncated,
                    "items": display,
                }
            )
        )

    lines = [
        "# Binance Pay History",
        "",
        f"Window walked: **{epoch_to_human(since_ms)}** -> **{epoch_to_human(until_ms)}** "
        f"({calls_used} call(s) spent).",
        f"Found **{len(transactions):,}** unique transaction(s) (deduped by transactionId, newest first).",
    ]
    if resume_before is not None:
        lines.append(
            f"⚠️ Call budget exhausted before reaching `since` — resume with "
            f"`resume_before={resume_before}` ({epoch_to_human(resume_before)})."
        )
    if truncated:
        lines.append(
            f"_Showing the {MAX_DISPLAY_ROWS} most recent of {len(transactions):,} — "
            'use response_format="json" for the full set.'
            "_"
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
    <=89-day windows, newest-first, from `since` (default 18 months ago — Binance's
    documented lookback for this endpoint) up to now, or up to a `resume_before` cursor
    from a previous truncated call. A window that comes back with exactly 100 rows (the
    page limit) is bisected — split at its midpoint and re-walked — so a dense period is
    not silently dropped. Spends at most `max_calls` requests (default 30 = 90,000 UID)
    before stopping and returning a `resume_before` cursor.

    Binance Card spending is NOT available via API; card-funded Pay payments appear here
    with `walletType` 4 or 6 ("card").

    When to Use:
    - To pull a full Pay history for reconciliation without hand-rolling the 90-day
      windowing or the 100-row-per-window cap.
    - To resume a previous run that stopped early — pass its `resume_before` back in.

    When NOT to Use:
    - For a single recent window — `binance_get_pay_transactions` is one call and cheaper.

    Returns:
    Markdown (or JSON) list of deduplicated transactions (by transactionId) sorted
    newest-first, the number of API calls spent, and — when the budget ran out before
    reaching `since` — a `resume_before` cursor to pass back on the next call. Display
    capped at 50 rows (JSON keeps the full set, up to the response-size cap).

    Pagination/Windows:
    `since` accepts an ms epoch or an ISO-8601 string, default now minus 18 months. Each
    top-level window is at most 89 days (under Binance's 90-day cap); a window may cost
    more than one call if it has to be bisected, so `max_calls` bounds total calls, not
    windows. Resuming re-walks the top-level 89-day window that was in progress when the
    budget ran out, so a handful of duplicate rows across calls is expected and handled
    by the transactionId dedupe.

    Examples:
    params = {}  # last 18 months, up to 30 calls
    params = {"resume_before": 1700000000000, "max_calls": 10}

    Error Handling:
    Any Binance error aborts the walk and is returned as `Error: ...` with no partial
    results — retry from the same `since`/`resume_before` once fixed. -2015 means the
    key lacks permission or the IP is not on the key's allowlist.
    """
    try:
        until_ms = params.resume_before if params.resume_before is not None else _now_ms()
        since_ms = params.since if params.since is not None else _months_ago_ms(PAY_HISTORY_LOOKBACK_MONTHS)

        if since_ms >= until_ms:
            return (
                "# Binance Pay History\n\n"
                f"_No window to walk: since ({epoch_to_human(since_ms)}) is not before the upper bound "
                f"({epoch_to_human(until_ms)})._"
            )

        client = get_client()
        raw, calls_used, resume_before = await _walk_pay_history(client, since_ms, until_ms, params.max_calls)
        transactions = _dedupe_by_transaction_id(raw)
        transactions.sort(key=lambda item: item.get("transactionTime", 0), reverse=True)

        return _render_pay_history(
            transactions,
            calls_used=calls_used,
            resume_before=resume_before,
            since_ms=since_ms,
            until_ms=until_ms,
            fmt=params.response_format,
        )
    except Exception as exc:
        return handle_api_error(exc)
