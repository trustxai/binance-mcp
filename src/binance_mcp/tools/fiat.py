"""Fiat rails: `/sapi/v1/fiat/orders` (bank deposit/withdraw) and `/sapi/v1/fiat/payments`
(buy/sell crypto with fiat), plus a budgeted since-walk over both. All SIGNED (USER_DATA).

`fiat/orders` is UID-weighted **45000** of the 180000/min UID budget — at most **4 calls
per minute**; `fiat/payments` is a cheap IP weight 1. A bank card buying crypto here
(`fiat/payments`, `paymentMethod` "Credit Card") is **NOT the Binance Card** — Binance
Card spend has no public API surface at all (`.memory/research/03-card-and-gaps.md` §1).
The closest proxies to card activity are `binance_get_funding_wallet` (t10) and
`binance_get_pay_transactions` (t4, `walletType` 4/6).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import httpx
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from binance_mcp.client import BinanceClient, get_client
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import ResponseFormat, clip_response, epoch_to_human, fmt_num, paginated_response, to_json
from binance_mcp.server import mcp

# Context-window guard: markdown rendering caps at this many rows; JSON keeps everything
# (still subject to `clip_response`'s byte ceiling).
MAX_DISPLAY_ROWS = 50

_ORDERS_PATH = "/sapi/v1/fiat/orders"
_PAYMENTS_PATH = "/sapi/v1/fiat/payments"

_ORDER_TRANSACTION_TYPES: dict[str, int] = {"deposit": 0, "withdraw": 1}
_PAYMENT_TRANSACTION_TYPES: dict[str, int] = {"buy": 0, "sell": 1}

# kind -> (endpoint, transactionType) for the since-walk.
_HISTORY_KIND_META: dict[str, tuple[str, int]] = {
    "deposits": (_ORDERS_PATH, 0),
    "withdrawals": (_ORDERS_PATH, 1),
    "buys": (_PAYMENTS_PATH, 0),
    "sells": (_PAYMENTS_PATH, 1),
}

# The fallback window size once the wide-span attempt errors (30 days, matching the
# "recent 30-day data" default both endpoints fall back to when begin/end are omitted).
_WINDOW_MS = 30 * 24 * 60 * 60 * 1000

# HTTP statuses that narrowing the time window would NOT fix (auth / rate limit) —
# these must propagate immediately instead of burning the budget on a fallback retry.
_UNRECOVERABLE_STATUS_CODES = frozenset({401, 403, 418, 429})


def _to_ms(value: int | str, *, field: str) -> int:
    """Accept an epoch-ms int, an epoch-ms numeric string (>=12 digits), or an
    ISO-8601 string (tool surface) and return epoch ms.

    Raises `ValueError` naming `field` when `value` matches none of those shapes.
    """
    if isinstance(value, int):
        return value
    text = value.strip()
    if text.isdigit() and len(text) >= 12:
        return int(text)
    iso_text = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError as exc:
        raise ValueError(f"{field} must be epoch milliseconds or an ISO-8601 string, got {value!r}.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def _format_fiat_order(item: dict[str, Any]) -> str:
    """Render one `fiat/orders` row (deposit or withdraw)."""
    order_no = item.get("orderNo", "?")
    currency = item.get("fiatCurrency", "?")
    amount = fmt_num(item.get("amount"))
    indicated = fmt_num(item.get("indicatedAmount"))
    fee = fmt_num(item.get("totalFee"))
    method = item.get("method", "?")
    status = item.get("status", "?")
    created = epoch_to_human(item.get("createTime"))
    return (
        f"- **{order_no}** — {amount} {currency} (indicated {indicated} {currency}, fee {fee} {currency}) "
        f"via {method} — status **{status}** — {created}"
    )


def _format_fiat_payment(item: dict[str, Any]) -> str:
    """Render one `fiat/payments` row (buy or sell)."""
    order_no = item.get("orderNo", "?")
    fiat_currency = item.get("fiatCurrency", "?")
    source_amount = fmt_num(item.get("sourceAmount"))
    crypto_currency = item.get("cryptoCurrency", "?")
    obtain_amount = fmt_num(item.get("obtainAmount"))
    price = fmt_num(item.get("price"))
    fee = fmt_num(item.get("totalFee"))
    status = item.get("status", "?")
    created = epoch_to_human(item.get("createTime"))
    method = item.get("paymentMethod")
    method_note = f" via {method}" if method else ""
    return (
        f"- **{order_no}** — {source_amount} {fiat_currency} → {obtain_amount} {crypto_currency} "
        f"@ {price}{method_note} (fee {fee} {fiat_currency}) — status **{status}** — {created}"
    )


def _sum_by_currency(rows: list[dict[str, Any]], *, amount_field: str, currency_field: str) -> dict[str, Decimal]:
    """Sum `amount_field` grouped by `currency_field`, skipping rows that lack either."""
    totals: dict[str, Decimal] = {}
    for row in rows:
        currency = row.get(currency_field)
        raw_amount = row.get(amount_field)
        if not currency or raw_amount is None:
            continue
        try:
            amount = Decimal(str(raw_amount))
        except InvalidOperation:
            continue
        totals[currency] = totals.get(currency, Decimal(0)) + amount
    return totals


def _dedupe_by_order_no(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe by `orderNo` (a later occurrence wins); rows without one are kept as-is.

    Overlapping fetches are expected here — resuming a walk re-touches the inclusive
    `endTime` boundary, and a mid-window retry re-walks the whole window — so this
    always runs before rows are returned.
    """
    by_order_no: dict[str, dict[str, Any]] = {}
    unkeyed: list[dict[str, Any]] = []
    for row in rows:
        order_no = row.get("orderNo")
        if order_no is None:
            unkeyed.append(row)
        else:
            by_order_no[str(order_no)] = row
    return [*by_order_no.values(), *unkeyed]


def _markdown_truncation_note(display_cap: int, total_on_page: int, next_page: int) -> str:
    return (
        f"\n\n_[display capped at {display_cap} of {total_on_page} rows fetched on this page — "
        f'pass `response_format="json"` for all of them, or `page={next_page}` for the next page.]_'
    )


class _WalkError(RuntimeError):
    """Raised by `_walk_span` when a request fails, carrying whatever rows were
    already collected and exactly how many calls were spent (including the failed
    one) before the failure — so the caller never has to guess or under/over-count
    the budget already spent.
    """

    def __init__(self, cause: Exception, rows: list[dict[str, Any]], calls_made: int) -> None:
        self.cause = cause
        self.rows = rows
        self.calls_made = calls_made
        super().__init__(str(cause))


def _is_recoverable(cause: Exception) -> bool:
    """True for an HTTP error plausibly caused by an over-wide time span (worth
    retrying with a narrower window, e.g. -1127); False for auth/rate-limit/any
    non-HTTP failure that narrowing the window would not fix, and that must
    propagate immediately instead of burning the budget on a doomed retry.
    """
    return isinstance(cause, httpx.HTTPStatusError) and cause.response.status_code not in _UNRECOVERABLE_STATUS_CODES


class FiatOrdersInput(BaseModel):
    """Input for `binance_get_fiat_orders` (`GET /sapi/v1/fiat/orders`)."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    transaction_type: Literal["deposit", "withdraw"] = Field(
        description="Fiat order side: 'deposit' (fiat -> Binance, API transactionType=0) or "
        "'withdraw' (Binance -> fiat, API transactionType=1)."
    )
    begin_time: int | str | None = Field(
        default=None,
        description="Window start — epoch ms, an epoch-ms numeric string, or ISO-8601. API name `beginTime`. "
        "Omit both begin/end for the API default of the most recent 30 days.",
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end — epoch ms, an epoch-ms numeric string, or ISO-8601. API name `endTime`.",
    )
    page: int = Field(default=1, ge=1, description="1-indexed page number, passed through as `page`.")
    rows: int = Field(default=100, ge=1, le=500, description="Rows per page, passed through as `rows` (max 500).")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="markdown or json output.")


class FiatPaymentsInput(BaseModel):
    """Input for `binance_get_fiat_payments` (`GET /sapi/v1/fiat/payments`)."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    transaction_type: Literal["buy", "sell"] = Field(
        description="Fiat payment side: 'buy' crypto with fiat (API transactionType=0) or 'sell' crypto "
        "for fiat (API transactionType=1)."
    )
    begin_time: int | str | None = Field(
        default=None,
        description="Window start — epoch ms, an epoch-ms numeric string, or ISO-8601. API name `beginTime`. "
        "Omit both begin/end for the API default of the most recent 30 days.",
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end — epoch ms, an epoch-ms numeric string, or ISO-8601. API name `endTime`.",
    )
    page: int = Field(default=1, ge=1, description="1-indexed page number, passed through as `page`.")
    rows: int = Field(default=100, ge=1, le=500, description="Rows per page, passed through as `rows` (max 500).")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="markdown or json output.")


class FiatHistoryInput(BaseModel):
    """Input for `binance_get_fiat_history` — a budgeted since-walk over orders or payments."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    kind: Literal["deposits", "withdrawals", "buys", "sells"] = Field(
        description="Which fiat history to walk: deposits/withdrawals hit `fiat/orders` (UID 45000/call, "
        "default budget 4 calls/min); buys/sells hit `fiat/payments` (IP 1/call, default budget 20)."
    )
    since: int | str = Field(
        default="2017-07-01",
        description="Earliest point to walk back to — epoch ms, an epoch-ms numeric string, or ISO-8601. "
        "Default is Binance's fiat-rail launch date; there is no account-creation endpoint to derive a "
        "tighter bound.",
    )
    resume_before: int | str | None = Field(
        default=None,
        description="Epoch ms, an epoch-ms numeric string, or ISO-8601 cursor copied from a previous call's "
        "`resume_before` output. Used as the new `endTime` to continue a walk that stopped early — Binance "
        "treats `endTime` as inclusive, so the boundary row may be re-fetched; rows are deduped by `orderNo` "
        "before being returned. Omit to start from now.",
    )
    rows: int = Field(default=100, ge=1, le=500, description="Page size passed to Binance as `rows` (max 500).")
    max_calls: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Hard cap on API calls this invocation makes. Defaults to 4 for kind=deposits/withdrawals "
        "(each call costs UID 45000 of the 180000/min budget — 4 calls spends it for the minute) and 20 for "
        "kind=buys/sells (IP 1/call, cheap). Pass an explicit value to override either default.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="markdown or json output.")


@mcp.tool(
    name="binance_get_fiat_orders",
    annotations=ToolAnnotations(
        title="Binance Fiat Deposit/Withdraw Orders",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_fiat_orders(params: FiatOrdersInput) -> str:
    """List fiat-rail deposit or withdraw orders (bank transfer/card top-up of the fiat wallet).

    Calls `GET /sapi/v1/fiat/orders`, signed. This is the **UID-weighted 45000** fiat-rail
    ledger — of the 180000/min UID budget that is at most **4 calls per minute**. Do not
    poll this tool in a loop; for a bulk backfill use `binance_get_fiat_history`, which
    already budgets calls.

    When to Use:
    - Seeing fiat bank-rail deposits into, or withdrawals out of, the fiat wallet (bank
      transfer, SEPA, card-funded top-ups of the *fiat* balance — not Spot).
    - Reconciling a specific order by scanning a narrow begin_time/end_time window.

    When NOT to Use:
    - Buying/selling crypto with fiat (a card or bank purchase of BTC/ETH/...) — use
      `binance_get_fiat_payments`.
    - A long backfill across many pages or months — use `binance_get_fiat_history`.
    - Binance Card spend — there is no API for that (see the module docstring).

    Returns:
    A markdown list (or JSON with `response_format="json"`) of orders: order number,
    fiat currency, indicated vs settled amount, fee, method, status, created/updated
    time. Status values Binance returns: Processing, Failed, Successful, Finished,
    Refunding, Refunded, Refund Failed, Order Partial credit Stopped.

    Pagination:
    `page` (1-indexed) / `rows` (max 500) map directly to the API; the response also
    reports Binance's own `total` row count across all pages. Markdown display is
    capped at 50 rows (with a note naming the next `page` to request); JSON keeps
    the full page.

    Examples:
    params = {"transaction_type": "deposit", "rows": 50}
    params = {"transaction_type": "withdraw", "begin_time": "2026-01-01", "end_time": "2026-02-01"}

    Error Handling:
    A 200 body with `success: false` is raised by the client as `BinanceEnvelopeError`
    and surfaced here as `Error: ...`; an undocumented span cap on this endpoint
    typically surfaces as -1127 — narrow begin_time/end_time and retry. A malformed
    begin_time/end_time returns `Error: <field> must be epoch milliseconds or an
    ISO-8601 string` without calling the API.
    """
    try:
        client = get_client()
        api_params: dict[str, Any] = {
            "transactionType": _ORDER_TRANSACTION_TYPES[params.transaction_type],
            "page": params.page,
            "rows": params.rows,
        }
        if params.begin_time is not None:
            api_params["beginTime"] = _to_ms(params.begin_time, field="begin_time")
        if params.end_time is not None:
            api_params["endTime"] = _to_ms(params.end_time, field="end_time")
        resp = await client.request("GET", _ORDERS_PATH, params=api_params, auth="signed")
        body: dict[str, Any] = resp.json()
        data: list[dict[str, Any]] = body.get("data") or []
        total_raw = body.get("total")
        total = int(total_raw) if isinstance(total_raw, int) else None
        display_items = data if params.response_format is ResponseFormat.JSON else data[:MAX_DISPLAY_ROWS]
        response = paginated_response(
            items=display_items,
            limit=params.rows,
            offset=(params.page - 1) * params.rows,
            fmt=params.response_format,
            item_formatter=_format_fiat_order,
            title="Binance Fiat Deposit/Withdraw Orders",
            total=total,
        )
        if params.response_format is not ResponseFormat.JSON and len(data) > MAX_DISPLAY_ROWS:
            response += _markdown_truncation_note(MAX_DISPLAY_ROWS, len(data), params.page + 1)
        return response
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_fiat_payments",
    annotations=ToolAnnotations(
        title="Binance Fiat Buy/Sell Payments",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_fiat_payments(params: FiatPaymentsInput) -> str:
    """List crypto buy/sell payments made with fiat (bank transfer or bank-issued card).

    Calls `GET /sapi/v1/fiat/payments`, signed, IP weight 1 (cheap — unlike
    `fiat/orders`). A "Credit Card" `paymentMethod` here is a **bank-issued card buying
    crypto**, and is **NOT the Binance Card** — Binance Card spend has no API surface at
    all (`.memory/research/03-card-and-gaps.md` §1); the closest proxies are
    `binance_get_funding_wallet` and `binance_get_pay_transactions` (walletType 4/6).

    When to Use:
    - Seeing crypto bought or sold with fiat: source/obtained amounts, price, fee,
      payment method (buy only), status.
    - Reconciling a specific purchase by scanning a narrow begin_time/end_time window.

    When NOT to Use:
    - Bank deposits/withdrawals of fiat itself — use `binance_get_fiat_orders`.
    - A long backfill across many pages or months — use `binance_get_fiat_history`.

    Returns:
    A markdown list (or JSON with `response_format="json"`) of payments: order number,
    fiat amount/currency, obtained crypto amount/currency, price, fee, payment method
    (buy only), status, created/updated time. Status values Binance returns: Processing,
    Failed, Successful, Finished, Refunding, Refunded, Refund Failed, Order Partial
    credit Stopped.

    Pagination:
    `page` (1-indexed) / `rows` (max 500) map directly to the API; the response also
    reports Binance's own `total` row count across all pages. Markdown display is
    capped at 50 rows (with a note naming the next `page` to request); JSON keeps
    the full page.

    Examples:
    params = {"transaction_type": "buy", "rows": 50}
    params = {"transaction_type": "sell", "begin_time": "2026-01-01", "end_time": "2026-02-01"}

    Error Handling:
    A 200 body with `success: false` is raised by the client as `BinanceEnvelopeError`
    and surfaced here as `Error: ...`. A malformed begin_time/end_time returns
    `Error: <field> must be epoch milliseconds or an ISO-8601 string` without calling
    the API.
    """
    try:
        client = get_client()
        api_params: dict[str, Any] = {
            "transactionType": _PAYMENT_TRANSACTION_TYPES[params.transaction_type],
            "page": params.page,
            "rows": params.rows,
        }
        if params.begin_time is not None:
            api_params["beginTime"] = _to_ms(params.begin_time, field="begin_time")
        if params.end_time is not None:
            api_params["endTime"] = _to_ms(params.end_time, field="end_time")
        resp = await client.request("GET", _PAYMENTS_PATH, params=api_params, auth="signed")
        body: dict[str, Any] = resp.json()
        data: list[dict[str, Any]] = body.get("data") or []
        total_raw = body.get("total")
        total = int(total_raw) if isinstance(total_raw, int) else None
        display_items = data if params.response_format is ResponseFormat.JSON else data[:MAX_DISPLAY_ROWS]
        response = paginated_response(
            items=display_items,
            limit=params.rows,
            offset=(params.page - 1) * params.rows,
            fmt=params.response_format,
            item_formatter=_format_fiat_payment,
            title="Binance Fiat Buy/Sell Payments",
            total=total,
        )
        if params.response_format is not ResponseFormat.JSON and len(data) > MAX_DISPLAY_ROWS:
            response += _markdown_truncation_note(MAX_DISPLAY_ROWS, len(data), params.page + 1)
        return response
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return handle_api_error(exc)


async def _fetch_history_page(
    client: BinanceClient,
    *,
    endpoint: str,
    transaction_type: int,
    begin_ms: int,
    end_ms: int,
    page: int,
    rows: int,
) -> list[dict[str, Any]]:
    resp = await client.request(
        "GET",
        endpoint,
        params={
            "transactionType": transaction_type,
            "beginTime": begin_ms,
            "endTime": end_ms,
            "page": page,
            "rows": rows,
        },
        auth="signed",
    )
    body: dict[str, Any] = resp.json()
    data: list[dict[str, Any]] = body.get("data") or []
    return data


async def _walk_span(
    client: BinanceClient,
    *,
    endpoint: str,
    transaction_type: int,
    begin_ms: int,
    end_ms: int,
    rows: int,
    max_calls: int,
    calls_made: int,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Page through one [begin_ms, end_ms] span with `page`, checking the budget
    before every request (a request that itself errors still spends one call).

    Returns (collected rows, updated calls_made, budget_exhausted) on success.
    Raises `_WalkError` — carrying the rows collected and the real call count
    (including the failed request) — if a request fails.
    """
    collected: list[dict[str, Any]] = []
    page = 1
    while calls_made < max_calls:
        try:
            data = await _fetch_history_page(
                client,
                endpoint=endpoint,
                transaction_type=transaction_type,
                begin_ms=begin_ms,
                end_ms=end_ms,
                page=page,
                rows=rows,
            )
        except Exception as exc:
            calls_made += 1  # the failed request still spent one call of the budget
            raise _WalkError(exc, collected, calls_made) from exc
        calls_made += 1
        collected.extend(data)
        if len(data) < rows:
            return collected, calls_made, False
        page += 1
    return collected, calls_made, True


@mcp.tool(
    name="binance_get_fiat_history",
    annotations=ToolAnnotations(
        title="Binance Fiat History (Budgeted Since-Walk)",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_fiat_history(params: FiatHistoryInput) -> str:
    """Walk fiat deposit/withdraw or buy/sell history across pages and time windows.

    Tries ONE call spanning `beginTime=since` .. `endTime=now` (or `resume_before` when
    resuming) and pages `page` until a short page signals the end. If that wide-span
    attempt fails with a plausibly span-related HTTP error (commonly -1127, an
    undocumented span cap on these two endpoints), falls back to walking 30-day windows
    newest-first instead.

    Honors a `max_calls` budget so one invocation can never blow past the weight-limited
    call rate — deposits/withdrawals cost UID 45000/call (default budget 4, i.e. the
    whole 180000/min UID budget for a minute), buys/sells cost IP 1/call (default budget
    20). Every request checks the budget first, and a request that itself errors still
    counts against it, since it still spent real quota.

    The walk NEVER discards rows it already has: whenever it stops early — budget
    exhaustion or ANY request failure, span-related or not — it renders the normal
    report (rows collected so far, totals, call count) plus a `resume_before` cursor,
    and for a failure it also shows the underlying error (via `handle_api_error`) as a
    prominent line. Binance treats `endTime` as inclusive, so a re-fetched boundary row
    is expected; rows are deduped by `orderNo` before being returned. The cursor comes
    in two flavours, worded differently so one is never mistaken for the other: a
    **redo** cursor (the wide span, or the window in progress, was not fully covered —
    page order inside it is undocumented, so the whole thing must be retried) says it
    "re-covers the same range, it does not advance"; an **advance** cursor (every window
    up to it is fully covered; only the budget stopped a NEW window from starting) says
    rows "were not fetched" before it and to "continue" from there.

    Status values Binance returns for fiat orders/payments: Processing, Failed,
    Successful, Finished, Refunding, Refunded, Refund Failed, Order Partial credit
    Stopped.

    When to Use:
    - A bulk backfill of fiat activity (deposits, withdrawals, or crypto buys/sells with
      fiat) since account creation, paged and budgeted automatically.
    - Continuing a previous walk that stopped early: pass its `resume_before` back in.

    When NOT to Use:
    - A single narrow lookup — use `binance_get_fiat_orders` / `binance_get_fiat_payments`
      directly with your own begin_time/end_time; it is one call instead of many.
    - Binance Card spend — not retrievable via any API (see the module docstring); a
      "Credit Card" `paymentMethod` in `buys`/`sells` is a bank card, not the Binance Card.

    Returns:
    Deduped rows sorted newest-first (display capped at 50, JSON keeps the full walked
    set), a per-fiat-currency total (and, for buys/sells, a per-crypto-currency received
    total), how many API calls were spent, whether the window fallback triggered, and —
    when the walk stopped early — a `resume_before` cursor plus (for a failure) the
    error that caused the stop.

    Examples:
    params = {"kind": "deposits"}
    params = {"kind": "buys", "since": "2023-01-01"}
    params = {"kind": "withdrawals", "resume_before": 1700000000000}

    Error Handling:
    A span-related HTTP error on the wide-span attempt triggers the 30-day-window
    fallback automatically. Any failure after that point — inside a window, or an
    auth/rate-limit/envelope/other error that was never span-related — stops the walk
    instead of raising: the response still shows the rows already collected, the
    API-call count, a `resume_before` cursor, and the failure itself via
    `handle_api_error`. A malformed since/resume_before returns
    `Error: <field> must be epoch milliseconds or an ISO-8601 string` before any call
    is made.
    """
    try:
        client = get_client()
        endpoint, transaction_type = _HISTORY_KIND_META[params.kind]
        is_payment = params.kind in ("buys", "sells")
        formatter = _format_fiat_payment if is_payment else _format_fiat_order
        max_calls = params.max_calls if params.max_calls is not None else (20 if is_payment else 4)

        since_ms = _to_ms(params.since, field="since")
        end_ms = (
            _to_ms(params.resume_before, field="resume_before")
            if params.resume_before is not None
            else int(time.time() * 1000)
        )

        all_rows: list[dict[str, Any]] = []
        calls_made = 0
        used_fallback = False
        stop_reason: Literal["budget", "error"] | None = None
        stop_error: Exception | None = None
        resume_before_ms: int | None = None
        resume_is_redo = False

        try:
            all_rows, calls_made, exhausted = await _walk_span(
                client,
                endpoint=endpoint,
                transaction_type=transaction_type,
                begin_ms=since_ms,
                end_ms=end_ms,
                rows=params.rows,
                max_calls=max_calls,
                calls_made=0,
            )
            if exhausted:
                # Page order within the wide span is undocumented, so there is no safe
                # partial boundary — the whole span must be retried with a bigger budget.
                stop_reason = "budget"
                resume_before_ms = end_ms
                resume_is_redo = True
        except _WalkError as werr:
            # Never discard what a failed attempt already collected, whether or not the
            # failure is the kind that's worth retrying via the windowed fallback.
            all_rows = werr.rows
            calls_made = werr.calls_made
            if _is_recoverable(werr.cause):
                used_fallback = True
            else:
                stop_reason = "error"
                stop_error = werr.cause
                resume_before_ms = end_ms
                resume_is_redo = True

        if used_fallback:
            window_end = end_ms
            while window_end >= since_ms:
                if calls_made >= max_calls:
                    stop_reason = "budget"
                    resume_before_ms = window_end
                    resume_is_redo = False
                    break
                window_start = max(since_ms, window_end - _WINDOW_MS)
                try:
                    window_rows, calls_made, window_exhausted = await _walk_span(
                        client,
                        endpoint=endpoint,
                        transaction_type=transaction_type,
                        begin_ms=window_start,
                        end_ms=window_end,
                        rows=params.rows,
                        max_calls=max_calls,
                        calls_made=calls_made,
                    )
                except _WalkError as werr:
                    # Once inside a window there is no further fallback to try — any
                    # failure here, recoverable-shaped or not, just stops the walk.
                    all_rows.extend(werr.rows)
                    calls_made = werr.calls_made
                    stop_reason = "error"
                    stop_error = werr.cause
                    resume_before_ms = window_end
                    resume_is_redo = True
                    break
                all_rows.extend(window_rows)
                if window_exhausted:
                    stop_reason = "budget"
                    resume_before_ms = window_end  # redo this same window on resume
                    resume_is_redo = True
                    break
                window_end = window_start - 1

        all_rows = _dedupe_by_order_no(all_rows)
        all_rows.sort(key=lambda row: int(row.get("createTime") or 0), reverse=True)

        amount_field = "sourceAmount" if is_payment else "amount"
        totals = _sum_by_currency(all_rows, amount_field=amount_field, currency_field="fiatCurrency")
        crypto_totals: dict[str, Decimal] = {}
        if is_payment:
            crypto_totals = _sum_by_currency(all_rows, amount_field="obtainAmount", currency_field="cryptoCurrency")

        if params.response_format is ResponseFormat.JSON:
            payload: dict[str, Any] = {
                "kind": params.kind,
                "since": since_ms,
                "until": end_ms,
                "calls_made": calls_made,
                "used_fallback": used_fallback,
                "resume_before": resume_before_ms,
                "resume_is_redo": resume_is_redo if resume_before_ms is not None else None,
                "stop_reason": stop_reason,
                "stop_error": handle_api_error(stop_error) if stop_error is not None else None,
                "count": len(all_rows),
                "totals": {currency: str(total) for currency, total in totals.items()},
                "crypto_totals": (
                    {currency: str(total) for currency, total in crypto_totals.items()} if is_payment else None
                ),
                "rows": all_rows,
            }
            return clip_response(to_json(payload))

        lines: list[str] = [
            f"# Binance Fiat History — {params.kind} (since {epoch_to_human(since_ms)})",
            "",
            f"Fetched **{len(all_rows)}** row(s) across **{calls_made}** API call(s)"
            + (" (fell back to 30-day windows)" if used_fallback else "")
            + ".",
        ]
        if stop_reason == "error":
            assert stop_error is not None
            lines.append(f"⚠️ Stopped early — {handle_api_error(stop_error)}")
            lines.append(
                f"`resume_before={resume_before_ms}` re-covers the same range, it does not advance — call "
                "again with a larger `max_calls`, a narrower `since`, or once the underlying issue is resolved."
            )
        elif stop_reason == "budget" and resume_is_redo:
            lines.append(
                f"⚠️ Stopped early — the budget stopped this span partway (`max_calls={max_calls}`). Call "
                f"again with a larger `max_calls` (or a narrower `since`); `resume_before={resume_before_ms}` "
                "re-covers the same range, it does not advance."
            )
        elif stop_reason == "budget":
            lines.append(
                f"⚠️ Stopped early — the `max_calls` budget ({max_calls}) ran out. "
                f"Rows before {epoch_to_human(resume_before_ms)} were not fetched; call again with "
                f"`resume_before={resume_before_ms}` to continue."
            )
        lines.append("")
        lines.append("**Totals by fiat currency:**")
        if totals:
            for currency, total in sorted(totals.items()):
                lines.append(f"- {currency}: {fmt_num(total)}")
        else:
            lines.append("_none_")
        if is_payment:
            lines.append("")
            lines.append("**Totals by crypto currency received:**")
            if crypto_totals:
                for currency, total in sorted(crypto_totals.items()):
                    lines.append(f"- {currency}: {fmt_num(total)}")
            else:
                lines.append("_none_")
        lines.append("")
        lines.append(f"Rows (newest first, display capped at {MAX_DISPLAY_ROWS}):")
        for row in all_rows[:MAX_DISPLAY_ROWS]:
            lines.append(formatter(row))
        if not all_rows:
            lines.append("_No items._")
        return clip_response("\n".join(lines))
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return handle_api_error(exc)
