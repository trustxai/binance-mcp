"""Unit tests for `binance_mcp.tools.fiat` against a fake client."""

from __future__ import annotations

import re
import time
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from binance_mcp.tools.fiat import (
    _WINDOW_MS,
    MAX_DISPLAY_ROWS,
    FiatHistoryInput,
    FiatOrdersInput,
    FiatPaymentsInput,
    binance_get_fiat_history,
    binance_get_fiat_orders,
    binance_get_fiat_payments,
)


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Records every call; replays a fixed queue of responses/exceptions in order."""

    def __init__(self, queue: list[Any]) -> None:
        self._queue = list(queue)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, path, kwargs))
        if not self._queue:
            raise AssertionError("no more fake responses queued")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


def _status_error(path: str, status: int, body: dict[str, Any]) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", f"https://api.binance.com{path}")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def _order_row(
    order_no: str, amount: str, currency: str, create_time: int, status: str = "Successful"
) -> dict[str, Any]:
    return {
        "orderNo": order_no,
        "fiatCurrency": currency,
        "indicatedAmount": amount,
        "amount": amount,
        "totalFee": "0.50",
        "method": "Bank Transfer",
        "status": status,
        "createTime": create_time,
        "updateTime": create_time + 1000,
    }


def _payment_row(
    order_no: str,
    source_amount: str,
    fiat_currency: str,
    obtain_amount: str,
    crypto_currency: str,
    create_time: int,
    payment_method: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "orderNo": order_no,
        "sourceAmount": source_amount,
        "fiatCurrency": fiat_currency,
        "obtainAmount": obtain_amount,
        "cryptoCurrency": crypto_currency,
        "totalFee": "1.00",
        "price": "50000.00",
        "status": "Successful",
        "createTime": create_time,
        "updateTime": create_time + 1000,
    }
    if payment_method is not None:
        row["paymentMethod"] = payment_method
    return row


def _orders_envelope(rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    return {"code": "000000", "message": "success", "data": rows, "total": total, "success": True}


# --------------------------------------------------------------------------- orders


async def test_fiat_orders_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = _orders_envelope([_order_row("O1", "100.00", "USD", 1700000000000)], total=1)
    fake = _FakeClient(queue=[envelope])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_orders(FiatOrdersInput(transaction_type="deposit"))

    assert "**O1**" in result
    assert "100 USD" in result
    assert "Bank Transfer" in result
    assert "Successful" in result
    method, path, kwargs = fake.calls[0]
    assert method == "GET"
    assert path == "/sapi/v1/fiat/orders"
    assert kwargs["auth"] == "signed"
    assert kwargs["params"] == {"transactionType": 0, "page": 1, "rows": 100}


async def test_fiat_orders_withdraw_with_iso_window_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = _orders_envelope([_order_row("O2", "250.00", "EUR", 1706745600000)], total=1)
    fake = _FakeClient(queue=[envelope])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_orders(
        FiatOrdersInput(
            transaction_type="withdraw",
            begin_time="2024-02-01T00:00:00Z",
            end_time="2024-02-02T00:00:00Z",
            response_format="json",
        )
    )

    assert '"orderNo": "O2"' in result
    method, path, kwargs = fake.calls[0]
    assert method == "GET"
    assert path == "/sapi/v1/fiat/orders"
    assert kwargs["auth"] == "signed"
    assert kwargs["params"] == {
        "transactionType": 1,
        "page": 1,
        "rows": 100,
        "beginTime": 1706745600000,
        "endTime": 1706832000000,
    }


async def test_fiat_orders_accepts_numeric_string_begin_time(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = _orders_envelope([_order_row("O3", "1.00", "USD", 1700000000000)], total=1)
    fake = _FakeClient(queue=[envelope])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    await binance_get_fiat_orders(FiatOrdersInput(transaction_type="deposit", begin_time="1700000000000"))

    _, _, kwargs = fake.calls[0]
    assert kwargs["params"]["beginTime"] == 1700000000000


async def test_fiat_orders_malformed_begin_time_returns_error_without_calling_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient(queue=[])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_orders(FiatOrdersInput(transaction_type="deposit", begin_time="not-a-date"))

    assert result.startswith("Error: begin_time must be epoch milliseconds or an ISO-8601 string")
    assert fake.calls == []


async def test_fiat_orders_display_truncation_note(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_order_row(f"O{i}", "1.00", "USD", 1700000000000 + i) for i in range(60)]
    envelope = _orders_envelope(rows, total=60)
    fake = _FakeClient(queue=[envelope])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_orders(FiatOrdersInput(transaction_type="deposit", rows=60))

    assert result.count("- **O") == MAX_DISPLAY_ROWS
    assert "display capped at 50 of 60 rows fetched on this page" in result
    assert "page=2" in result


async def test_fiat_orders_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        queue=[_status_error("/sapi/v1/fiat/orders", 400, {"code": -1127, "msg": "Start time is too far away."})]
    )
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_orders(FiatOrdersInput(transaction_type="deposit"))

    assert result.startswith("Error (400)")
    assert "-1127" in result


def test_fiat_orders_rejects_out_of_range_rows() -> None:
    with pytest.raises(ValidationError):
        FiatOrdersInput(transaction_type="deposit", rows=501)


def test_fiat_orders_rejects_unknown_transaction_type() -> None:
    with pytest.raises(ValidationError):
        FiatOrdersInput(transaction_type="bogus")  # type: ignore[arg-type]


@pytest.mark.live
async def test_fiat_orders_live_smoke() -> None:
    """Live smoke: read-only, needs BINANCE_API_KEY + secret/PEM in the environment.
    Not runnable on the spot testnet — /sapi does not exist there."""
    result = await binance_get_fiat_orders(FiatOrdersInput(transaction_type="deposit", rows=1))
    assert not result.startswith("Error: unexpected failure")


# ------------------------------------------------------------------------- payments


async def test_fiat_payments_buy_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = _orders_envelope(
        [_payment_row("P1", "500.00", "USD", "0.01", "BTC", 1700000000000, payment_method="Credit Card")],
        total=1,
    )
    fake = _FakeClient(queue=[envelope])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_payments(FiatPaymentsInput(transaction_type="buy"))

    assert "**P1**" in result
    assert "500 USD" in result
    assert "0.01 BTC" in result
    assert "via Credit Card" in result
    method, path, kwargs = fake.calls[0]
    assert method == "GET"
    assert path == "/sapi/v1/fiat/payments"
    assert kwargs["auth"] == "signed"
    assert kwargs["params"] == {"transactionType": 0, "page": 1, "rows": 100}


async def test_fiat_payments_sell_has_no_payment_method_note(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = _orders_envelope(
        [_payment_row("P2", "0.02", "BTC", "1000.00", "USD", 1700000001000)],
        total=1,
    )
    fake = _FakeClient(queue=[envelope])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_payments(FiatPaymentsInput(transaction_type="sell"))

    assert "**P2**" in result
    assert " via " not in result
    method, path, kwargs = fake.calls[0]
    assert method == "GET"
    assert path == "/sapi/v1/fiat/payments"
    assert kwargs["auth"] == "signed"
    assert kwargs["params"]["transactionType"] == 1


async def test_fiat_payments_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(queue=[_status_error("/sapi/v1/fiat/payments", 401, {"code": -2015, "msg": "Invalid API-key."})])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_payments(FiatPaymentsInput(transaction_type="buy"))

    assert result.startswith("Error (401)")


@pytest.mark.live
async def test_fiat_payments_live_smoke() -> None:
    """Live smoke: read-only, needs BINANCE_API_KEY + secret/PEM in the environment.
    Not runnable on the spot testnet — /sapi does not exist there."""
    result = await binance_get_fiat_payments(FiatPaymentsInput(transaction_type="buy", rows=1))
    assert not result.startswith("Error: unexpected failure")


# --------------------------------------------------------------------------- history


async def test_fiat_history_single_call_totals_and_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _order_row("H1", "10.00", "USD", 1700000000000),
        _order_row("H2", "5.00", "USD", 1700000500000),
        _order_row("H3", "20.00", "EUR", 1700001000000),
    ]
    envelope = _orders_envelope(rows, total=3)
    fake = _FakeClient(queue=[envelope])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_history(FiatHistoryInput(kind="deposits", rows=100))

    assert len(fake.calls) == 1
    method, path, kwargs = fake.calls[0]
    assert method == "GET"
    assert path == "/sapi/v1/fiat/orders"
    assert kwargs["auth"] == "signed"
    assert kwargs["params"]["transactionType"] == 0
    assert "Fetched **3** row(s) across **1** API call(s)." in result
    assert "USD: 15" in result
    assert "EUR: 20" in result
    # newest-first: H3 (1700001000000) before H1 (1700000000000).
    assert result.index("**H3**") < result.index("**H1**")
    assert "resume_before" not in result


async def test_fiat_history_pages_within_span(monkeypatch: pytest.MonkeyPatch) -> None:
    page1 = _orders_envelope(
        [_order_row("H1", "1.00", "USD", 1700000000000), _order_row("H2", "1.00", "USD", 1700000001000)], total=3
    )
    page2 = _orders_envelope([_order_row("H3", "1.00", "USD", 1700000002000)], total=3)
    fake = _FakeClient(queue=[page1, page2])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_history(FiatHistoryInput(kind="deposits", rows=2, max_calls=10))

    assert len(fake.calls) == 2
    method0, _, kwargs0 = fake.calls[0]
    assert method0 == "GET"
    assert kwargs0["auth"] == "signed"
    assert kwargs0["params"]["page"] == 1
    assert fake.calls[1][2]["params"]["page"] == 2
    assert "Fetched **3** row(s) across **2** API call(s)." in result


async def test_fiat_history_falls_back_to_windows_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # `since` within the 30-day fallback window of "now" so the whole span is covered
    # by exactly one window call — otherwise the walk would keep requesting earlier
    # windows until it reaches `since`, needing more fake responses than queued here.
    recent_since_ms = int(time.time() * 1000) - 5 * 24 * 60 * 60 * 1000
    fallback_row = _payment_row("B1", "100.00", "USD", "0.002", "BTC", recent_since_ms + 1000)
    fake = _FakeClient(
        queue=[
            _status_error("/sapi/v1/fiat/payments", 400, {"code": -1127, "msg": "span too wide"}),
            _orders_envelope([fallback_row], total=1),
        ]
    )
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_history(FiatHistoryInput(kind="buys", since=recent_since_ms, max_calls=10))

    assert len(fake.calls) == 2
    assert fake.calls[1][2]["auth"] == "signed"
    assert fake.calls[1][1] == "/sapi/v1/fiat/payments"
    assert "fell back to 30-day windows" in result
    assert "**B1**" in result
    assert "Totals by crypto currency received:" in result
    assert "BTC: 0.002" in result


async def test_fiat_history_unrecoverable_error_propagates_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(queue=[_status_error("/sapi/v1/fiat/orders", 401, {"code": -2015, "msg": "Invalid API-key."})])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_history(FiatHistoryInput(kind="deposits", max_calls=10))

    assert len(fake.calls) == 1  # no fallback retry attempted — auth errors are not span-related
    assert result.startswith("Error (401)")
    assert "fell back" not in result


async def test_fiat_history_wide_span_budget_exhausted_resumes_at_end(monkeypatch: pytest.MonkeyPatch) -> None:
    full_page = _orders_envelope(
        [_order_row("H1", "1.00", "USD", 1700000000000), _order_row("H2", "1.00", "USD", 1700000001000)], total=99
    )
    fake = _FakeClient(queue=[full_page])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    before = int(time.time() * 1000)
    result = await binance_get_fiat_history(FiatHistoryInput(kind="withdrawals", rows=2, max_calls=1))
    after = int(time.time() * 1000)

    assert len(fake.calls) == 1
    assert "Stopped early" in result
    # Page order within a single wide span is undocumented, so there is no safe
    # narrower boundary — resume_before falls back to the original end (now).
    match = re.search(r"resume_before=(\d+)", result)
    assert match is not None
    assert before <= int(match.group(1)) <= after


async def test_fiat_history_window_budget_exhaustion_and_resume_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact blocking repro: max_calls=2, the wide-span attempt errors (1 call),
    one 30-day window succeeds with a short page (2nd call) — the walk must stop
    there (not claim to have covered everything back to `since`) and must emit a
    `resume_before` equal to the boundary of the next, still-unfetched window. The
    emitted cursor is then chained into a fresh call to prove it is actually usable.
    """
    since_ms = 1_000_000_000_000
    initial_end_ms = 1_700_000_000_000

    span_error = _status_error("/sapi/v1/fiat/orders", 400, {"code": -1127, "msg": "span too wide"})
    window_row = _order_row("W1", "10.00", "USD", 1_699_000_000_000)
    fake = _FakeClient(queue=[span_error, _orders_envelope([window_row], total=1)])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_history(
        FiatHistoryInput(kind="withdrawals", since=since_ms, resume_before=initial_end_ms, max_calls=2)
    )

    assert len(fake.calls) == 2
    assert "fell back to 30-day windows" in result
    assert "Stopped early" in result

    match = re.search(r"resume_before=(-?\d+)", result)
    assert match is not None
    resume_before = int(match.group(1))

    expected_window_start = initial_end_ms - _WINDOW_MS  # since_ms is far below, so max() picks this
    expected_next_boundary = expected_window_start - 1
    assert resume_before == expected_next_boundary

    # The window call used the expected [window_start, initial_end_ms] bounds.
    window_kwargs = fake.calls[1][2]
    assert window_kwargs["params"]["beginTime"] == expected_window_start
    assert window_kwargs["params"]["endTime"] == initial_end_ms

    # Chain the emitted cursor into a fresh call: it must become the new endTime.
    fake2 = _FakeClient(queue=[_orders_envelope([window_row], total=1)])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake2)

    await binance_get_fiat_history(
        FiatHistoryInput(kind="withdrawals", since=since_ms, resume_before=resume_before, max_calls=5)
    )

    resumed_kwargs = fake2.calls[0][2]
    assert resumed_kwargs["params"]["endTime"] == resume_before


async def test_fiat_history_resume_before_input_becomes_end_time(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = _orders_envelope([_order_row("H1", "1.00", "USD", 1699999999000)], total=1)
    fake = _FakeClient(queue=[envelope])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    await binance_get_fiat_history(FiatHistoryInput(kind="deposits", resume_before=1700000000999))

    _, _, kwargs = fake.calls[0]
    assert kwargs["params"]["endTime"] == 1700000000999


async def test_fiat_history_dedupes_by_order_no(monkeypatch: pytest.MonkeyPatch) -> None:
    recent_since_ms = int(time.time() * 1000) - 5 * 24 * 60 * 60 * 1000
    span_error = _status_error("/sapi/v1/fiat/orders", 400, {"code": -1127, "msg": "span too wide"})
    dup_row_a = _order_row("DUP1", "1.00", "USD", recent_since_ms + 1000)
    dup_row_b = dict(dup_row_a)
    fake = _FakeClient(queue=[span_error, _orders_envelope([dup_row_a, dup_row_b], total=2)])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_history(FiatHistoryInput(kind="deposits", since=recent_since_ms, max_calls=10))

    assert result.count("**DUP1**") == 1


async def test_fiat_history_default_max_calls_for_deposits_is_four(monkeypatch: pytest.MonkeyPatch) -> None:
    full_page = _orders_envelope(
        [_order_row("H1", "1.00", "USD", 1700000000000), _order_row("H2", "1.00", "USD", 1700000001000)], total=999
    )
    fake = _FakeClient(queue=[full_page, full_page, full_page, full_page])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_history(FiatHistoryInput(kind="deposits", rows=2))

    assert len(fake.calls) == 4  # default budget for deposits/withdrawals: UID 45000/call
    assert "Stopped early" in result


async def test_fiat_history_default_max_calls_for_buys_is_twenty(monkeypatch: pytest.MonkeyPatch) -> None:
    full_page = _orders_envelope([_payment_row("P1", "1.00", "USD", "0.0001", "BTC", 1700000000000)], total=999)
    fake = _FakeClient(queue=[full_page] * 5 + [_orders_envelope([], total=6)])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_history(FiatHistoryInput(kind="buys", rows=1))

    assert len(fake.calls) == 6  # well under the default budget of 20 for buys/sells
    assert "Stopped early" not in result


async def test_fiat_history_malformed_since_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(queue=[])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_history(FiatHistoryInput(kind="deposits", since="not-a-date"))

    assert result.startswith("Error: since must be epoch milliseconds or an ISO-8601 string")
    assert fake.calls == []


def test_fiat_history_rejects_bad_kind() -> None:
    with pytest.raises(ValidationError):
        FiatHistoryInput(kind="bogus")  # type: ignore[arg-type]


@pytest.mark.live
async def test_fiat_history_live_smoke() -> None:
    """Live smoke: read-only, needs BINANCE_API_KEY + secret/PEM in the environment.
    Not runnable on the spot testnet — /sapi does not exist there."""
    result = await binance_get_fiat_history(FiatHistoryInput(kind="deposits", max_calls=1))
    assert not result.startswith("Error: unexpected failure")
