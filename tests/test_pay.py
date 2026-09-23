"""Unit tests for tools/pay.py against a fake client."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from binance_mcp.tools.pay import (
    PAY_HISTORY_LOOKBACK_MONTHS,
    PAY_LOOKBACK_MARGIN_MS,
    WALK_WINDOW_MS,
    PayHistoryInput,
    PayTransactionsInput,
    _format_amount,
    _months_ago_ms,
    _to_ms,
    _walk_pay_history,
    _wallet_type_name,
    binance_get_pay_history,
    binance_get_pay_transactions,
)


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Routes by path; records every call (method, path, kwargs).

    For pay/transactions calls, `data_by_window` maps (startTime, endTime) -> list of
    raw rows, so tests can script the windowed walk deterministically.
    """

    def __init__(
        self,
        payload: Any = None,
        data_by_window: dict[tuple[int, int], list[dict[str, Any]]] | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._payload = payload
        self._use_window_routing = data_by_window is not None
        self._data_by_window = data_by_window or {}
        self._exc = exc
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, path, kwargs))
        if self._exc is not None:
            raise self._exc
        if self._use_window_routing:
            params = kwargs.get("params", {})
            key = (params.get("startTime"), params.get("endTime"))
            data = self._data_by_window.get(key, [])
            return _FakeResponse({"code": "000000", "message": "success", "data": data, "success": True})
        return _FakeResponse(self._payload)


def _status_error(status: int, body: dict[str, Any]) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.binance.com/sapi/v1/pay/transactions")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def _tx(
    tx_id: int,
    *,
    when: int = 1_700_000_000_000,
    order_type: str = "PAY",
    amount: str = "-12.5",
    currency: str = "USDT",
    wallet_type: int = 2,
    payer: dict[str, Any] | None = None,
    receiver: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "orderType": order_type,
        "transactionId": tx_id,
        "transactionTime": when,
        "amount": amount,
        "currency": currency,
        "walletType": wallet_type,
        "payerInfo": payer or {"name": "Alejandro"},
        "receiverInfo": receiver or {"name": "Merchant Corp"},
    }


# -- pure helpers --------------------------------------------------------


def test_to_ms_passes_through_ints_and_none() -> None:
    assert _to_ms(1700000000000) == 1700000000000
    assert _to_ms(None) is None


def test_to_ms_parses_digit_strings_and_iso() -> None:
    assert _to_ms("1700000000000") == 1700000000000
    ms = _to_ms("2023-11-14T22:13:20Z")
    assert ms == int(datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC).timestamp() * 1000)
    ms_date_only = _to_ms("2023-11-14")
    assert ms_date_only == int(datetime(2023, 11, 14, tzinfo=UTC).timestamp() * 1000)


def test_to_ms_rejects_short_digit_strings_and_garbage() -> None:
    # "12345" is only 5 digits — ambiguous between an epoch and a bare number, so it
    # must not be silently misread as epoch ms.
    with pytest.raises(ValueError):
        _to_ms("12345")
    with pytest.raises(ValueError):
        _to_ms("not-a-date")


def test_months_ago_ms_subtracts_calendar_months() -> None:
    now_ms = int(datetime(2026, 3, 15, tzinfo=UTC).timestamp() * 1000)
    result = _months_ago_ms(18, now_ms=now_ms)
    expected = int(datetime(2024, 9, 15, tzinfo=UTC).timestamp() * 1000)
    assert result == expected


def test_months_ago_ms_handles_day_overflow() -> None:
    # Aug 31 minus 6 months would be "Feb 31" — must clamp to a valid day.
    now_ms = int(datetime(2026, 8, 31, tzinfo=UTC).timestamp() * 1000)
    result = _months_ago_ms(6, now_ms=now_ms)
    expected = int(datetime(2026, 2, 28, tzinfo=UTC).timestamp() * 1000)
    assert result == expected


def test_wallet_type_name_and_format_amount_handle_none() -> None:
    assert _wallet_type_name(None) == "N/A"
    assert _format_amount(None) == "N/A"


# -- pydantic model validation --------------------------------------------


def test_pay_transactions_input_rejects_bad_time_string() -> None:
    with pytest.raises(ValidationError) as exc_info:
        PayTransactionsInput(start_time="not-a-date")
    assert "start_time must be epoch ms or ISO-8601" in str(exc_info.value)


def test_pay_history_input_rejects_bad_time_string() -> None:
    with pytest.raises(ValidationError) as exc_info:
        PayHistoryInput(since="12345")
    assert "since must be epoch ms or ISO-8601" in str(exc_info.value)


def test_pay_transactions_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        PayTransactionsInput(bogus=1)  # type: ignore[call-arg]


# -- binance_get_pay_transactions ----------------------------------------


async def test_pay_transactions_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "code": "000000",
        "message": "success",
        "data": [
            _tx(1, amount="-12.5", wallet_type=2, receiver={"name": "Coffee Shop"}),
            _tx(2, amount="50.0", wallet_type=4, payer={"name": "Friend"}),
        ],
        "success": True,
    }
    fake = _FakeClient(payload=payload)
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    result = await binance_get_pay_transactions(PayTransactionsInput())

    assert "Binance Pay Transactions" in result
    assert "-12.5 USDT" in result
    assert "+50 USDT" in result
    assert "(spot wallet)" in result
    assert "(card wallet)" in result
    assert "Coffee Shop" in result
    assert "Friend" in result
    assert fake.calls[0][0] == "GET"
    assert fake.calls[-1][1] == "/sapi/v1/pay/transactions"
    assert fake.calls[-1][2] == {"params": {"limit": 20}, "auth": "signed"}  # default limit is 20


async def test_pay_transactions_sends_times_and_drops_none(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"code": "000000", "message": "success", "data": [], "success": True}
    fake = _FakeClient(payload=payload)
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    params = PayTransactionsInput(start_time=1_700_000_000_000, end_time=1_700_000_500_000, limit=10)
    await binance_get_pay_transactions(params)

    assert fake.calls[0][0] == "GET"
    assert fake.calls[-1][2] == {
        "params": {"startTime": 1_700_000_000_000, "endTime": 1_700_000_500_000, "limit": 10},
        "auth": "signed",
    }


async def test_pay_transactions_accepts_iso_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"code": "000000", "message": "success", "data": [], "success": True}
    fake = _FakeClient(payload=payload)
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    params = PayTransactionsInput(start_time="2026-06-01", end_time="2026-06-02")
    await binance_get_pay_transactions(params)

    expected_start = int(datetime(2026, 6, 1, tzinfo=UTC).timestamp() * 1000)
    expected_end = int(datetime(2026, 6, 2, tzinfo=UTC).timestamp() * 1000)
    assert fake.calls[-1][2]["params"]["startTime"] == expected_start
    assert fake.calls[-1][2]["params"]["endTime"] == expected_end


async def test_pay_transactions_wallet_type_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "code": "000000",
        "message": "success",
        "data": [_tx(1, wallet_type=2), _tx(2, wallet_type=4), _tx(3, wallet_type=6)],
        "success": True,
    }
    fake = _FakeClient(payload=payload)
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    result = await binance_get_pay_transactions(PayTransactionsInput(wallet_type=4))

    assert "walletType=4 card" in result
    assert "txId `2`" in result
    assert "txId `1`" not in result
    assert "txId `3`" not in result


async def test_pay_transactions_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"code": "000000", "message": "success", "data": [_tx(1)], "success": True}
    fake = _FakeClient(payload=payload)
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    result = await binance_get_pay_transactions(PayTransactionsInput(response_format="json"))

    assert '"transactionId": 1' in result
    assert '"title": "Binance Pay Transactions"' in result


async def test_pay_transactions_json_carries_full_set_past_display_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [_tx(i) for i in range(1, 61)]  # 60 rows > MAX_DISPLAY_ROWS (50)
    payload = {"code": "000000", "message": "success", "data": items, "success": True}
    fake = _FakeClient(payload=payload)
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    result = await binance_get_pay_transactions(PayTransactionsInput(response_format="json"))

    parsed = json.loads(result)
    assert parsed["count"] == 60
    assert len(parsed["items"]) == 60  # NOT capped at 50 — only markdown display is


async def test_pay_transactions_markdown_truncates_and_points_to_pay_history(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [_tx(i) for i in range(1, 61)]
    payload = {"code": "000000", "message": "success", "data": items, "success": True}
    fake = _FakeClient(payload=payload)
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    result = await binance_get_pay_transactions(PayTransactionsInput())

    assert "Showing **50** of **60** row(s)." in result
    assert "binance_get_pay_history" in result
    assert "txId `51`" not in result


async def test_pay_transactions_rejects_span_over_90_days(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)
    params = PayTransactionsInput(start_time=0, end_time=91 * 86_400_000)

    result = await binance_get_pay_transactions(params)

    assert result.startswith("Error:")
    assert "90 days" in result
    assert fake.calls == []


async def test_pay_transactions_rejects_only_start_time(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)
    monkeypatch.setattr("binance_mcp.tools.pay._now_ms", lambda: 2_000_000_000_000)

    result = await binance_get_pay_transactions(PayTransactionsInput(start_time=0))

    assert result.startswith("Error:")
    assert "give both start_time and end_time" in result
    assert fake.calls == []


async def test_pay_transactions_rejects_only_end_time(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    result = await binance_get_pay_transactions(PayTransactionsInput(end_time=1_700_000_000_000))

    assert result.startswith("Error:")
    assert "give both start_time and end_time" in result
    assert fake.calls == []


async def test_pay_transactions_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(exc=_status_error(401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions."}))
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    result = await binance_get_pay_transactions(PayTransactionsInput())

    assert "Error (401)" in result
    assert "allowlist" in result


# -- binance_get_pay_history ----------------------------------------------


async def test_pay_history_single_window(monkeypatch: pytest.MonkeyPatch) -> None:
    since_ms = 1_700_000_000_000
    until_ms = since_ms + 1_000_000  # well within one 89-day window
    fake = _FakeClient(
        data_by_window={(since_ms, until_ms): [_tx(1, when=since_ms + 500), _tx(2, when=since_ms + 900)]}
    )
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    params = PayHistoryInput(since=since_ms, resume_before=until_ms)
    result = await binance_get_pay_history(params)

    assert "Found **2** unique" in result
    assert "1 call(s) spent" in result
    assert "resume with" not in result
    assert fake.calls[0][:2] == ("GET", "/sapi/v1/pay/transactions")
    assert fake.calls[0][2]["params"] == {"startTime": since_ms, "endTime": until_ms, "limit": 100}
    assert fake.calls[0][2]["auth"] == "signed"


async def test_pay_history_bisects_full_page(monkeypatch: pytest.MonkeyPatch) -> None:
    since_ms = 0
    until_ms = 1_000_000  # single top-level window: [0, 1_000_000]
    mid = (since_ms + until_ms) // 2
    full_page = [_tx(i, when=i) for i in range(100)]
    fake = _FakeClient(
        data_by_window={
            (since_ms, until_ms): full_page,
            (since_ms, mid): [_tx(9001, when=1)],
            (mid + 1, until_ms): [_tx(9002, when=mid + 2)],
        }
    )
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    params = PayHistoryInput(since=since_ms, resume_before=until_ms, max_calls=10)
    result = await binance_get_pay_history(params)

    # The full 100-row page must have been discarded in favour of its two halves.
    assert "Found **2** unique" in result
    assert "3 call(s) spent" in result
    assert fake.calls[0][:2] == ("GET", "/sapi/v1/pay/transactions")
    called_windows = {(c[2]["params"]["startTime"], c[2]["params"]["endTime"]) for c in fake.calls}
    assert (since_ms, until_ms) in called_windows
    assert (since_ms, mid) in called_windows
    assert (mid + 1, until_ms) in called_windows


async def test_pay_history_dedupes_across_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    since_ms = 0
    until_ms = 1_000_000
    mid = (since_ms + until_ms) // 2
    full_page = [_tx(i, when=i) for i in range(100)]
    fake = _FakeClient(
        data_by_window={
            (since_ms, until_ms): full_page,
            # transactionId=1 shows up in both halves (a boundary overlap) on purpose.
            (since_ms, mid): [_tx(0, when=0), _tx(1, when=1)],
            (mid + 1, until_ms): [_tx(1, when=mid + 5), _tx(2, when=mid + 6)],
        }
    )
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    params = PayHistoryInput(since=since_ms, resume_before=until_ms, max_calls=10)
    result = await binance_get_pay_history(params)

    # 4 raw rows collected, 1 duplicate transactionId -> 3 unique.
    assert "Found **3** unique" in result


async def test_pay_history_respects_max_calls_and_returns_resume_before(monkeypatch: pytest.MonkeyPatch) -> None:
    until_ms = 1_000_000_000
    since_ms = until_ms - 5 * WALK_WINDOW_MS  # 5 top-level windows
    fake = _FakeClient(data_by_window={})  # every window returns [] (no data scripted)
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    params = PayHistoryInput(since=since_ms, resume_before=until_ms, max_calls=2)
    result = await binance_get_pay_history(params)

    assert len(fake.calls) == 2
    assert fake.calls[0][:2] == ("GET", "/sapi/v1/pay/transactions")
    # Real progress: two whole top-level windows were fully drained before the budget
    # ran out, so the cursor is two windows older than `until_ms`, not `until_ms` itself.
    expected_resume = until_ms - 2 * WALK_WINDOW_MS
    assert expected_resume < until_ms
    assert f"since={since_ms}, resume_before={expected_resume}" in result
    assert "could not complete even the newest window" not in result


async def test_pay_history_no_window_when_since_after_until() -> None:
    params = PayHistoryInput(since=2_000_000, resume_before=1_000_000)

    result = await binance_get_pay_history(params)

    assert "No window to walk" in result


async def test_pay_history_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    since_ms = 0
    until_ms = 1_000
    fake = _FakeClient(data_by_window={(since_ms, until_ms): [_tx(1, when=500)]})
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    params = PayHistoryInput(since=since_ms, resume_before=until_ms, response_format="json")
    result = await binance_get_pay_history(params)

    assert '"title": "Binance Pay History"' in result
    assert '"resume_before": null' in result
    assert '"no_progress": false' in result
    assert '"possibly_incomplete": false' in result


async def test_pay_history_json_carries_full_set_past_display_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    since_ms = 0
    until_ms = 1_000
    rows = [_tx(i, when=i) for i in range(1, 61)]  # 60 unique rows, one call, no bisection
    fake = _FakeClient(data_by_window={(since_ms, until_ms): rows})
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    params = PayHistoryInput(since=since_ms, resume_before=until_ms, response_format="json")
    result = await binance_get_pay_history(params)

    parsed = json.loads(result)
    assert parsed["count"] == 60
    assert len(parsed["items"]) == 60  # NOT capped at 50 — only markdown display is


async def test_pay_history_markdown_truncates_past_display_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    since_ms = 0
    until_ms = 1_000
    rows = [_tx(i, when=i) for i in range(1, 61)]
    fake = _FakeClient(data_by_window={(since_ms, until_ms): rows})
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    params = PayHistoryInput(since=since_ms, resume_before=until_ms)
    result = await binance_get_pay_history(params)

    assert "Found **60** unique" in result
    assert "Showing the 50 most recent of 60" in result


async def test_pay_history_sort_handles_missing_transaction_time(monkeypatch: pytest.MonkeyPatch) -> None:
    since_ms = 0
    until_ms = 1_000
    rows = [_tx(1, when=500), _tx(2, when=100)]
    rows[1]["transactionTime"] = None  # a row with a null/missing time must not crash the sort
    fake = _FakeClient(data_by_window={(since_ms, until_ms): rows})
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    params = PayHistoryInput(since=since_ms, resume_before=until_ms)
    result = await binance_get_pay_history(params)

    assert "Found **2** unique" in result
    # Newest-first: the row with a null time sorts as epoch 0, i.e. last.
    assert result.index("txId `1`") < result.index("txId `2`")


# -- _walk_pay_history: the review's blocking + should-fix findings -------


async def test_walk_pay_history_no_progress_guard() -> None:
    """A window that STILL returns a full page after every bisection (an unsplittable
    dense instant) must never emit a resume_before that equals the original until_ms —
    that would just repeat the exact same calls forever (the reported blocking bug:
    4 rounds of 30 calls each, zero progress). Instead it must report no_progress and
    omit the cursor entirely.
    """

    class _AlwaysFullClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, Any]]] = []

        async def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
            self.calls.append((method, path, kwargs))
            full_page = [_tx(i, when=i) for i in range(100)]
            return _FakeResponse({"code": "000000", "message": "success", "data": full_page, "success": True})

    fake = _AlwaysFullClient()

    result = await _walk_pay_history(fake, since_ms=0, until_ms=1_000_000, max_calls=1)

    assert result.calls_used == 1
    assert result.resume_before is None
    assert result.no_progress is True
    assert result.transactions == []


async def test_walk_pay_history_resume_round_trip_has_no_gap() -> None:
    until_ms = 1_000_000
    mid = until_ms // 2
    full_page = [_tx(i, when=i) for i in range(100)]
    newer_half = [_tx(200 + i, when=mid + 1 + i) for i in range(40)]
    older_half = [_tx(300 + i, when=i) for i in range(30)]
    fake = _FakeClient(
        data_by_window={
            (0, until_ms): full_page,
            (mid + 1, until_ms): newer_half,
            (0, mid): older_half,
        }
    )

    round1 = await _walk_pay_history(fake, since_ms=0, until_ms=until_ms, max_calls=2)

    assert round1.calls_used == 2
    assert round1.no_progress is False
    assert round1.resume_before is not None
    assert round1.resume_before < until_ms  # genuine progress, strictly older
    round1_ids = {item["transactionId"] for item in round1.transactions}
    assert round1_ids == {200 + i for i in range(40)}

    # Resume: feed the cursor back in as the new upper bound, same since_ms.
    round2 = await _walk_pay_history(fake, since_ms=0, until_ms=round1.resume_before, max_calls=10)

    assert round2.resume_before is None  # fully drained down to `since`
    round2_ids = {item["transactionId"] for item in round2.transactions}
    assert round2_ids == {300 + i for i in range(30)}

    # No gap and no unexpected overlap: the two rounds' ranges tile [0, until_ms) exactly.
    assert round1_ids.isdisjoint(round2_ids)
    assert len(round1_ids | round2_ids) == 70


# -- live smoke (read-only; skipped without creds) -------------------------
# /sapi/v1/pay/transactions does not exist on the spot testnet, so this can only
# ever be exercised against a real account — see conftest.py's skip gating.


@pytest.mark.live
async def test_pay_transactions_live() -> None:
    result = await binance_get_pay_transactions(PayTransactionsInput())
    assert isinstance(result, str)


@pytest.mark.live
async def test_pay_history_live() -> None:
    result = await binance_get_pay_history(PayHistoryInput(max_calls=1))
    assert isinstance(result, str)


# -- t15 live findings (2026-09-23): 18-month boundary + error mid-walk -------------


class _FailOnSecondWindowClient:
    """First pay window answers normally; the second raises the given exception."""

    def __init__(self, first_rows: list[dict[str, Any]], exc: Exception) -> None:
        self._first_rows = first_rows
        self._exc = exc
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, path, kwargs))
        if len(self.calls) >= 2:
            raise self._exc
        return _FakeResponse({"code": "000000", "message": "success", "data": self._first_rows, "success": True})


async def test_walk_pay_history_error_mid_walk_keeps_rows_and_cursor() -> None:
    until_ms = 3 * WALK_WINDOW_MS
    boom = _status_error(400, {"code": 403004, "msg": "The request has an invalid parameter"})
    fake = _FailOnSecondWindowClient([_tx(1, when=until_ms - 10)], boom)

    result = await _walk_pay_history(fake, since_ms=0, until_ms=until_ms, max_calls=10)

    assert [t["transactionId"] for t in result.transactions] == [1]
    assert result.calls_used == 2  # the failed request still counts
    assert result.stop_error is not None and result.stop_error.startswith("Error (400)")
    assert result.no_progress is False
    # The failed range was window 2: [until - 2W, until - W); its upper boundary is the cursor.
    assert result.resume_before == until_ms - WALK_WINDOW_MS


async def test_pay_history_error_mid_walk_renders_rows_cursor_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    until_ms = 3 * WALK_WINDOW_MS
    boom = _status_error(429, {"code": -1003, "msg": "Too many requests."})
    fake = _FailOnSecondWindowClient([_tx(1, when=until_ms - 10)], boom)
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)
    monkeypatch.setattr("binance_mcp.tools.pay._now_ms", lambda: until_ms + 20 * WALK_WINDOW_MS)

    md = await binance_get_pay_history(PayHistoryInput(since=0, resume_before=until_ms))
    fake_json = _FailOnSecondWindowClient([_tx(1, when=until_ms - 10)], boom)  # fresh call counter
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake_json)
    js = json.loads(
        await binance_get_pay_history(PayHistoryInput(since=0, resume_before=until_ms, response_format="json"))
    )

    assert "Found **1** unique" in md
    assert "Stopped early on a request failure: Error (429)" in md
    assert f"resume_before={until_ms - WALK_WINDOW_MS}" in md
    assert js["count"] == 1 and js["stop_error"].startswith("Error (429)")
    assert js["resume_before"] == until_ms - WALK_WINDOW_MS and js["no_progress"] is False


async def test_walk_pay_history_first_request_error_is_no_progress_not_raise() -> None:
    boom = _status_error(500, {"code": -1000, "msg": "An unknown error occurred."})
    fake = _FakeClient(exc=boom)

    result = await _walk_pay_history(fake, since_ms=0, until_ms=WALK_WINDOW_MS, max_calls=5)

    assert result.transactions == [] and result.calls_used == 1
    assert result.no_progress is True and result.resume_before is None
    assert result.stop_error is not None and "UNKNOWN" in result.stop_error


async def test_walk_pay_history_config_error_with_no_rows_propagates() -> None:
    fake = _FakeClient(exc=RuntimeError("No Binance API key configured."))

    with pytest.raises(RuntimeError, match="No Binance API key"):
        await _walk_pay_history(fake, since_ms=0, until_ms=WALK_WINDOW_MS, max_calls=5)


async def test_pay_history_since_is_clamped_to_the_lookback(monkeypatch: pytest.MonkeyPatch) -> None:
    now_ms = 1_790_000_000_000
    monkeypatch.setattr("binance_mcp.tools.pay._now_ms", lambda: now_ms)
    floor_ms = _months_ago_ms(PAY_HISTORY_LOOKBACK_MONTHS, now_ms=now_ms) + PAY_LOOKBACK_MARGIN_MS
    fake = _FakeClient(data_by_window={})  # every window answers empty
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    js = json.loads(await binance_get_pay_history(PayHistoryInput(since="2017-07-01", response_format="json")))
    md = await binance_get_pay_history(PayHistoryInput(since="2017-07-01"))
    default = json.loads(await binance_get_pay_history(PayHistoryInput(response_format="json")))

    assert js["since"] == floor_ms and js["since_clamped"] is True
    assert "clamped to Binance's 18-month lookback" in md
    assert default["since"] == floor_ms and default["since_clamped"] is False
    # No request may start before the floor.
    assert all(c[2]["params"]["startTime"] >= floor_ms for c in fake.calls)
