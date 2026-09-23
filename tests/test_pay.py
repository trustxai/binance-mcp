"""Unit tests for tools/pay.py against a fake client."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from binance_mcp.tools.pay import (
    PayHistoryInput,
    PayTransactionsInput,
    _months_ago_ms,
    _to_ms,
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
    assert fake.calls[-1][1] == "/sapi/v1/pay/transactions"
    assert fake.calls[-1][2] == {"params": {"limit": 100}, "auth": "signed"}


async def test_pay_transactions_sends_times_and_drops_none(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"code": "000000", "message": "success", "data": [], "success": True}
    fake = _FakeClient(payload=payload)
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    params = PayTransactionsInput(start_time=1_700_000_000_000, end_time=1_700_000_500_000, limit=10)
    await binance_get_pay_transactions(params)

    assert fake.calls[-1][2] == {
        "params": {"startTime": 1_700_000_000_000, "endTime": 1_700_000_500_000, "limit": 10},
        "auth": "signed",
    }


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


async def test_pay_transactions_rejects_span_over_90_days() -> None:
    params = PayTransactionsInput(start_time=0, end_time=91 * 86_400_000)

    result = await binance_get_pay_transactions(params)

    assert result.startswith("Error:")
    assert "90 days" in result


async def test_pay_transactions_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(exc=_status_error(401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions."}))
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    result = await binance_get_pay_transactions(PayTransactionsInput())

    assert "Error (401)" in result
    assert "allowlist" in result


async def test_pay_transactions_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        PayTransactionsInput(bogus=1)  # type: ignore[call-arg]


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
    since_ms = until_ms - 5 * (89 * 24 * 60 * 60 * 1000)  # 5 top-level windows
    fake = _FakeClient(data_by_window={})  # every window returns [] (no data scripted)
    monkeypatch.setattr("binance_mcp.tools.pay.get_client", lambda: fake)

    params = PayHistoryInput(since=since_ms, resume_before=until_ms, max_calls=2)
    result = await binance_get_pay_history(params)

    assert len(fake.calls) == 2
    assert "⚠️" in result
    assert "resume with" in result


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
