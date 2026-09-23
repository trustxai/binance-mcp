"""Unit tests for `binance_mcp.tools.fiat` against a fake client."""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from binance_mcp.tools.fiat import (
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
    _, _, kwargs = fake.calls[0]
    assert kwargs["params"] == {
        "transactionType": 1,
        "page": 1,
        "rows": 100,
        "beginTime": 1706745600000,
        "endTime": 1706832000000,
    }


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
    _, _, kwargs = fake.calls[0]
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
    _, _, kwargs = fake.calls[0]
    assert kwargs["params"]["transactionType"] == 1


async def test_fiat_payments_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(queue=[_status_error("/sapi/v1/fiat/payments", 401, {"code": -2015, "msg": "Invalid API-key."})])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_payments(FiatPaymentsInput(transaction_type="buy"))

    assert result.startswith("Error (401)")


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
    _, path, kwargs = fake.calls[0]
    assert path == "/sapi/v1/fiat/orders"
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
    assert fake.calls[0][2]["params"]["page"] == 1
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
    assert "fell back to 30-day windows" in result
    assert "**B1**" in result
    assert "Totals by crypto currency received:" in result
    assert "BTC: 0.002" in result


async def test_fiat_history_budget_exhausted_returns_resume_before(monkeypatch: pytest.MonkeyPatch) -> None:
    full_page = _orders_envelope(
        [_order_row("H1", "1.00", "USD", 1700000000000), _order_row("H2", "1.00", "USD", 1700000001000)], total=99
    )
    fake = _FakeClient(queue=[full_page])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    result = await binance_get_fiat_history(FiatHistoryInput(kind="withdrawals", rows=2, max_calls=1))

    assert len(fake.calls) == 1
    assert "Stopped early" in result
    # oldest createTime fetched (H1 = 1700000000000) minus 1 ms.
    assert "resume_before=1699999999999" in result


async def test_fiat_history_resume_before_input_becomes_end_time(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = _orders_envelope([_order_row("H1", "1.00", "USD", 1699999999000)], total=1)
    fake = _FakeClient(queue=[envelope])
    monkeypatch.setattr("binance_mcp.tools.fiat.get_client", lambda: fake)

    await binance_get_fiat_history(FiatHistoryInput(kind="deposits", resume_before=1700000000999))

    _, _, kwargs = fake.calls[0]
    assert kwargs["params"]["endTime"] == 1700000000999


def test_fiat_history_rejects_bad_kind() -> None:
    with pytest.raises(ValidationError):
        FiatHistoryInput(kind="bogus")  # type: ignore[arg-type]
