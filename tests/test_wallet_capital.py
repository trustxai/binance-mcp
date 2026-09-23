"""Unit tests for the wallet capital tools against a fake client.

The walk tests are the load-bearing ones: they pin the two invariants the first wave of
walk tools got wrong — the budget is checked BEFORE every request (and the reported
call count is real), and `resume_before` is the boundary of the next UNFETCHED range,
taken from the loop's own `window_end`, never derived from the rows that came back.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from binance_mcp.formatters import ResponseFormat
from binance_mcp.tools.wallet_capital import (
    AllDepositsInput,
    AllWithdrawalsInput,
    CoinConfigInput,
    DepositAddressesInput,
    DepositAddressInput,
    DepositHistoryInput,
    DepositStatus,
    WithdrawHistoryInput,
    WithdrawStatus,
    binance_get_all_deposits,
    binance_get_all_withdrawals,
    binance_get_coin_config,
    binance_get_deposit_address,
    binance_get_deposit_addresses,
    binance_get_deposit_history,
    binance_get_withdraw_history,
)

MODULE = "binance_mcp.tools.wallet_capital"

DAY = 24 * 60 * 60 * 1000
WINDOW = 89 * DAY
UNTIL = 1_800_000_000_000  # fixed "now" for the walk tests (2027-01-15T08:00:00Z)


# -- fakes ----------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Routes by path; records every call as (method, path, kwargs)."""

    def __init__(self, routes: dict[str, Any] | None = None) -> None:
        self._routes = routes or {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, path, kwargs))
        payload = self._routes.get(path)
        if isinstance(payload, Exception):
            raise payload
        return _FakeResponse(payload if payload is not None else {})


class _WalkClient:
    """Serves a scripted queue of pages (or exceptions), then falls back to `default`.

    `default` receives the request's `params`, so a test can make every window answer
    with a row derived from its own boundaries — which is what makes the chained-resume
    test able to prove "no gap and no duplicate".
    """

    def __init__(
        self,
        queue: list[Any] | None = None,
        default: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.queue = list(queue or [])
        self.default = default
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, path, kwargs))
        params = dict(kwargs.get("params") or {})
        if self.queue:
            item = self.queue.pop(0)
        elif self.default is not None:
            item = self.default(params)
        else:
            item = []
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(params)
        return _FakeResponse(item)


def _status_error(status: int, body: dict[str, Any], path: str = "/sapi/v1/capital/x") -> httpx.HTTPStatusError:
    request = httpx.Request("GET", f"https://api.binance.com{path}")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def _deposit(idx: int, *, coin: str = "USDT", amount: str = "10", ms: int = 1_700_000_000_000) -> dict[str, Any]:
    return {
        "id": idx,
        "amount": amount,
        "coin": coin,
        "network": "TRX",
        "status": 1,
        "address": "TAddr",
        "txId": f"tx{idx}",
        "insertTime": ms,
        "completeTime": ms,
        "walletType": 0,
    }


def _withdrawal(idx: int, *, coin: str = "BTC", amount: str = "0.5", fee: str = "0.0005") -> dict[str, Any]:
    return {
        "id": f"w{idx}",
        "amount": amount,
        "transactionFee": fee,
        "coin": coin,
        "status": 6,
        "address": "bc1addr",
        "txId": f"wtx{idx}",
        "applyTime": "2026-09-01 10:00:00",
        "completeTime": 1_756_720_000_000,
        "network": "BTC",
        "withdrawOrderId": f"ord{idx}",
        "walletType": 1,
    }


def _walk_json(result: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(result)
    return payload


def _windows(client: _WalkClient) -> list[tuple[int, int]]:
    """The (startTime, endTime) pairs actually requested, deduped, oldest first."""
    seen = {(call[2]["params"]["startTime"], call[2]["params"]["endTime"]) for call in client.calls}
    return sorted(seen)


# -- binance_get_deposit_history -------------------------------------------------------


async def test_deposit_history_happy_path_exact_params(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/capital/deposit/hisrec": [_deposit(1), _deposit(2, amount="5")]})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_deposit_history(
        DepositHistoryInput(
            coin="usdt",
            status=DepositStatus.SUCCESS,
            start_time=UNTIL - 10 * DAY,
            end_time=UNTIL,
            offset=0,
            limit=500,
            include_source=True,
            tx_id="tx1",
        )
    )

    assert "Binance deposit history" in result
    assert "| USDT | 2 | 15 |" in result  # per-coin totals: 10 + 5
    assert "success (1)" in result
    assert "`tx1`" in result
    assert fake.calls == [
        (
            "GET",
            "/sapi/v1/capital/deposit/hisrec",
            {
                "auth": "signed",
                "params": {
                    "offset": 0,
                    "limit": 500,
                    "startTime": UNTIL - 10 * DAY,
                    "endTime": UNTIL,
                    "coin": "USDT",
                    "status": 1,
                    "includeSource": True,
                    "txId": "tx1",
                },
            },
        )
    ]


async def test_deposit_history_minimal_params_drops_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/capital/deposit/hisrec": []})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_deposit_history(DepositHistoryInput())

    assert "No records in range" in result
    assert fake.calls == [
        ("GET", "/sapi/v1/capital/deposit/hisrec", {"auth": "signed", "params": {"offset": 0, "limit": 1000}})
    ]


async def test_deposit_history_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/capital/deposit/hisrec": [_deposit(1)]})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_deposit_history(DepositHistoryInput(response_format=ResponseFormat.JSON))
    payload = json.loads(result)

    assert payload["count"] == 1
    assert payload["totals"] == {"USDT": {"count": 1, "amount": "10"}}
    assert payload["items"][0]["txId"] == "tx1"


async def test_deposit_history_window_over_90_days_rejected_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both-sided rejection: nothing is sent to Binance at all."""
    fake = _FakeClient()
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_deposit_history(DepositHistoryInput(start_time=UNTIL - 91 * DAY, end_time=UNTIL))

    assert result.startswith("Error")
    assert "90 days" in result
    assert "binance_get_all_deposits" in result
    assert fake.calls == []


async def test_deposit_history_one_sided_window_rejected_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    """start_time alone: end_time defaults to now, so the span is still validated."""
    fake = _FakeClient()
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    result = await binance_get_deposit_history(DepositHistoryInput(start_time=now_ms - 120 * DAY))

    assert result.startswith("Error")
    assert "end_time defaults to now" in result
    assert fake.calls == []


async def test_deposit_history_end_time_only_fills_start_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    """end_time alone gets startTime = end - 89 days, so the window brackets it."""
    fake = _FakeClient(routes={"/sapi/v1/capital/deposit/hisrec": []})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    await binance_get_deposit_history(DepositHistoryInput(end_time=UNTIL))

    assert fake.calls[0][2]["params"] == {
        "offset": 0,
        "limit": 1000,
        "startTime": UNTIL - WINDOW,
        "endTime": UNTIL,
    }


async def test_deposit_history_start_after_end_rejected_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_deposit_history(DepositHistoryInput(start_time=UNTIL, end_time=UNTIL - DAY))

    assert result == "Error: end_time must be after start_time."
    assert fake.calls == []


async def test_deposit_history_accepts_iso_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/capital/deposit/hisrec": []})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    await binance_get_deposit_history(DepositHistoryInput(start_time="2026-08-01", end_time="2026-09-01"))

    assert fake.calls[0][2]["params"]["startTime"] == 1785542400000
    assert fake.calls[0][2]["params"]["endTime"] == 1788220800000


async def test_deposit_history_invalid_time_returns_named_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_deposit_history(DepositHistoryInput(start_time="not-a-date"))

    assert result.startswith("Error: start_time must be")
    assert fake.calls == []


async def test_deposit_history_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/capital/deposit/hisrec": _status_error(
                401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."}
            )
        }
    )
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_deposit_history(DepositHistoryInput())

    assert result.startswith("Error (401)")
    assert "allowlist" in result


async def test_deposit_history_display_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_deposit(i) for i in range(130)]
    fake = _FakeClient(routes={"/sapi/v1/capital/deposit/hisrec": rows})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_deposit_history(DepositHistoryInput())

    assert "80 more row(s) not shown" in result  # MAX_DISPLAY_ROWS = 50


# -- binance_get_withdraw_history ------------------------------------------------------


async def test_withdraw_history_happy_path_exact_params(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/capital/withdraw/history": [_withdrawal(1), _withdrawal(2)]})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_withdraw_history(
        WithdrawHistoryInput(
            coin="btc",
            status=WithdrawStatus.COMPLETED,
            id_list=["aaa", "bbb"],
            start_time=UNTIL - 10 * DAY,
            end_time=UNTIL,
            limit=1000,
        )
    )

    assert "Binance withdrawal history" in result
    assert "completed (6)" in result
    assert "| BTC | 2 | 1 | 0.001 |" in result  # amount 0.5+0.5, fees 0.0005+0.0005
    assert fake.calls == [
        (
            "GET",
            "/sapi/v1/capital/withdraw/history",
            {
                "auth": "signed",
                "params": {
                    "offset": 0,
                    "limit": 1000,
                    "startTime": UNTIL - 10 * DAY,
                    "endTime": UNTIL,
                    "coin": "BTC",
                    "status": 6,
                    "idList": "aaa,bbb",
                },
            },
        )
    ]


async def test_withdraw_history_window_over_90_days_rejected_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_withdraw_history(WithdrawHistoryInput(start_time=UNTIL - 95 * DAY, end_time=UNTIL))

    assert result.startswith("Error")
    assert "90 days" in result
    assert fake.calls == []


async def test_withdraw_history_order_id_narrows_window_to_7_days(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binance's own rule: with withdrawOrderId the window must be under 7 days."""
    fake = _FakeClient()
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_withdraw_history(
        WithdrawHistoryInput(withdraw_order_id="ord1", start_time=UNTIL - 8 * DAY, end_time=UNTIL)
    )

    assert result.startswith("Error")
    assert "7 days" in result
    assert fake.calls == []


async def test_withdraw_history_order_id_one_sided_window_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    result = await binance_get_withdraw_history(
        WithdrawHistoryInput(withdraw_order_id="ord1", start_time=now_ms - 10 * DAY)
    )

    assert result.startswith("Error")
    assert "7 days" in result
    assert fake.calls == []


async def test_withdraw_history_order_id_end_only_fills_6_day_start(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/capital/withdraw/history": []})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    await binance_get_withdraw_history(WithdrawHistoryInput(withdraw_order_id="ord1", end_time=UNTIL))

    assert fake.calls[0][2]["params"] == {
        "offset": 0,
        "limit": 1000,
        "startTime": UNTIL - 6 * DAY,
        "endTime": UNTIL,
        "withdrawOrderId": "ord1",
    }


async def test_withdraw_history_id_list_over_45_rejected_by_model() -> None:
    with pytest.raises(ValidationError):
        WithdrawHistoryInput(id_list=[f"id{i}" for i in range(46)])


async def test_withdraw_history_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """429 here means the endpoint's 10 req/s ceiling was hit."""
    fake = _FakeClient(
        routes={
            "/sapi/v1/capital/withdraw/history": _status_error(
                429, {"code": -1003, "msg": "Too many requests; current limit is 10 requests per second."}
            )
        }
    )
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_withdraw_history(WithdrawHistoryInput())

    assert result.startswith("Error (429)")
    assert "do not retry in a tight loop" in result


# -- binance_get_all_deposits (the walk) -----------------------------------------------


async def test_all_deposits_multi_window_with_offset_paging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two windows; the first needs two pages (1000 then 200)."""
    since = UNTIL - 100 * DAY
    page_full = [_deposit(i) for i in range(1000)]
    page_tail = [_deposit(1000 + i) for i in range(200)]
    page_old = [_deposit(1200 + i) for i in range(5)]
    fake = _WalkClient(queue=[page_full, page_tail, page_old])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_all_deposits(
        AllDepositsInput(since=since, until=UNTIL, response_format=ResponseFormat.JSON)
    )
    payload = _walk_json(result)

    assert payload["count"] == 1205
    assert payload["calls_made"] == 3
    assert payload["truncated"] is False
    assert payload["resume_before"] is None
    assert payload["totals"] == {"USDT": {"count": 1205, "amount": "12050"}}
    assert fake.calls == [
        (
            "GET",
            "/sapi/v1/capital/deposit/hisrec",
            {
                "auth": "signed",
                "params": {"startTime": UNTIL - WINDOW, "endTime": UNTIL, "offset": 0, "limit": 1000},
            },
        ),
        (
            "GET",
            "/sapi/v1/capital/deposit/hisrec",
            {
                "auth": "signed",
                "params": {"startTime": UNTIL - WINDOW, "endTime": UNTIL, "offset": 1000, "limit": 1000},
            },
        ),
        (
            "GET",
            "/sapi/v1/capital/deposit/hisrec",
            {
                "auth": "signed",
                "params": {"startTime": since, "endTime": UNTIL - WINDOW, "offset": 0, "limit": 1000},
            },
        ),
    ]


async def test_all_deposits_budget_exhausted_at_window_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cursor is the NEXT unfetched boundary, not anything read off the rows."""
    since = UNTIL - 200 * DAY
    fake = _WalkClient(queue=[[_deposit(1), _deposit(2)]])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_all_deposits(
        AllDepositsInput(since=since, until=UNTIL, max_calls=1, response_format=ResponseFormat.JSON)
    )
    payload = _walk_json(result)

    assert payload["calls_made"] == 1
    assert payload["count"] == 2
    assert payload["truncated"] is True
    assert payload["resume_before"] == UNTIL - WINDOW  # the boundary of window 2, untouched
    # The rows' own timestamps are 1_700_000_000_000 — the cursor is not derived from them.
    assert payload["resume_before"] != 1_700_000_000_000
    assert len(fake.calls) == 1


async def test_all_deposits_stop_inside_first_window_emits_no_progress_not_a_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping inside the FIRST window leaves `resume_before == until`, which is not a
    resume but a loop: a chained call would re-issue the identical request and get the
    identical cursor back forever. The cursor is suppressed and `no_progress` says so."""
    since = UNTIL - 200 * DAY
    fake = _WalkClient(queue=[[_deposit(i) for i in range(1000)]])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_all_deposits(
        AllDepositsInput(since=since, until=UNTIL, max_calls=1, response_format=ResponseFormat.JSON)
    )
    payload = _walk_json(result)

    assert payload["calls_made"] == 1
    assert payload["count"] == 1000  # partial rows preserved
    assert payload["truncated"] is True  # it really did not reach `since`
    assert payload["no_progress"] is True
    assert payload["resume_before"] is None  # nothing to chain on
    assert len(fake.calls) == 1


async def test_all_deposits_mid_window_stop_after_a_completed_window_still_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard is scoped to "no window finished" — a mid-window stop that follows a
    completed window still yields a usable cursor at that window's own end."""
    since = UNTIL - 300 * DAY
    fake = _WalkClient(queue=[[_deposit(1)], [_deposit(100 + i) for i in range(1000)]])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    payload = _walk_json(
        await binance_get_all_deposits(
            AllDepositsInput(since=since, until=UNTIL, max_calls=2, response_format=ResponseFormat.JSON)
        )
    )

    assert payload["no_progress"] is False
    assert payload["resume_before"] == UNTIL - WINDOW  # window 2's own end: it is re-fetched
    assert payload["count"] == 1001


async def test_all_deposits_no_progress_markdown_refuses_to_offer_a_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The markdown must not print a "resume with ..." instruction that loops."""
    since = UNTIL - 200 * DAY
    fake = _WalkClient(queue=[[_deposit(i) for i in range(1000)]])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_all_deposits(AllDepositsInput(since=since, until=UNTIL, max_calls=1))

    assert "no forward progress" in result
    assert "raise `max_calls`" in result
    assert "Resume with" not in result


async def test_all_deposits_no_progress_second_run_is_not_chained_blindly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loop this guard prevents, made explicit: with the cursor suppressed there is
    nothing to feed back, and re-running with `resume_before = until` reproduces the
    identical single call and the identical no-progress verdict."""
    since = UNTIL - 200 * DAY

    first = _WalkClient(queue=[[_deposit(i) for i in range(1000)]])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: first)
    run1 = _walk_json(
        await binance_get_all_deposits(
            AllDepositsInput(since=since, until=UNTIL, max_calls=1, response_format=ResponseFormat.JSON)
        )
    )
    assert run1["resume_before"] is None, "a caller chaining on the cursor has nothing to chain"

    # Chaining on `until` anyway (what the old code invited) makes zero progress:
    second = _WalkClient(queue=[[_deposit(i) for i in range(1000)]])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: second)
    run2 = _walk_json(
        await binance_get_all_deposits(
            AllDepositsInput(since=since, resume_before=UNTIL, max_calls=1, response_format=ResponseFormat.JSON)
        )
    )

    assert second.calls == first.calls, "byte-identical request: no forward progress"
    assert run2["no_progress"] is True
    assert run2["resume_before"] is None

    # And the escape hatch the guard points at does work: a bigger budget moves on.
    # 3 calls drain window 1 (a full page then a short one) and window 2 (empty), so the
    # cursor lands two windows back and IS chainable.
    third = _WalkClient(queue=[[_deposit(i) for i in range(1000)], [_deposit(9999)]])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: third)
    run3 = _walk_json(
        await binance_get_all_deposits(
            AllDepositsInput(since=since, until=UNTIL, max_calls=3, response_format=ResponseFormat.JSON)
        )
    )

    assert run3["no_progress"] is False
    assert run3["resume_before"] == UNTIL - 2 * WINDOW
    assert run3["calls_made"] == 3


async def test_all_deposits_budget_exhausted_markdown_mentions_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    since = UNTIL - 200 * DAY
    fake = _WalkClient(queue=[[_deposit(1)]])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_all_deposits(AllDepositsInput(since=since, until=UNTIL, max_calls=1))

    assert "Stopped early" in result
    assert f"`since={since}`" in result
    assert f"`resume_before={UNTIL - WINDOW}`" in result


async def test_all_deposits_chained_resume_has_no_gap_and_no_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feed the emitted cursor back in; the union must tile [since, until] exactly."""
    since = UNTIL - 200 * DAY

    def one_row_per_window(params: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "id": f"d-{params['startTime']}",
                "coin": "USDT",
                "amount": "1",
                "network": "TRX",
                "status": 1,
                "insertTime": params["endTime"],
                "completeTime": params["endTime"],
                "walletType": 0,
                "txId": f"tx-{params['startTime']}",
            }
        ]

    first = _WalkClient(default=one_row_per_window)
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: first)
    run1 = _walk_json(
        await binance_get_all_deposits(
            AllDepositsInput(since=since, until=UNTIL, max_calls=2, response_format=ResponseFormat.JSON)
        )
    )

    assert run1["truncated"] is True
    assert run1["resume_before"] == UNTIL - 2 * WINDOW
    assert run1["count"] == 2

    second = _WalkClient(default=one_row_per_window)
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: second)
    run2 = _walk_json(
        await binance_get_all_deposits(
            AllDepositsInput(
                since=since,
                resume_before=run1["resume_before"],
                response_format=ResponseFormat.JSON,
            )
        )
    )

    assert run2["truncated"] is False
    assert run2["resume_before"] is None

    ids = [row["id"] for row in run1["items"]] + [row["id"] for row in run2["items"]]
    assert len(ids) == len(set(ids)), "the resumed run must not re-report a row already returned"

    windows = sorted(_windows(first) + _windows(second))
    assert windows[0][0] == since, "the union must start exactly at `since`"
    assert windows[-1][1] == UNTIL, "the union must end exactly at `until`"
    for previous, following in zip(windows, windows[1:], strict=False):
        assert previous[1] == following[0], f"gap between {previous} and {following}"


async def test_all_deposits_resume_before_overrides_until(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _WalkClient(default=lambda _params: [])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    await binance_get_all_deposits(AllDepositsInput(since=UNTIL - 10 * DAY, until=UNTIL, resume_before=UNTIL - 5 * DAY))

    assert fake.calls[0][2]["params"]["endTime"] == UNTIL - 5 * DAY


async def test_all_deposits_error_mid_walk_returns_partial_rows_and_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    since = UNTIL - 200 * DAY
    fake = _WalkClient(queue=[[_deposit(1)], _status_error(500, {"code": -1000, "msg": "Internal error."})])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_all_deposits(
        AllDepositsInput(since=since, until=UNTIL, response_format=ResponseFormat.JSON)
    )
    payload = _walk_json(result)

    assert payload["count"] == 1  # the row from before the failure survives
    assert payload["calls_made"] == 2  # the failed request is counted
    assert payload["resume_before"] == UNTIL - WINDOW
    assert len(fake.calls) == 2


async def test_all_deposits_auth_error_stops_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auth is never tolerated: one request, no retry. It fails inside the first window,
    so the no-progress guard fires instead of handing back an unusable cursor."""
    since = UNTIL - 200 * DAY
    fake = _WalkClient(
        queue=[_status_error(401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."})]
    )
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_all_deposits(AllDepositsInput(since=since, until=UNTIL))

    assert len(fake.calls) == 1
    assert "Error (401)" in result
    assert "no forward progress" in result
    assert "Resume with" not in result


async def test_all_deposits_span_error_is_tolerated_by_halving_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """-1127 is the ONE tolerated failure: the window halves and the walk continues."""
    since = UNTIL - 100 * DAY
    fake = _WalkClient(
        queue=[_status_error(400, {"code": -1127, "msg": "More than xx hours aggregate data is not available."})],
        default=lambda _params: [],
    )
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_all_deposits(
        AllDepositsInput(since=since, until=UNTIL, response_format=ResponseFormat.JSON)
    )
    payload = _walk_json(result)

    assert payload["truncated"] is False  # the walk recovered and reached `since`
    assert fake.calls[1][2]["params"] == {
        "startTime": UNTIL - WINDOW // 2,
        "endTime": UNTIL,
        "offset": 0,
        "limit": 1000,
    }


async def test_all_deposits_coin_filter_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _WalkClient(default=lambda _params: [])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    await binance_get_all_deposits(AllDepositsInput(since=UNTIL - 10 * DAY, until=UNTIL, coin="usdt"))

    assert fake.calls == [
        (
            "GET",
            "/sapi/v1/capital/deposit/hisrec",
            {
                "auth": "signed",
                "params": {
                    "coin": "USDT",
                    "startTime": UNTIL - 10 * DAY,
                    "endTime": UNTIL,
                    "offset": 0,
                    "limit": 1000,
                },
            },
        )
    ]


async def test_all_deposits_rows_sorted_newest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _WalkClient(
        queue=[
            [
                _deposit(1, ms=1_700_000_000_000),
                _deposit(2, ms=1_750_000_000_000),
                _deposit(3, ms=1_720_000_000_000),
            ]
        ]
    )
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    payload = _walk_json(
        await binance_get_all_deposits(
            AllDepositsInput(since=UNTIL - 10 * DAY, until=UNTIL, response_format=ResponseFormat.JSON)
        )
    )

    assert [row["id"] for row in payload["items"]] == [2, 3, 1]


async def test_all_deposits_until_before_since_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _WalkClient()
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_all_deposits(AllDepositsInput(since=UNTIL, until=UNTIL - DAY))

    assert result.startswith("Error")
    assert fake.calls == []


async def test_all_deposits_default_since_is_binance_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """No account-creation endpoint exists, so the floor is the exchange's launch."""
    fake = _WalkClient(default=lambda _params: [])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    payload = _walk_json(
        await binance_get_all_deposits(AllDepositsInput(until=UNTIL, max_calls=1, response_format=ResponseFormat.JSON))
    )

    assert payload["since"] == 1498867200000  # 2017-07-01T00:00:00Z


# -- binance_get_all_withdrawals (the walk) --------------------------------------------


async def test_all_withdrawals_happy_path_exact_params_and_fee_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    since = UNTIL - 10 * DAY
    fake = _WalkClient(queue=[[_withdrawal(1), _withdrawal(2)]])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_all_withdrawals(AllWithdrawalsInput(since=since, until=UNTIL))

    assert "Binance withdrawals — full history walk" in result
    assert "| BTC | 2 | 1 | 0.001 |" in result
    assert "Complete: the walk reached `since`" in result
    assert fake.calls == [
        (
            "GET",
            "/sapi/v1/capital/withdraw/history",
            {
                "auth": "signed",
                "params": {"startTime": since, "endTime": UNTIL, "offset": 0, "limit": 1000},
            },
        )
    ]


async def test_all_withdrawals_default_budget_is_ten_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """UID weight 18000 per call: the default budget is one minute of UID budget."""
    fake = _WalkClient(default=lambda _params: [])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    payload = _walk_json(
        await binance_get_all_withdrawals(
            AllWithdrawalsInput(since=UNTIL - 2000 * DAY, until=UNTIL, response_format=ResponseFormat.JSON)
        )
    )

    assert payload["calls_made"] == 10
    assert payload["truncated"] is True
    assert payload["resume_before"] == UNTIL - 10 * WINDOW
    assert len(fake.calls) == 10


async def test_all_withdrawals_error_path_returns_partial_and_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    since = UNTIL - 200 * DAY
    fake = _WalkClient(
        queue=[
            [_withdrawal(1)],
            _status_error(429, {"code": -1003, "msg": "Too many requests."}),
        ]
    )
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    payload = _walk_json(
        await binance_get_all_withdrawals(
            AllWithdrawalsInput(since=since, until=UNTIL, response_format=ResponseFormat.JSON)
        )
    )

    assert payload["count"] == 1
    assert payload["calls_made"] == 2
    assert payload["resume_before"] == UNTIL - WINDOW


async def test_all_withdrawals_string_apply_time_sorts_newest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`applyTime` comes back as a "YYYY-MM-DD HH:MM:SS" string, not an epoch."""
    older = {**_withdrawal(1), "applyTime": "2020-01-01 00:00:00"}
    newer = {**_withdrawal(2), "applyTime": "2026-09-01 10:00:00"}
    fake = _WalkClient(queue=[[older, newer]])
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    payload = _walk_json(
        await binance_get_all_withdrawals(
            AllWithdrawalsInput(since=UNTIL - 10 * DAY, until=UNTIL, response_format=ResponseFormat.JSON)
        )
    )

    assert [row["id"] for row in payload["items"]] == ["w2", "w1"]


# -- binance_get_deposit_address -------------------------------------------------------


async def test_deposit_address_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/capital/deposit/address": {
                "coin": "USDT",
                "address": "TAddr123",
                "tag": "",
                "url": "https://tronscan.org/#/address/TAddr123",
            }
        }
    )
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_deposit_address(DepositAddressInput(coin="usdt", network="TRX", amount="10.5"))

    assert "`TAddr123`" in result
    assert "wrong-network transfer is not recoverable" in result
    assert fake.calls == [
        (
            "GET",
            "/sapi/v1/capital/deposit/address",
            {"auth": "signed", "params": {"coin": "USDT", "network": "TRX", "amount": "10.5"}},
        )
    ]


async def test_deposit_address_minimal_params(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/capital/deposit/address": {"coin": "BTC", "address": "bc1x"}})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    await binance_get_deposit_address(DepositAddressInput(coin="BTC"))

    assert fake.calls == [("GET", "/sapi/v1/capital/deposit/address", {"auth": "signed", "params": {"coin": "BTC"}})]


async def test_deposit_address_rejects_non_decimal_amount() -> None:
    with pytest.raises(ValidationError):
        DepositAddressInput(coin="BTC", amount="ten")


async def test_deposit_address_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={"/sapi/v1/capital/deposit/address": _status_error(400, {"code": -4001, "msg": "Invalid network."})}
    )
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_deposit_address(DepositAddressInput(coin="BTC", network="NOPE"))

    assert result.startswith("Error (400)")
    assert "Invalid network." in result


# -- binance_get_deposit_addresses -----------------------------------------------------


async def test_deposit_addresses_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/capital/deposit/address/list": [
                {"coin": "USDT", "address": "TAddr", "network": "TRX", "isDefault": True, "tag": ""},
                {"coin": "USDT", "address": "0xAddr", "network": "ETH", "isDefault": False, "tag": ""},
            ]
        }
    )
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_deposit_addresses(DepositAddressesInput(coin="usdt", network="TRX"))

    assert "| TRX | `TAddr` | _none_ | True |" in result
    assert fake.calls == [
        (
            "GET",
            "/sapi/v1/capital/deposit/address/list",
            {"auth": "signed", "params": {"coin": "USDT", "network": "TRX"}},
        )
    ]


async def test_deposit_addresses_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/capital/deposit/address/list": []})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_deposit_addresses(DepositAddressesInput(coin="BTC"))

    assert "No addresses issued" in result
    assert fake.calls[0][2]["params"] == {"coin": "BTC"}


async def test_deposit_addresses_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/capital/deposit/address/list": _status_error(
                401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."}
            )
        }
    )
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_deposit_addresses(DepositAddressesInput(coin="BTC"))

    assert result.startswith("Error (401)")


# -- binance_get_coin_config -----------------------------------------------------------

COIN_ROWS = [
    {
        "coin": "BTC",
        "name": "Bitcoin",
        "depositAllEnable": True,
        "withdrawAllEnable": True,
        "networkList": [
            {
                "network": "BTC",
                "isDefault": True,
                "depositEnable": True,
                "withdrawEnable": True,
                "withdrawFee": "0.00020000",
                "withdrawMin": "0.00100000",
            }
        ],
    },
    {
        "coin": "USDT",
        "name": "TetherUS",
        "depositAllEnable": True,
        "withdrawAllEnable": False,
        "networkList": [
            {
                "network": "TRX",
                "isDefault": False,
                "depositEnable": True,
                "withdrawEnable": False,
                "withdrawFee": "1",
                "withdrawMin": "10",
            }
        ],
    },
]


async def test_coin_config_happy_path_no_params_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/capital/config/getall": COIN_ROWS})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_coin_config(CoinConfigInput())

    assert "**2** coin(s)" in result
    assert "| BTC | True | True | True | 0.0002 | 0.001 |" in result
    assert "withdrawals enabled (all networks)**: False" in result
    assert fake.calls == [("GET", "/sapi/v1/capital/config/getall", {"auth": "signed"})]


async def test_coin_config_filters_client_side(monkeypatch: pytest.MonkeyPatch) -> None:
    """The endpoint takes no filter — `coin` is applied after the call."""
    fake = _FakeClient(routes={"/sapi/v1/capital/config/getall": COIN_ROWS})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_coin_config(CoinConfigInput(coin="usdt"))

    assert "**1** coin(s)" in result
    assert "TetherUS" in result
    assert "Bitcoin" not in result
    assert fake.calls == [("GET", "/sapi/v1/capital/config/getall", {"auth": "signed"})]


async def test_coin_config_unknown_coin_is_empty_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/capital/config/getall": COIN_ROWS})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_coin_config(CoinConfigInput(coin="NOPE"))

    assert "No coin named NOPE is listed" in result


async def test_coin_config_caps_at_50_coins(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"coin": f"C{i}", "name": f"Coin {i}", "networkList": []} for i in range(60)]
    fake = _FakeClient(routes={"/sapi/v1/capital/config/getall": rows})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_coin_config(CoinConfigInput())

    assert "10 more coin(s) not shown" in result


async def test_coin_config_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/capital/config/getall": COIN_ROWS})
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    payload = json.loads(await binance_get_coin_config(CoinConfigInput(response_format=ResponseFormat.JSON)))

    assert payload["count"] == 2


async def test_coin_config_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/capital/config/getall": _status_error(
                401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."}
            )
        }
    )
    monkeypatch.setattr(f"{MODULE}.get_client", lambda: fake)

    result = await binance_get_coin_config(CoinConfigInput())

    assert result.startswith("Error (401)")


# -- the endpoint that must not exist --------------------------------------------------


async def test_no_withdrawal_tool_is_exported() -> None:
    """Belt and braces on top of the client's FORBIDDEN_PATHS: this module must not
    contain anything that could submit a withdrawal."""
    import binance_mcp.tools.wallet_capital as module

    names = [name for name in dir(module) if name.startswith("binance_")]
    assert names == sorted(
        [
            "binance_get_all_deposits",
            "binance_get_all_withdrawals",
            "binance_get_coin_config",
            "binance_get_deposit_address",
            "binance_get_deposit_addresses",
            "binance_get_deposit_history",
            "binance_get_withdraw_history",
        ]
    )
    # Nothing in this module issues a non-GET request: every path it touches is a read.
    # Walked as an AST rather than grepped, so a `request(` split over lines, renamed,
    # or built from a variable method cannot slip past a substring count.
    source = (module.__file__ or "").replace(".pyc", ".py")
    with open(source) as handle:
        tree = ast.parse(handle.read())
    methods = [
        node.args[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "request"
    ]
    assert methods, "expected at least one client.request(...) call to inspect"
    for arg in methods:
        assert isinstance(arg, ast.Constant), f"HTTP method must be a literal, got {ast.dump(arg)}"
        assert arg.value == "GET", f"non-GET request in a read-only module: {arg.value!r}"


# -- live smoke ------------------------------------------------------------------------


@pytest.mark.live
async def test_live_smoke_read_tools() -> None:
    """Smoke-test the read tools against the real configured account.

    Requires BINANCE_API_KEY + secret/PEM in the environment or .env (see conftest's
    live-marker gating). Every endpoint here is under `/sapi`, which the spot **testnet
    does not serve** — so this module has no `@pytest.mark.trading` counterpart, and
    running these against BINANCE_TESTNET=1 returns 404s.

    The windows are deliberately tiny: `binance_get_withdraw_history` costs UID weight
    18000 per call (and tops out at 10 req/s), so the walk below is budgeted to a
    single call.
    """
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    since = now_ms - 5 * DAY

    results = [
        await binance_get_deposit_history(DepositHistoryInput(start_time=since, limit=10)),
        await binance_get_withdraw_history(WithdrawHistoryInput(start_time=since, limit=10)),
        await binance_get_coin_config(CoinConfigInput(coin="BTC")),
        await binance_get_deposit_address(DepositAddressInput(coin="BTC")),
        await binance_get_deposit_addresses(DepositAddressesInput(coin="BTC")),
    ]
    for result in results:
        assert isinstance(result, str)
        assert result
        assert not result.startswith("Error")


@pytest.mark.live
async def test_live_smoke_walks_are_budgeted() -> None:
    """The two walks against the real account, over a 5-day span.

    `binance_get_all_withdrawals` is pinned to `max_calls=1` on purpose: one call is
    UID weight 18000 of the 180000/min per-account budget.
    """
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    since = now_ms - 5 * DAY

    deposits = await binance_get_all_deposits(AllDepositsInput(since=since, max_calls=2))
    withdrawals = await binance_get_all_withdrawals(AllWithdrawalsInput(since=since, max_calls=1))

    for result in (deposits, withdrawals):
        assert isinstance(result, str)
        assert "full history walk" in result
