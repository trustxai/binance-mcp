"""Unit tests for the spot TWAP algo tools against a fake client.

Every tool asserts BOTH the rendered string and the captured
`(method, path, auth, params)` — the exact compact dict, with numbers as strings and
`None` fields dropped. The rejection tests additionally assert `fake.calls == []`:
a malformed algo order must die locally, before anything is signed or sent.

The two gated tools (`binance_place_twap_order`, `binance_cancel_algo_order`) have NO
live or trading test, by design: `/sapi` does not exist on the spot testnet, so the only
place they could run is a real account with real money. The kill-switch message is
tested verbatim instead.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from binance_mcp.client import TradingDisabledError
from binance_mcp.formatters import ResponseFormat
from binance_mcp.tools.spot_algo import (
    CancelAlgoOrderInput,
    GetAlgoOrderHistoryInput,
    GetAlgoSubOrdersInput,
    GetOpenAlgoOrdersInput,
    PlaceTwapOrderInput,
    binance_cancel_algo_order,
    binance_get_algo_order_history,
    binance_get_algo_sub_orders,
    binance_get_open_algo_orders,
    binance_place_twap_order,
)

NEW_TWAP_PATH = "/sapi/v1/algo/spot/newOrderTwap"
CANCEL_PATH = "/sapi/v1/algo/spot/order"
OPEN_PATH = "/sapi/v1/algo/spot/openOrders"
HISTORY_PATH = "/sapi/v1/algo/spot/historicalOrders"
SUB_ORDERS_PATH = "/sapi/v1/algo/spot/subOrders"

CLIENT_ALGO_ID = "abcdefghijklmnopqrstuvwxyz012345"  # exactly 32 characters


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Routes by path; records every call (method, path, kwargs)."""

    def __init__(self, routes: dict[str, Any] | None = None) -> None:
        self._routes = routes or {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, path, kwargs))
        payload = self._routes.get(path)
        if isinstance(payload, Exception):
            raise payload
        return _FakeResponse(payload if payload is not None else {})


def _status_error(method: str, path: str, status: int, body: dict[str, Any]) -> httpx.HTTPStatusError:
    request = httpx.Request(method, f"https://api.binance.com{path}")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def _patch(monkeypatch: pytest.MonkeyPatch, fake: _FakeClient) -> None:
    monkeypatch.setattr("binance_mcp.tools.spot_algo.get_client", lambda: fake)


TWAP_ACCEPTED: dict[str, Any] = {
    "clientAlgoId": CLIENT_ALGO_ID,
    "success": True,
    "code": 0,
    "msg": "OK",
}

ALGO_ORDER: dict[str, Any] = {
    "algoId": 14511,
    "symbol": "BTCUSDT",
    "side": "BUY",
    "totalQty": "0.50000000",
    "executedQty": "0.20000000",
    "executedAmt": "13000.00000000",
    "avgPrice": "65000.00",
    "clientAlgoId": CLIENT_ALGO_ID,
    "bookTime": 1788220800000,
    "endTime": 1788224400000,
    "algoStatus": "WORKING",
    "algoType": "TWAP",
    "urgency": "LOW",
}

SUB_ORDER: dict[str, Any] = {
    "algoId": 14511,
    "orderId": 991,
    "orderStatus": "FILLED",
    "executedQty": "0.05000000",
    "executedAmt": "3250.00000000",
    "feeAmt": "0.00005000",
    "feeAsset": "BTC",
    "bookTime": 1788220860000,
    "avgPrice": "65000.00",
    "side": "BUY",
    "symbol": "BTCUSDT",
    "subId": 1,
    "timeInForce": "IMMEDIATE_OR_CANCEL",
    "origQty": "0.05000000",
}


# -- binance_place_twap_order ---------------------------------------------------------


async def test_place_twap_order_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={NEW_TWAP_PATH: TWAP_ACCEPTED})
    _patch(monkeypatch, fake)

    result = await binance_place_twap_order(
        PlaceTwapOrderInput(symbol="btcusdt", side="BUY", quantity="0.50000000", duration=3600)
    )

    assert result.startswith("# TWAP order accepted on BTCUSDT")
    assert f"- **clientAlgoId**: {CLIENT_ALGO_ID}" in result
    assert "- **success**: True" in result
    assert "- **code**: 0" in result
    assert "- **msg**: OK" in result
    assert "accepted, not executed" in result
    assert "binance_get_open_algo_orders" in result
    assert "binance_get_algo_order_history" in result
    assert fake.calls == [
        (
            "POST",
            NEW_TWAP_PATH,
            {
                "auth": "signed",
                "params": {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "quantity": "0.50000000",
                    "duration": 3600,
                },
            },
        )
    ]


async def test_place_twap_order_sends_optional_params_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """limitPrice and clientAlgoId travel as given; decimals are never re-formatted."""
    fake = _FakeClient(routes={NEW_TWAP_PATH: TWAP_ACCEPTED})
    _patch(monkeypatch, fake)

    result = await binance_place_twap_order(
        PlaceTwapOrderInput(
            symbol="BTCUSDT",
            side="SELL",
            quantity="1.25000000",
            duration=86400,
            limit_price="65000.10",
            client_algo_id=CLIENT_ALGO_ID,
        )
    )

    assert "TWAP order accepted on BTCUSDT" in result
    assert fake.calls == [
        (
            "POST",
            NEW_TWAP_PATH,
            {
                "auth": "signed",
                "params": {
                    "symbol": "BTCUSDT",
                    "side": "SELL",
                    "quantity": "1.25000000",
                    "duration": 86400,
                    "clientAlgoId": CLIENT_ALGO_ID,
                    "limitPrice": "65000.10",
                },
            },
        )
    ]


async def test_place_twap_order_success_false_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `success: false` body arrives with HTTP 200 — it must never read as a confirmation."""
    # In production the client's `_check_envelope` raises BinanceEnvelopeError on this shape
    # before the tool ever sees it; the fake client bypasses that, so this exercises the
    # tool-level net that has to hold if the client's check ever stops covering it.
    fake = _FakeClient(
        routes={
            NEW_TWAP_PATH: {
                "clientAlgoId": "",
                "success": False,
                "code": -1121,
                "msg": "Invalid symbol.",
            }
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_place_twap_order(
        PlaceTwapOrderInput(symbol="BTCUSDT", side="BUY", quantity="0.5", duration=300)
    )

    assert result == "Error: Invalid symbol. (code -1121)"
    assert "accepted" not in result


async def test_place_twap_order_rejects_duration_out_of_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="duration"):
        PlaceTwapOrderInput(symbol="BTCUSDT", side="BUY", quantity="0.5", duration=299)
    with pytest.raises(ValidationError, match="duration"):
        PlaceTwapOrderInput(symbol="BTCUSDT", side="BUY", quantity="0.5", duration=86401)
    assert fake.calls == []


async def test_place_twap_order_rejects_wrong_client_algo_id_length(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="client_algo_id"):
        PlaceTwapOrderInput(
            symbol="BTCUSDT", side="BUY", quantity="0.5", duration=3600, client_algo_id=CLIENT_ALGO_ID[:31]
        )
    with pytest.raises(ValidationError, match="client_algo_id"):
        PlaceTwapOrderInput(
            symbol="BTCUSDT", side="BUY", quantity="0.5", duration=3600, client_algo_id=CLIENT_ALGO_ID + "x"
        )
    assert fake.calls == []


async def test_place_twap_order_rejects_non_decimal_quantity(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="quantity"):
        PlaceTwapOrderInput(symbol="BTCUSDT", side="BUY", quantity="lots", duration=3600)
    with pytest.raises(ValidationError, match="quantity"):
        PlaceTwapOrderInput(symbol="BTCUSDT", side="BUY", quantity="-1", duration=3600)
    assert fake.calls == []


async def test_place_twap_order_unconfirmed_body_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty 200 body declares nothing — on a money path that is not a confirmation."""
    fake = _FakeClient(routes={NEW_TWAP_PATH: {}})
    _patch(monkeypatch, fake)

    result = await binance_place_twap_order(
        PlaceTwapOrderInput(symbol="BTCUSDT", side="BUY", quantity="0.5", duration=3600)
    )

    assert result.startswith("Error: Binance did not confirm the TWAP")
    assert "accepted" not in result


async def test_place_twap_order_rejects_scientific_notation(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Decimal("1e-3")` parses fine; Binance's filters read the literal string and reject it."""
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="scientific"):
        PlaceTwapOrderInput(symbol="BTCUSDT", side="BUY", quantity="1e-3", duration=3600)
    with pytest.raises(ValidationError, match="scientific"):
        PlaceTwapOrderInput(symbol="BTCUSDT", side="BUY", quantity="0.5", duration=3600, limit_price="6.5E4")
    assert fake.calls == []


async def test_place_twap_order_kill_switch_message_is_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client's kill-switch surfaces as `Error: … trading is disabled …`, unaltered."""
    message = (
        "POST /sapi/v1/algo/spot/newOrderTwap would move funds or change account state, but trading is "
        "disabled. Set BINANCE_ALLOW_TRADING=1 to enable order placement/cancellation, transfers, convert "
        "and algo orders. Dry-run validation (POST /api/v3/order/test) works without it."
    )
    fake = _FakeClient(routes={NEW_TWAP_PATH: TradingDisabledError(message)})
    _patch(monkeypatch, fake)

    result = await binance_place_twap_order(
        PlaceTwapOrderInput(symbol="BTCUSDT", side="BUY", quantity="0.5", duration=3600)
    )

    assert result == f"Error: {message}"
    assert "trading is disabled" in result


async def test_place_twap_order_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            NEW_TWAP_PATH: _status_error(
                "POST",
                NEW_TWAP_PATH,
                400,
                {"code": -2010, "msg": "The total notional is less than the minimum (1000 USDT)."},
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_place_twap_order(
        PlaceTwapOrderInput(symbol="BTCUSDT", side="BUY", quantity="0.0001", duration=3600)
    )

    assert result.startswith("Error (400)")
    assert "(code -2010)" in result


# -- binance_cancel_algo_order --------------------------------------------------------


async def test_cancel_algo_order_by_algo_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={CANCEL_PATH: {"algoId": 14511, "success": True, "code": 0, "msg": "OK"}})
    _patch(monkeypatch, fake)

    result = await binance_cancel_algo_order(CancelAlgoOrderInput(algo_id=14511))

    assert result.startswith("# Algo order cancelled")
    assert "- **algoId**: 14511" in result
    assert "- **success**: True" in result
    assert "- **code**: 0" in result
    assert "- **msg**: OK" in result
    assert "UNEXECUTED remainder" in result
    assert fake.calls == [("DELETE", CANCEL_PATH, {"auth": "signed", "params": {"algoId": 14511}})]


async def test_cancel_algo_order_by_client_algo_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={CANCEL_PATH: {"algoId": 14511, "success": True, "code": 0, "msg": "OK"}})
    _patch(monkeypatch, fake)

    result = await binance_cancel_algo_order(CancelAlgoOrderInput(client_algo_id=CLIENT_ALGO_ID))

    assert "Algo order cancelled" in result
    assert fake.calls == [("DELETE", CANCEL_PATH, {"auth": "signed", "params": {"clientAlgoId": CLIENT_ALGO_ID}})]


async def test_cancel_algo_order_requires_exactly_one_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    both = await binance_cancel_algo_order(CancelAlgoOrderInput(algo_id=14511, client_algo_id=CLIENT_ALGO_ID))
    neither = await binance_cancel_algo_order(CancelAlgoOrderInput())

    assert both.startswith("Error: Pass exactly one of algo_id")
    assert neither.startswith("Error: Pass exactly one of algo_id")
    assert fake.calls == []


async def test_cancel_algo_order_success_false_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same as the placement: the client's `_check_envelope` fires first in production, and
    # this asserts the tool-level net underneath it.
    fake = _FakeClient(
        routes={CANCEL_PATH: {"algoId": 14511, "success": False, "code": -1146, "msg": "Order does not exist."}}
    )
    _patch(monkeypatch, fake)

    result = await binance_cancel_algo_order(CancelAlgoOrderInput(algo_id=14511))

    assert result == "Error: Order does not exist. (code -1146)"
    assert "cancelled" not in result


async def test_cancel_algo_order_unconfirmed_body_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty 200 body is not a cancellation — the order may well still be working."""
    fake = _FakeClient(routes={CANCEL_PATH: {}})
    _patch(monkeypatch, fake)

    result = await binance_cancel_algo_order(CancelAlgoOrderInput(algo_id=14511))

    assert result.startswith("Error: Binance did not confirm the TWAP")
    assert "cancelled" not in result


async def test_cancel_algo_order_rejects_short_client_algo_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cancel reuses the placement's exact-32 bound, so a truncated id dies locally."""
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="client_algo_id"):
        CancelAlgoOrderInput(client_algo_id=CLIENT_ALGO_ID[:31])
    with pytest.raises(ValidationError, match="algo_id"):
        CancelAlgoOrderInput(algo_id=0)
    assert fake.calls == []


async def test_cancel_algo_order_kill_switch_message_is_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client's kill-switch surfaces as `Error: … trading is disabled …`, unaltered."""
    message = (
        "DELETE /sapi/v1/algo/spot/order would move funds or change account state, but trading is disabled. "
        "Set BINANCE_ALLOW_TRADING=1 to enable order placement/cancellation, transfers, convert and algo "
        "orders. Dry-run validation (POST /api/v3/order/test) works without it."
    )
    fake = _FakeClient(routes={CANCEL_PATH: TradingDisabledError(message)})
    _patch(monkeypatch, fake)

    result = await binance_cancel_algo_order(CancelAlgoOrderInput(algo_id=14511))

    assert result == f"Error: {message}"
    assert "trading is disabled" in result


async def test_cancel_algo_order_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            CANCEL_PATH: _status_error(
                "DELETE", CANCEL_PATH, 401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_cancel_algo_order(CancelAlgoOrderInput(algo_id=14511))

    assert result.startswith("Error (401)")
    assert "(code -2015)" in result


# -- binance_get_open_algo_orders -----------------------------------------------------


async def test_get_open_algo_orders_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={OPEN_PATH: {"total": 1, "orders": [ALGO_ORDER]}})
    _patch(monkeypatch, fake)

    result = await binance_get_open_algo_orders(GetOpenAlgoOrdersInput())

    assert result.startswith("# Open spot algo orders")
    assert "| 14511 | BTCUSDT | BUY | WORKING | TWAP |" in result
    assert "2026-09-01 00:00:00 UTC" in result  # bookTime rendered by epoch_to_human
    assert "0.5" in result and "0.2" in result  # totalQty / executedQty via fmt_num
    assert "LOW" in result
    assert fake.calls == [("GET", OPEN_PATH, {"auth": "signed", "params": {}})]


async def test_get_open_algo_orders_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={OPEN_PATH: {"total": 0, "orders": []}})
    _patch(monkeypatch, fake)

    result = await binance_get_open_algo_orders(GetOpenAlgoOrdersInput())

    assert "_No algo orders._" in result
    assert fake.calls == [("GET", OPEN_PATH, {"auth": "signed", "params": {}})]


async def test_get_open_algo_orders_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON keeps the fields the table drops — clientAlgoId above all."""
    fake = _FakeClient(routes={OPEN_PATH: {"total": 1, "orders": [ALGO_ORDER]}})
    _patch(monkeypatch, fake)

    result = await binance_get_open_algo_orders(GetOpenAlgoOrdersInput(response_format=ResponseFormat.JSON))

    assert '"algoId": 14511' in result
    assert f'"clientAlgoId": "{CLIENT_ALGO_ID}"' in result


async def test_get_open_algo_orders_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            OPEN_PATH: _status_error(
                "GET", OPEN_PATH, 401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_open_algo_orders(GetOpenAlgoOrdersInput())

    assert result.startswith("Error (401)")
    assert "(code -2015)" in result


# -- binance_get_algo_order_history ---------------------------------------------------


async def test_get_algo_order_history_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """symbol and side are optional (S3 supersedes S1): the default call carries neither."""
    fake = _FakeClient(routes={HISTORY_PATH: {"total": 1, "orders": [ALGO_ORDER]}})
    _patch(monkeypatch, fake)

    result = await binance_get_algo_order_history(GetAlgoOrderHistoryInput())

    assert result.startswith("# Historical spot algo orders (page 1)")
    assert "| 14511 | BTCUSDT | BUY | WORKING | TWAP |" in result
    assert fake.calls == [("GET", HISTORY_PATH, {"auth": "signed", "params": {"page": 1, "pageSize": 100}})]


async def test_get_algo_order_history_all_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """ISO-8601 timestamps are normalised to epoch ms; the symbol is uppercased."""
    fake = _FakeClient(routes={HISTORY_PATH: {"total": 1, "orders": [ALGO_ORDER]}})
    _patch(monkeypatch, fake)

    result = await binance_get_algo_order_history(
        GetAlgoOrderHistoryInput(
            symbol="btcusdt",
            side="BUY",
            start_time="2026-09-01T00:00:00Z",
            end_time=1788307200000,
            page=2,
            page_size=20,
        )
    )

    assert "Historical spot algo orders on BTCUSDT (page 2)" in result
    assert fake.calls == [
        (
            "GET",
            HISTORY_PATH,
            {
                "auth": "signed",
                "params": {
                    "page": 2,
                    "pageSize": 20,
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "startTime": 1788220800000,
                    "endTime": 1788307200000,
                },
            },
        )
    ]


async def test_get_algo_order_history_rejects_bad_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    result = await binance_get_algo_order_history(GetAlgoOrderHistoryInput(start_time="last tuesday"))

    assert result.startswith("Error: start_time must be an epoch-ms integer")
    assert fake.calls == []


async def test_get_algo_order_history_rejects_inverted_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    result = await binance_get_algo_order_history(
        GetAlgoOrderHistoryInput(start_time="2026-09-23", end_time="2026-09-01")
    )

    assert result == "Error: end_time must be after start_time."
    assert fake.calls == []


async def test_get_algo_order_history_rejects_page_size_over_100(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="page_size"):
        GetAlgoOrderHistoryInput(page_size=101)
    with pytest.raises(ValidationError, match="page"):
        GetAlgoOrderHistoryInput(page=0)
    assert fake.calls == []


async def test_get_algo_order_history_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={HISTORY_PATH: _status_error("GET", HISTORY_PATH, 400, {"code": -1121, "msg": "Invalid symbol."})}
    )
    _patch(monkeypatch, fake)

    result = await binance_get_algo_order_history(GetAlgoOrderHistoryInput(symbol="NOSUCH"))

    assert result.startswith("Error (400)")
    assert "(code -1121)" in result


# -- binance_get_algo_sub_orders ------------------------------------------------------


async def test_get_algo_sub_orders_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            SUB_ORDERS_PATH: {
                "total": 1,
                "executedQty": "0.05000000",
                "executedAmt": "3250.00000000",
                "subOrders": [SUB_ORDER],
            }
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_algo_sub_orders(GetAlgoSubOrdersInput(algo_id=14511))

    assert result.startswith("# Sub-orders of algo order 14511")
    assert "- **executedQty**: 0.05" in result
    assert "- **executedAmt**: 3250" in result
    assert "| 1 | 991 | BTCUSDT | BUY | FILLED |" in result
    assert "0.00005 BTC" in result  # feeAmt + feeAsset
    assert fake.calls == [
        ("GET", SUB_ORDERS_PATH, {"auth": "signed", "params": {"algoId": 14511, "page": 1, "pageSize": 100}})
    ]


async def test_get_algo_sub_orders_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An accepted TWAP that has not traded yet: totals of zero and no rows."""
    fake = _FakeClient(routes={SUB_ORDERS_PATH: {"total": 0, "executedQty": "0", "executedAmt": "0", "subOrders": []}})
    _patch(monkeypatch, fake)

    result = await binance_get_algo_sub_orders(GetAlgoSubOrdersInput(algo_id=14511, page=3, page_size=50))

    assert "_No sub-orders yet — the TWAP has not traded._" in result
    assert fake.calls == [
        ("GET", SUB_ORDERS_PATH, {"auth": "signed", "params": {"algoId": 14511, "page": 3, "pageSize": 50}})
    ]


async def test_get_algo_sub_orders_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            SUB_ORDERS_PATH: {
                "total": 1,
                "executedQty": "0.05000000",
                "executedAmt": "3250.00000000",
                "subOrders": [SUB_ORDER],
            }
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_algo_sub_orders(
        GetAlgoSubOrdersInput(algo_id=14511, response_format=ResponseFormat.JSON)
    )

    assert '"orderId": 991' in result
    assert '"timeInForce": "IMMEDIATE_OR_CANCEL"' in result


async def test_get_algo_sub_orders_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            SUB_ORDERS_PATH: _status_error(
                "GET", SUB_ORDERS_PATH, 400, {"code": -1102, "msg": "Mandatory parameter 'algoId' was not sent."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_algo_sub_orders(GetAlgoSubOrdersInput(algo_id=1))

    assert result.startswith("Error (400)")
    assert "(code -1102)" in result


async def test_get_algo_sub_orders_requires_algo_id() -> None:
    """algoId is mandatory on this endpoint — pydantic refuses the call without it."""
    with pytest.raises(ValidationError, match="algo_id"):
        GetAlgoSubOrdersInput()


# -- live smoke (read tools only) -----------------------------------------------------


@pytest.mark.live
async def test_live_smoke_read_tools() -> None:
    """Smoke-test the three READ tools against the real configured account.

    Read-only by construction: neither gated tool runs here, and there is no testnet
    fallback for them — **the spot testnet has no `/sapi` endpoints at all**, so the only
    place a TWAP placement or cancellation could execute is a real account with real
    money. That test does not exist, deliberately.

    `binance_get_algo_sub_orders` needs an algoId that belongs to this account, so it is
    exercised only when the history returns one; an account that has never run a TWAP
    still passes the other two. Requires BINANCE_API_KEY + secret/PEM (see the conftest's
    live-marker gating).
    """
    open_orders = await binance_get_open_algo_orders(GetOpenAlgoOrdersInput(response_format=ResponseFormat.JSON))
    history = await binance_get_algo_order_history(
        GetAlgoOrderHistoryInput(page_size=5, response_format=ResponseFormat.JSON)
    )
    for result in (open_orders, history):
        assert isinstance(result, str)
        assert result
        assert not result.startswith("Error")

    orders = json.loads(history).get("orders") or []
    if orders:
        sub_orders = await binance_get_algo_sub_orders(GetAlgoSubOrdersInput(algo_id=int(orders[0]["algoId"])))
        assert not sub_orders.startswith("Error")
