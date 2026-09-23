"""Unit tests for the spot order tools against a fake client.

Every tool asserts BOTH the rendered string and the captured
`(method, path, auth, params)` — the exact compact dict, with numbers as strings and
`None` fields dropped. The validation tests additionally assert `fake.calls == []`:
a malformed order must die locally, before anything is signed or sent.
"""

from __future__ import annotations

import os
import time
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from binance_mcp.client import TradingDisabledError
from binance_mcp.formatters import ResponseFormat
from binance_mcp.tools.spot_orders import (
    CancelAllOpenOrdersInput,
    CancelOrderInput,
    CancelReplaceOrderInput,
    GetAllOrdersInput,
    GetOpenOrdersInput,
    GetOrderInput,
    PlaceOrderInput,
    TestOrderInput,
    binance_cancel_all_open_orders,
    binance_cancel_order,
    binance_cancel_replace_order,
    binance_get_all_orders,
    binance_get_open_orders,
    binance_get_order,
    binance_place_order,
    binance_test_order,
)


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
    monkeypatch.setattr("binance_mcp.tools.spot_orders.get_client", lambda: fake)


LIMIT_ORDER: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "side": "BUY",
    "type": "LIMIT",
    "time_in_force": "GTC",
    "quantity": "0.00100000",
    "price": "20000.10",
}

FULL_RESPONSE: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "orderId": 28,
    "orderListId": -1,
    "clientOrderId": "my-entry-001",
    "transactTime": 1758500000000,
    "price": "20000.10",
    "origQty": "0.00100000",
    "executedQty": "0.00100000",
    "cummulativeQuoteQty": "20.00010000",
    "status": "FILLED",
    "timeInForce": "GTC",
    "type": "LIMIT",
    "side": "BUY",
    "workingTime": 1758500000000,
    "selfTradePreventionMode": "NONE",
    "fills": [
        {"price": "20000.10", "qty": "0.00050000", "commission": "0.00000050", "commissionAsset": "BTC", "tradeId": 1},
        {"price": "20000.10", "qty": "0.00050000", "commission": "0.00000050", "commissionAsset": "BTC", "tradeId": 2},
    ],
}


# -- binance_test_order ---------------------------------------------------------------


async def test_test_order_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/order/test": {}})
    _patch(monkeypatch, fake)

    result = await binance_test_order(TestOrderInput(**LIMIT_ORDER))

    assert "Order validation passed" in result
    assert "NOT sent to the order book" in result
    assert fake.calls == [
        (
            "POST",
            "/api/v3/order/test",
            {
                "auth": "signed",
                "params": {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "type": "LIMIT",
                    "timeInForce": "GTC",
                    "quantity": "0.00100000",
                    "price": "20000.10",
                },
            },
        )
    ]


async def test_test_order_with_commission_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    """`computeCommissionRates` is only sent when True, and the breakdown is rendered."""
    payload = {"standardCommissionForOrder": {"maker": "0.00100000", "taker": "0.00100000"}}
    fake = _FakeClient(routes={"/api/v3/order/test": payload})
    _patch(monkeypatch, fake)

    result = await binance_test_order(
        TestOrderInput(symbol="btcusdt", side="SELL", type="MARKET", quantity="0.001", compute_commission_rates=True)
    )

    assert "Commission rates for this order" in result
    assert "standardCommissionForOrder" in result
    assert fake.calls == [
        (
            "POST",
            "/api/v3/order/test",
            {
                "auth": "signed",
                "params": {
                    "symbol": "BTCUSDT",
                    "side": "SELL",
                    "type": "MARKET",
                    "quantity": "0.001",
                    "computeCommissionRates": True,
                },
            },
        )
    ]


async def test_test_order_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/order/test": _status_error(
                "POST", "/api/v3/order/test", 400, {"code": -1013, "msg": "Filter failure: LOT_SIZE"}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_test_order(TestOrderInput(**LIMIT_ORDER))

    assert result.startswith("Error (400)")
    assert "LOT_SIZE" in result


# -- binance_place_order --------------------------------------------------------------


async def test_place_order_full_response(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/order": FULL_RESPONSE})
    _patch(monkeypatch, fake)

    result = await binance_place_order(PlaceOrderInput(**LIMIT_ORDER, new_client_order_id="my-entry-001"))

    assert "# Order placed on BTCUSDT" in result
    assert "**orderId**: 28" in result
    assert "**clientOrderId**: my-entry-001" in result
    assert "**status**: FILLED" in result
    assert "**executedQty**: 0.001" in result
    assert "**cummulativeQuoteQty**: 20.0001" in result
    assert "**Fills**" in result
    assert "Commission across the fills above: 0.000001 BTC" in result
    assert fake.calls == [
        (
            "POST",
            "/api/v3/order",
            {
                "auth": "signed",
                "params": {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "type": "LIMIT",
                    "timeInForce": "GTC",
                    "quantity": "0.00100000",
                    "price": "20000.10",
                    "newClientOrderId": "my-entry-001",
                },
            },
        )
    ]


async def test_place_order_ack_response_claims_nothing_more(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ACK carries no status: the confirmation must say so, not imply a fill."""
    ack = {
        "symbol": "BTCUSDT",
        "orderId": 29,
        "orderListId": -1,
        "clientOrderId": "ack-1",
        "transactTime": 1758500000000,
    }
    fake = _FakeClient(routes={"/api/v3/order": ack})
    _patch(monkeypatch, fake)

    result = await binance_place_order(PlaceOrderInput(**LIMIT_ORDER, new_order_resp_type="ACK"))

    assert "acknowledgement only" in result
    assert "FILLED" not in result
    assert "executedQty" not in result
    assert fake.calls[0][2]["params"]["newOrderRespType"] == "ACK"


async def test_place_order_market_quote_order_qty(monkeypatch: pytest.MonkeyPatch) -> None:
    resting = {
        "symbol": "BTCUSDT",
        "orderId": 30,
        "orderListId": -1,
        "clientOrderId": "x",
        "transactTime": 1758500000000,
        "status": "NEW",
        "executedQty": "0.00000000",
        "cummulativeQuoteQty": "0.00000000",
        "origQuoteOrderQty": "50.00000000",
    }
    fake = _FakeClient(routes={"/api/v3/order": resting})
    _patch(monkeypatch, fake)

    result = await binance_place_order(
        PlaceOrderInput(
            symbol="BTCUSDT",
            side="BUY",
            type="MARKET",
            quote_order_qty="50",
            self_trade_prevention_mode="EXPIRE_MAKER",
        )
    )

    assert "still working on the book" in result
    assert fake.calls == [
        (
            "POST",
            "/api/v3/order",
            {
                "auth": "signed",
                "params": {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "type": "MARKET",
                    "quoteOrderQty": "50",
                    "selfTradePreventionMode": "EXPIRE_MAKER",
                },
            },
        )
    ]


async def test_place_order_kill_switch_message_is_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client's kill-switch surfaces as `Error: … trading is disabled …`, unaltered."""
    message = (
        "POST /api/v3/order would move funds or change account state, but trading is disabled. "
        "Set BINANCE_ALLOW_TRADING=1 to enable order placement/cancellation, transfers, convert and "
        "algo orders. Dry-run validation (POST /api/v3/order/test) works without it."
    )
    fake = _FakeClient(routes={"/api/v3/order": TradingDisabledError(message)})
    _patch(monkeypatch, fake)

    result = await binance_place_order(PlaceOrderInput(**LIMIT_ORDER))

    assert result == f"Error: {message}"
    assert "trading is disabled" in result


async def test_place_order_error_path_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/order": _status_error(
                "POST",
                "/api/v3/order",
                400,
                {"code": -2010, "msg": "Account has insufficient balance for requested action."},
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_place_order(PlaceOrderInput(**LIMIT_ORDER))

    assert result.startswith("Error (400)")
    assert "(code -2010)" in result
    assert "LOT_SIZE" in result  # the -2010 hint names the filter families


# -- binance_get_order ----------------------------------------------------------------


ORDER_ROW: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "orderId": 28,
    "orderListId": -1,
    "clientOrderId": "my-entry-001",
    "price": "20000.10",
    "origQty": "0.00100000",
    "executedQty": "0.00000000",
    "cummulativeQuoteQty": "0.00000000",
    "status": "NEW",
    "timeInForce": "GTC",
    "type": "LIMIT",
    "side": "BUY",
    "time": 1758500000000,
    "updateTime": 1758500000000,
    "isWorking": True,
}


async def test_get_order_by_order_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/order": ORDER_ROW})
    _patch(monkeypatch, fake)

    result = await binance_get_order(GetOrderInput(symbol="BTCUSDT", order_id=28))

    assert "# Order 28 on BTCUSDT" in result
    assert "**status**: NEW" in result
    assert fake.calls == [("GET", "/api/v3/order", {"auth": "signed", "params": {"symbol": "BTCUSDT", "orderId": 28}})]


async def test_get_order_by_client_order_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/order": ORDER_ROW})
    _patch(monkeypatch, fake)

    result = await binance_get_order(GetOrderInput(symbol="btcusdt", orig_client_order_id="my-entry-001"))

    assert "my-entry-001" in result
    assert fake.calls == [
        (
            "GET",
            "/api/v3/order",
            {"auth": "signed", "params": {"symbol": "BTCUSDT", "origClientOrderId": "my-entry-001"}},
        )
    ]


async def test_get_order_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/order": ORDER_ROW})
    _patch(monkeypatch, fake)

    result = await binance_get_order(GetOrderInput(symbol="BTCUSDT", order_id=28, response_format=ResponseFormat.JSON))

    assert '"orderId": 28' in result


async def test_get_order_rejects_both_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    result = await binance_get_order(GetOrderInput(symbol="BTCUSDT", order_id=28, orig_client_order_id="x"))

    assert result.startswith("Error: Pass exactly one of order_id")
    assert fake.calls == []


async def test_get_order_rejects_no_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    result = await binance_get_order(GetOrderInput(symbol="BTCUSDT"))

    assert result.startswith("Error: Pass exactly one of order_id")
    assert fake.calls == []


async def test_get_order_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/order": _status_error("GET", "/api/v3/order", 400, {"code": -2013, "msg": "Order does not exist."})
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_order(GetOrderInput(symbol="BTCUSDT", order_id=28))

    assert result.startswith("Error (400)")
    assert "(code -2013)" in result


# -- binance_cancel_order -------------------------------------------------------------


CANCELLED: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "origClientOrderId": "my-entry-001",
    "orderId": 28,
    "orderListId": -1,
    "clientOrderId": "cancel-1",
    "price": "20000.10",
    "origQty": "0.00100000",
    "executedQty": "0.00000000",
    "cummulativeQuoteQty": "0.00000000",
    "status": "CANCELED",
    "timeInForce": "GTC",
    "type": "LIMIT",
    "side": "BUY",
}


async def test_cancel_order_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/order": CANCELLED})
    _patch(monkeypatch, fake)

    result = await binance_cancel_order(CancelOrderInput(symbol="BTCUSDT", order_id=28))

    assert "# Order cancelled on BTCUSDT" in result
    assert "**status**: CANCELED" in result
    assert "**origClientOrderId**: my-entry-001" in result
    assert fake.calls == [
        ("DELETE", "/api/v3/order", {"auth": "signed", "params": {"symbol": "BTCUSDT", "orderId": 28}})
    ]


async def test_cancel_order_with_restrictions(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/order": CANCELLED})
    _patch(monkeypatch, fake)

    result = await binance_cancel_order(
        CancelOrderInput(
            symbol="BTCUSDT",
            orig_client_order_id="my-entry-001",
            new_client_order_id="cancel-1",
            cancel_restrictions="ONLY_NEW",
        )
    )

    assert "CANCELED" in result
    assert fake.calls == [
        (
            "DELETE",
            "/api/v3/order",
            {
                "auth": "signed",
                "params": {
                    "symbol": "BTCUSDT",
                    "origClientOrderId": "my-entry-001",
                    "newClientOrderId": "cancel-1",
                    "cancelRestrictions": "ONLY_NEW",
                },
            },
        )
    ]


async def test_cancel_order_rejects_both_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    result = await binance_cancel_order(CancelOrderInput(symbol="BTCUSDT", order_id=28, orig_client_order_id="x"))

    assert result.startswith("Error: Pass exactly one of order_id")
    assert fake.calls == []


async def test_cancel_order_error_path_unknown_order(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/order": _status_error(
                "DELETE", "/api/v3/order", 400, {"code": -2011, "msg": "Unknown order sent."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_cancel_order(CancelOrderInput(symbol="BTCUSDT", order_id=28))

    assert result.startswith("Error (400)")
    assert "(code -2011)" in result
    assert "already filled or cancelled" in result


# -- binance_cancel_all_open_orders ---------------------------------------------------


async def test_cancel_all_open_orders_renders_orders_and_list_legs(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [
        {**CANCELLED, "orderId": 28},
        {
            "orderListId": 99,
            "contingencyType": "OCO",
            "listStatusType": "ALL_DONE",
            "listOrderStatus": "ALL_DONE",
            "orderReports": [
                {**CANCELLED, "orderId": 31, "orderListId": 99},
                {**CANCELLED, "orderId": 32, "orderListId": 99, "type": "STOP_LOSS_LIMIT"},
            ],
        },
    ]
    fake = _FakeClient(routes={"/api/v3/openOrders": payload})
    _patch(monkeypatch, fake)

    result = await binance_cancel_all_open_orders(CancelAllOpenOrdersInput(symbol="BTCUSDT"))

    assert "cancelled **1** order(s) and **1** order list(s)" in result
    assert "## Order list 99 (OCO)" in result
    assert "| 31 |" in result
    assert "| 32 |" in result
    assert fake.calls == [("DELETE", "/api/v3/openOrders", {"auth": "signed", "params": {"symbol": "BTCUSDT"}})]


async def test_cancel_all_open_orders_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/openOrders": []})
    _patch(monkeypatch, fake)

    result = await binance_cancel_all_open_orders(CancelAllOpenOrdersInput(symbol="ETHUSDT"))

    assert "there were no open orders on this symbol" in result
    assert fake.calls == [("DELETE", "/api/v3/openOrders", {"auth": "signed", "params": {"symbol": "ETHUSDT"}})]


async def test_cancel_all_open_orders_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/openOrders": _status_error(
                "DELETE", "/api/v3/openOrders", 400, {"code": -2011, "msg": "Unknown order sent."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_cancel_all_open_orders(CancelAllOpenOrdersInput(symbol="BTCUSDT"))

    assert result.startswith("Error (400)")
    assert "(code -2011)" in result


# -- binance_cancel_replace_order -----------------------------------------------------


async def test_cancel_replace_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "cancelResult": "SUCCESS",
        "newOrderResult": "SUCCESS",
        "cancelResponse": {**CANCELLED, "orderId": 28},
        "newOrderResponse": {**FULL_RESPONSE, "orderId": 33, "status": "NEW", "fills": []},
    }
    fake = _FakeClient(routes={"/api/v3/order/cancelReplace": payload})
    _patch(monkeypatch, fake)

    result = await binance_cancel_replace_order(
        CancelReplaceOrderInput(
            **LIMIT_ORDER,
            cancel_replace_mode="STOP_ON_FAILURE",
            cancel_order_id=28,
        )
    )

    assert "**cancelResult**: SUCCESS" in result
    assert "**newOrderResult**: SUCCESS" in result
    assert "## Cancelled order" in result
    assert "## New order" in result
    assert "**orderId**: 33" in result
    assert fake.calls == [
        (
            "POST",
            "/api/v3/order/cancelReplace",
            {
                "auth": "signed",
                "params": {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "type": "LIMIT",
                    "timeInForce": "GTC",
                    "quantity": "0.00100000",
                    "price": "20000.10",
                    "cancelReplaceMode": "STOP_ON_FAILURE",
                    "cancelOrderId": 28,
                },
            },
        )
    ]


async def test_cancel_replace_allow_failure_flags_dead_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "cancelResult": "SUCCESS",
        "newOrderResult": "FAILURE",
        "cancelResponse": {**CANCELLED, "orderId": 28},
        "newOrderResponse": {"code": -2010, "msg": "Account has insufficient balance for requested action."},
    }
    fake = _FakeClient(routes={"/api/v3/order/cancelReplace": payload})
    _patch(monkeypatch, fake)

    result = await binance_cancel_replace_order(
        CancelReplaceOrderInput(
            **LIMIT_ORDER,
            cancel_replace_mode="ALLOW_FAILURE",
            cancel_orig_client_order_id="my-entry-001",
            cancel_restrictions="ONLY_NEW",
            order_rate_limit_exceeded_mode="CANCEL_ONLY",
        )
    )

    assert "The replacement is NOT live" in result
    assert "insufficient balance" in result
    assert fake.calls == [
        (
            "POST",
            "/api/v3/order/cancelReplace",
            {
                "auth": "signed",
                "params": {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "type": "LIMIT",
                    "timeInForce": "GTC",
                    "quantity": "0.00100000",
                    "price": "20000.10",
                    "cancelReplaceMode": "ALLOW_FAILURE",
                    "cancelOrigClientOrderId": "my-entry-001",
                    "cancelRestrictions": "ONLY_NEW",
                    "orderRateLimitExceededMode": "CANCEL_ONLY",
                },
            },
        )
    ]


async def test_cancel_replace_requires_exactly_one_cancel_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="cancel_order_id"):
        CancelReplaceOrderInput(**LIMIT_ORDER, cancel_replace_mode="STOP_ON_FAILURE")
    with pytest.raises(ValidationError, match="cancel_order_id"):
        CancelReplaceOrderInput(
            **LIMIT_ORDER, cancel_replace_mode="STOP_ON_FAILURE", cancel_order_id=28, cancel_orig_client_order_id="x"
        )
    assert fake.calls == []


async def test_cancel_replace_error_path_409(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 409 = the cancel succeeded and the replacement failed."""
    fake = _FakeClient(
        routes={
            "/api/v3/order/cancelReplace": _status_error(
                "POST",
                "/api/v3/order/cancelReplace",
                409,
                {
                    "code": -2021,
                    "msg": "Order cancel-replace partially failed",
                    "data": {"cancelResult": "SUCCESS", "newOrderResult": "FAILURE"},
                },
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_cancel_replace_order(
        CancelReplaceOrderInput(**LIMIT_ORDER, cancel_replace_mode="ALLOW_FAILURE", cancel_order_id=28)
    )

    assert result.startswith("Error (409)")
    assert "Partial success" in result


# -- binance_get_open_orders ----------------------------------------------------------


async def test_get_open_orders_with_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/openOrders": [ORDER_ROW]})
    _patch(monkeypatch, fake)

    result = await binance_get_open_orders(GetOpenOrdersInput(symbol="btcusdt"))

    assert "# Open orders on BTCUSDT" in result
    assert "| 28 |" in result
    assert fake.calls == [("GET", "/api/v3/openOrders", {"auth": "signed", "params": {"symbol": "BTCUSDT"}})]


async def test_get_open_orders_without_symbol_sends_no_params(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/openOrders": []})
    _patch(monkeypatch, fake)

    result = await binance_get_open_orders(GetOpenOrdersInput())

    assert "# Open orders (all symbols)" in result
    assert "_No orders._" in result
    assert fake.calls == [("GET", "/api/v3/openOrders", {"auth": "signed", "params": {}})]


async def test_get_open_orders_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/openOrders": _status_error(
                "GET", "/api/v3/openOrders", 401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_open_orders(GetOpenOrdersInput(symbol="BTCUSDT"))

    assert result.startswith("Error (401)")
    assert "allowlist" in result


# -- binance_get_all_orders -----------------------------------------------------------


async def test_get_all_orders_happy_path_with_iso_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/allOrders": [ORDER_ROW]})
    _patch(monkeypatch, fake)

    result = await binance_get_all_orders(
        GetAllOrdersInput(
            symbol="BTCUSDT",
            start_time="2026-09-22T00:00:00Z",
            end_time="2026-09-22T12:00:00Z",
            limit=1000,
        )
    )

    assert "# Orders on BTCUSDT" in result
    assert "| BTCUSDT | 28 |" in result
    assert fake.calls == [
        (
            "GET",
            "/api/v3/allOrders",
            {
                "auth": "signed",
                "params": {
                    "symbol": "BTCUSDT",
                    "limit": 1000,
                    "startTime": 1790035200000,
                    "endTime": 1790078400000,
                },
            },
        )
    ]


async def test_get_all_orders_with_order_id_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/allOrders": []})
    _patch(monkeypatch, fake)

    result = await binance_get_all_orders(GetAllOrdersInput(symbol="BTCUSDT", order_id=28))

    assert "_No orders._" in result
    assert fake.calls == [
        ("GET", "/api/v3/allOrders", {"auth": "signed", "params": {"symbol": "BTCUSDT", "limit": 500, "orderId": 28}})
    ]


async def test_get_all_orders_display_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{**ORDER_ROW, "orderId": index} for index in range(60)]
    fake = _FakeClient(routes={"/api/v3/allOrders": rows})
    _patch(monkeypatch, fake)

    result = await binance_get_all_orders(GetAllOrdersInput(symbol="BTCUSDT"))

    assert "Showing **50** of **60** order(s)." in result
    assert "_[10 more order(s) not shown" in result
    assert "| 49 |" in result
    assert "| 50 |" not in result


async def test_get_all_orders_rejects_window_over_24h(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    result = await binance_get_all_orders(
        GetAllOrdersInput(symbol="BTCUSDT", start_time="2026-09-20T00:00:00Z", end_time="2026-09-22T00:00:00Z")
    )

    assert result.startswith("Error: the start_time/end_time window spans ~48.0 hours")
    assert "caps it at 24 hours" in result
    assert fake.calls == []


async def test_get_all_orders_rejects_reversed_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    result = await binance_get_all_orders(
        GetAllOrdersInput(symbol="BTCUSDT", start_time=1789128000000, end_time=1789084800000)
    )

    assert result == "Error: end_time must be after start_time."
    assert fake.calls == []


async def test_get_all_orders_rejects_malformed_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    result = await binance_get_all_orders(GetAllOrdersInput(symbol="BTCUSDT", start_time="yesterday"))

    assert result.startswith("Error: start_time must be an epoch-ms integer")
    assert fake.calls == []


async def test_get_all_orders_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/allOrders": _status_error(
                "GET",
                "/api/v3/allOrders",
                400,
                {"code": -1127, "msg": "More than 24 hours between startTime and endTime."},
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_all_orders(GetAllOrdersInput(symbol="BTCUSDT"))

    assert result.startswith("Error (400)")
    assert "(code -1127)" in result


# -- local validation of the per-type mandatory sets ----------------------------------
# Each of these must fail BEFORE any request is signed or sent: `fake.calls == []`.


async def test_limit_requires_time_in_force_quantity_price(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="missing: price"):
        PlaceOrderInput(symbol="BTCUSDT", side="BUY", type="LIMIT", time_in_force="GTC", quantity="0.001")
    with pytest.raises(ValidationError, match="missing: time_in_force"):
        PlaceOrderInput(symbol="BTCUSDT", side="BUY", type="LIMIT", quantity="0.001", price="20000")
    assert fake.calls == []


async def test_market_requires_exactly_one_quantity_field(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="exactly one of quantity"):
        PlaceOrderInput(symbol="BTCUSDT", side="BUY", type="MARKET")
    with pytest.raises(ValidationError, match="exactly one of quantity"):
        PlaceOrderInput(symbol="BTCUSDT", side="BUY", type="MARKET", quantity="0.001", quote_order_qty="50")
    assert fake.calls == []


async def test_stop_loss_requires_quantity_and_exactly_one_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="missing: quantity"):
        PlaceOrderInput(symbol="BTCUSDT", side="SELL", type="STOP_LOSS", stop_price="19000")
    with pytest.raises(ValidationError, match="exactly one trigger"):
        PlaceOrderInput(symbol="BTCUSDT", side="SELL", type="STOP_LOSS", quantity="0.001")
    with pytest.raises(ValidationError, match="exactly one trigger"):
        PlaceOrderInput(
            symbol="BTCUSDT", side="SELL", type="STOP_LOSS", quantity="0.001", stop_price="19000", trailing_delta="100"
        )
    assert fake.calls == []


async def test_take_profit_requires_a_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="exactly one trigger"):
        PlaceOrderInput(symbol="BTCUSDT", side="SELL", type="TAKE_PROFIT", quantity="0.001")
    assert fake.calls == []


async def test_stop_loss_limit_requires_price_tif_and_a_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="missing: price"):
        PlaceOrderInput(
            symbol="BTCUSDT",
            side="SELL",
            type="STOP_LOSS_LIMIT",
            time_in_force="GTC",
            quantity="0.001",
            stop_price="19000",
        )
    with pytest.raises(ValidationError, match="require a trigger"):
        PlaceOrderInput(
            symbol="BTCUSDT",
            side="SELL",
            type="TAKE_PROFIT_LIMIT",
            time_in_force="GTC",
            quantity="0.001",
            price="21000",
        )
    assert fake.calls == []


async def test_stop_loss_limit_accepts_both_triggers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The _LIMIT variants take stopPrice AND/OR trailingDelta (inventory C)."""
    fake = _FakeClient(routes={"/api/v3/order/test": {}})
    _patch(monkeypatch, fake)

    result = await binance_test_order(
        TestOrderInput(
            symbol="BTCUSDT",
            side="SELL",
            type="STOP_LOSS_LIMIT",
            time_in_force="GTC",
            quantity="0.001",
            price="19000.00",
            stop_price="19100.00",
            trailing_delta="100",
        )
    )

    assert "Order validation passed" in result
    assert fake.calls[0][2]["params"]["stopPrice"] == "19100.00"
    assert fake.calls[0][2]["params"]["trailingDelta"] == "100"


async def test_limit_maker_requires_quantity_and_price(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="missing: price"):
        PlaceOrderInput(symbol="BTCUSDT", side="SELL", type="LIMIT_MAKER", quantity="0.001")
    assert fake.calls == []


async def test_iceberg_qty_requires_gtc(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="only accepted with time_in_force=GTC"):
        PlaceOrderInput(
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            time_in_force="IOC",
            quantity="1",
            price="20000",
            iceberg_qty="0.1",
        )
    assert fake.calls == []


async def test_amount_fields_must_be_positive_decimal_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="positive, finite decimal"):
        PlaceOrderInput(symbol="BTCUSDT", side="BUY", type="LIMIT", time_in_force="GTC", quantity="0", price="20000")
    with pytest.raises(ValidationError, match="decimal number sent as a string"):
        PlaceOrderInput(symbol="BTCUSDT", side="BUY", type="LIMIT", time_in_force="GTC", quantity="1", price="cheap")
    assert fake.calls == []


async def test_symbol_is_uppercased_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    assert PlaceOrderInput(**{**LIMIT_ORDER, "symbol": " btcusdt "}).symbol == "BTCUSDT"
    with pytest.raises(ValidationError, match="Binance trading pair"):
        PlaceOrderInput(**{**LIMIT_ORDER, "symbol": "BTC-USDT"})
    assert fake.calls == []


async def test_precision_strings_travel_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trailing zeros and long decimals must survive untouched — filters are exact."""
    fake = _FakeClient(routes={"/api/v3/order/test": {}})
    _patch(monkeypatch, fake)

    await binance_test_order(
        TestOrderInput(
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            time_in_force="GTC",
            quantity="0.00010000",
            price="19999.99000000",
        )
    )

    assert fake.calls[0][2]["params"]["quantity"] == "0.00010000"
    assert fake.calls[0][2]["params"]["price"] == "19999.99000000"


# -- live smoke ------------------------------------------------------------------------


@pytest.mark.live
async def test_live_smoke_read_tools() -> None:
    """Smoke-test the READ tools against the real configured account.

    Read-only by construction: no placement or cancellation runs here. `binance_test_order`
    is deliberately excluded — although it never touches the book, it needs a key with Spot
    Trading permission, and the live key for this suite is read-only. Requires
    BINANCE_API_KEY + secret/PEM (see the conftest's live-marker gating).
    """
    open_orders = await binance_get_open_orders(GetOpenOrdersInput(symbol="BTCUSDT"))
    all_orders = await binance_get_all_orders(GetAllOrdersInput(symbol="BTCUSDT", limit=5))
    for result in (open_orders, all_orders):
        assert isinstance(result, str)
        assert result
        assert not result.startswith("Error")


# -- testnet trading round-trip --------------------------------------------------------


def _quantize(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    """Snap a value to a Binance filter step (tickSize / stepSize)."""
    if step <= 0:
        return value
    return (value / step).quantize(Decimal(1), rounding=rounding) * step


@pytest.mark.trading
async def test_trading_round_trip_on_testnet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Place a far-from-market LIMIT BUY on the spot testnet, then cancel it.

    Skipped by the conftest unless BINANCE_TESTNET=1 **and** BINANCE_TEST_ALLOW_TRADING=1.
    The price is 50% of the last trade, snapped down to the PRICE_FILTER tick, so the order
    rests and never fills; the quantity is the smallest that clears LOT_SIZE and NOTIONAL.
    The order carries a unique client id, and the cancel goes by that id — so the round-trip
    can only ever touch the order this test created.

    BINANCE_ALLOW_TRADING is set here because the client's kill-switch reads that variable,
    while the conftest gate reads BINANCE_TEST_ALLOW_TRADING: opting into the marker is the
    deliberate act, this just wires it through.
    """
    from binance_mcp.client import get_client
    from binance_mcp.config import get_settings

    monkeypatch.setenv("BINANCE_ALLOW_TRADING", "1")
    get_settings.cache_clear()
    monkeypatch.setattr("binance_mcp.client._client", None)
    assert os.environ.get("BINANCE_TESTNET", "").lower() in ("1", "true", "yes")

    symbol = "BTCUSDT"
    client = get_client()
    info = (await client.request("GET", "/api/v3/exchangeInfo", params={"symbol": symbol})).json()
    filters = {f["filterType"]: f for f in info["symbols"][0]["filters"]}
    tick = Decimal(filters["PRICE_FILTER"]["tickSize"])
    step = Decimal(filters["LOT_SIZE"]["stepSize"])
    min_qty = Decimal(filters["LOT_SIZE"]["minQty"])
    min_notional = Decimal(filters.get("NOTIONAL", {}).get("minNotional", "0"))

    last = Decimal((await client.request("GET", "/api/v3/ticker/price", params={"symbol": symbol})).json()["price"])
    price = _quantize(last / 2, tick, ROUND_DOWN)
    quantity = max(min_qty, _quantize(min_notional / price, step, ROUND_UP) if price > 0 else min_qty)
    if quantity * price < min_notional:
        quantity += step
    client_order_id = f"t7-smoke-{int(time.time())}"

    dry_run = await binance_test_order(
        TestOrderInput(
            symbol=symbol,
            side="BUY",
            type="LIMIT",
            time_in_force="GTC",
            quantity=str(quantity),
            price=str(price),
        )
    )
    assert "Order validation passed" in dry_run, dry_run

    placed = await binance_place_order(
        PlaceOrderInput(
            symbol=symbol,
            side="BUY",
            type="LIMIT",
            time_in_force="GTC",
            quantity=str(quantity),
            price=str(price),
            new_client_order_id=client_order_id,
        )
    )
    assert "# Order placed on BTCUSDT" in placed, placed
    assert client_order_id in placed

    cancelled = await binance_cancel_order(CancelOrderInput(symbol=symbol, orig_client_order_id=client_order_id))
    assert "# Order cancelled on BTCUSDT" in cancelled, cancelled
    assert "CANCELED" in cancelled
