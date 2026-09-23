"""Unit tests for the order-list tools (OCO / OTO / OTOCO) against a fake client.

Every tool asserts BOTH the rendered string and the captured
`(method, path, auth, params)` — the exact compact dict, with numbers as strings and
`None` fields dropped. The validation tests additionally assert `fake.calls == []`: a
malformed list must die locally, before anything is signed or sent.
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
from binance_mcp.tools.order_lists import (
    CancelOrderListInput,
    GetAllOrderListsInput,
    GetOpenOrderListsInput,
    GetOrderListInput,
    PlaceOcoOrderInput,
    PlaceOtocoOrderInput,
    PlaceOtoOrderInput,
    binance_cancel_order_list,
    binance_get_all_order_lists,
    binance_get_open_order_lists,
    binance_get_order_list,
    binance_place_oco_order,
    binance_place_oto_order,
    binance_place_otoco_order,
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
    monkeypatch.setattr("binance_mcp.tools.order_lists.get_client", lambda: fake)


# A SELL bracket: take profit above, stop below.
OCO_SELL: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "side": "SELL",
    "quantity": "0.00100000",
    "above_type": "LIMIT_MAKER",
    "above_price": "72000.00",
    "below_type": "STOP_LOSS_LIMIT",
    "below_price": "58000.00",
    "below_stop_price": "58500.00",
    "below_time_in_force": "GTC",
}

OCO_SELL_PARAMS: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "side": "SELL",
    "quantity": "0.00100000",
    "aboveType": "LIMIT_MAKER",
    "abovePrice": "72000.00",
    "belowType": "STOP_LOSS_LIMIT",
    "belowPrice": "58000.00",
    "belowStopPrice": "58500.00",
    "belowTimeInForce": "GTC",
}

OCO_RESPONSE: dict[str, Any] = {
    "orderListId": 1,
    "contingencyType": "OCO",
    "listStatusType": "EXEC_STARTED",
    "listOrderStatus": "EXECUTING",
    "listClientOrderId": "btc-bracket-001",
    "transactionTime": 1758500000000,
    "symbol": "BTCUSDT",
    "orders": [
        {"symbol": "BTCUSDT", "orderId": 10, "clientOrderId": "leg-above"},
        {"symbol": "BTCUSDT", "orderId": 11, "clientOrderId": "leg-below"},
    ],
    "orderReports": [
        {
            "symbol": "BTCUSDT",
            "orderId": 10,
            "orderListId": 1,
            "clientOrderId": "leg-above",
            "price": "72000.00",
            "origQty": "0.00100000",
            "status": "NEW",
            "timeInForce": "GTC",
            "type": "LIMIT_MAKER",
            "side": "SELL",
        },
        {
            "symbol": "BTCUSDT",
            "orderId": 11,
            "orderListId": 1,
            "clientOrderId": "leg-below",
            "price": "58000.00",
            "stopPrice": "58500.00",
            "origQty": "0.00100000",
            "status": "NEW",
            "timeInForce": "GTC",
            "type": "STOP_LOSS_LIMIT",
            "side": "SELL",
        },
    ],
}


# -- binance_place_oco_order ----------------------------------------------------------


async def test_place_oco_sell_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/orderList/oco": OCO_RESPONSE})
    _patch(monkeypatch, fake)

    result = await binance_place_oco_order(PlaceOcoOrderInput(**OCO_SELL, list_client_order_id="btc-bracket-001"))

    assert result.startswith("# OCO order list placed on BTCUSDT")
    assert "**orderListId**: 1" in result
    assert "**contingencyType**: OCO" in result
    assert "**listStatusType**: EXEC_STARTED" in result
    assert "**listOrderStatus**: EXECUTING" in result
    assert "**listClientOrderId**: btc-bracket-001" in result
    assert "### Legs" in result
    assert "| 10 | leg-above | N/A | LIMIT_MAKER | SELL | NEW |" in result
    assert "| 11 | leg-below | N/A | STOP_LOSS_LIMIT | SELL | NEW |" in result
    assert "listOrderStatus **EXECUTING**" in result
    assert fake.calls == [
        (
            "POST",
            "/api/v3/orderList/oco",
            {"auth": "signed", "params": {**OCO_SELL_PARAMS, "listClientOrderId": "btc-bracket-001"}},
        )
    ]


async def test_place_oco_buy_puts_the_stop_leg_above(monkeypatch: pytest.MonkeyPatch) -> None:
    """A BUY OCO is the mirror image: stop above the market, limit below it."""
    fake = _FakeClient(routes={"/api/v3/orderList/oco": {**OCO_RESPONSE, "orderReports": []}})
    _patch(monkeypatch, fake)

    result = await binance_place_oco_order(
        PlaceOcoOrderInput(
            symbol="btcusdt",
            side="BUY",
            quantity="0.001",
            above_type="STOP_LOSS_LIMIT",
            above_price="71000.00",
            above_stop_price="70500.00",
            above_time_in_force="GTC",
            below_type="LIMIT_MAKER",
            below_price="60000.00",
            new_order_resp_type="FULL",
            self_trade_prevention_mode="EXPIRE_MAKER",
        )
    )

    assert "# OCO order list placed on BTCUSDT" in result
    assert fake.calls == [
        (
            "POST",
            "/api/v3/orderList/oco",
            {
                "auth": "signed",
                "params": {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "quantity": "0.001",
                    "aboveType": "STOP_LOSS_LIMIT",
                    "abovePrice": "71000.00",
                    "aboveStopPrice": "70500.00",
                    "aboveTimeInForce": "GTC",
                    "belowType": "LIMIT_MAKER",
                    "belowPrice": "60000.00",
                    "newOrderRespType": "FULL",
                    "selfTradePreventionMode": "EXPIRE_MAKER",
                },
            },
        )
    ]


async def test_place_oco_sends_every_optional_leg_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """Client ids, iceberg, trailing delta and the strategy fields all map to camelCase."""
    fake = _FakeClient(routes={"/api/v3/orderList/oco": OCO_RESPONSE})
    _patch(monkeypatch, fake)

    await binance_place_oco_order(
        PlaceOcoOrderInput(
            symbol="BTCUSDT",
            side="SELL",
            quantity="0.001",
            above_type="LIMIT_MAKER",
            above_price="72000.00",
            above_client_order_id="above-1",
            above_iceberg_qty="0.0001",
            above_strategy_id=7,
            above_strategy_type=1000001,
            below_type="STOP_LOSS_LIMIT",
            below_price="58000.00",
            below_stop_price="58500.00",
            below_trailing_delta="100",
            below_time_in_force="GTC",
            below_client_order_id="below-1",
            below_iceberg_qty="0.0002",
            below_strategy_id=8,
            below_strategy_type=1000002,
        )
    )

    assert fake.calls[0][2]["params"] == {
        "symbol": "BTCUSDT",
        "side": "SELL",
        "quantity": "0.001",
        "aboveType": "LIMIT_MAKER",
        "abovePrice": "72000.00",
        "aboveClientOrderId": "above-1",
        "aboveIcebergQty": "0.0001",
        "aboveStrategyId": 7,
        "aboveStrategyType": 1000001,
        "belowType": "STOP_LOSS_LIMIT",
        "belowPrice": "58000.00",
        "belowStopPrice": "58500.00",
        "belowTrailingDelta": "100",
        "belowTimeInForce": "GTC",
        "belowClientOrderId": "below-1",
        "belowIcebergQty": "0.0002",
        "belowStrategyId": 8,
        "belowStrategyType": 1000002,
    }


async def test_place_oco_trailing_delta_only_leg_skips_the_price_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trailing-delta-only stop has no price to compare; the pair must still be accepted."""
    fake = _FakeClient(routes={"/api/v3/orderList/oco": OCO_RESPONSE})
    _patch(monkeypatch, fake)

    result = await binance_place_oco_order(
        PlaceOcoOrderInput(
            symbol="BTCUSDT",
            side="SELL",
            quantity="0.001",
            above_type="LIMIT_MAKER",
            above_price="72000.00",
            below_type="STOP_LOSS",
            below_trailing_delta="100",
        )
    )

    assert "# OCO order list placed on BTCUSDT" in result
    assert fake.calls[0][2]["params"] == {
        "symbol": "BTCUSDT",
        "side": "SELL",
        "quantity": "0.001",
        "aboveType": "LIMIT_MAKER",
        "abovePrice": "72000.00",
        "belowType": "STOP_LOSS",
        "belowTrailingDelta": "100",
    }


async def test_place_oco_rejects_two_legs_from_the_same_family(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="exactly one take-profit leg"):
        PlaceOcoOrderInput(
            symbol="BTCUSDT",
            side="SELL",
            quantity="0.001",
            above_type="STOP_LOSS",
            above_stop_price="72000.00",
            below_type="STOP_LOSS",
            below_stop_price="58000.00",
        )
    assert fake.calls == []


async def test_place_oco_rejects_sell_with_the_stop_leg_above(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a SELL the take-profit leg must be the `above` one (S2 L3234)."""
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="On a SELL order list the take-profit leg sits ABOVE"):
        PlaceOcoOrderInput(
            symbol="BTCUSDT",
            side="SELL",
            quantity="0.001",
            above_type="STOP_LOSS_LIMIT",
            above_price="71000.00",
            above_stop_price="70500.00",
            above_time_in_force="GTC",
            below_type="LIMIT_MAKER",
            below_price="60000.00",
        )
    assert fake.calls == []


async def test_place_oco_rejects_buy_with_the_limit_leg_above(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="On a BUY order list the stop leg sits ABOVE"):
        PlaceOcoOrderInput(**{**OCO_SELL, "side": "BUY"})
    assert fake.calls == []


async def test_place_oco_rejects_above_price_below_the_below_price(monkeypatch: pytest.MonkeyPatch) -> None:
    """The price-ordering rule, on the only half of it that is checkable locally."""
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="must be strictly greater than"):
        PlaceOcoOrderInput(
            symbol="BTCUSDT",
            side="SELL",
            quantity="0.001",
            above_type="LIMIT_MAKER",
            above_price="50000.00",
            below_type="STOP_LOSS",
            below_stop_price="60000.00",
        )
    assert fake.calls == []


async def test_place_oco_limit_maker_leg_requires_its_price(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match=r"above_type=LIMIT_MAKER requires above_price"):
        PlaceOcoOrderInput(
            symbol="BTCUSDT",
            side="SELL",
            quantity="0.001",
            above_type="LIMIT_MAKER",
            below_type="STOP_LOSS",
            below_stop_price="58000.00",
        )
    assert fake.calls == []


async def test_place_oco_stop_limit_leg_requires_price_tif_and_a_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match=r"below_type=STOP_LOSS_LIMIT requires below_time_in_force"):
        PlaceOcoOrderInput(**{**OCO_SELL, "below_time_in_force": None})
    with pytest.raises(ValidationError, match="requires a trigger"):
        PlaceOcoOrderInput(**{**OCO_SELL, "below_stop_price": None})
    assert fake.calls == []


async def test_place_oco_stop_leg_requires_a_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match=r"below_type=STOP_LOSS requires a trigger"):
        PlaceOcoOrderInput(
            symbol="BTCUSDT",
            side="SELL",
            quantity="0.001",
            above_type="LIMIT_MAKER",
            above_price="72000.00",
            below_type="STOP_LOSS",
        )
    assert fake.calls == []


async def test_place_oco_iceberg_needs_gtc_or_limit_maker(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="below_iceberg_qty .* is only accepted"):
        PlaceOcoOrderInput(**{**OCO_SELL, "below_time_in_force": "IOC", "below_iceberg_qty": "0.0001"})
    assert fake.calls == []


async def test_place_oco_amounts_must_be_positive_decimal_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="positive, finite decimal"):
        PlaceOcoOrderInput(**{**OCO_SELL, "quantity": "0"})
    with pytest.raises(ValidationError, match="decimal number sent as a string"):
        PlaceOcoOrderInput(**{**OCO_SELL, "above_price": "expensive"})
    assert fake.calls == []


async def test_place_oco_symbol_is_uppercased_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    assert PlaceOcoOrderInput(**{**OCO_SELL, "symbol": " btcusdt "}).symbol == "BTCUSDT"
    with pytest.raises(ValidationError, match="Binance trading pair"):
        PlaceOcoOrderInput(**{**OCO_SELL, "symbol": "BTC-USDT"})
    assert fake.calls == []


async def test_place_oco_precision_strings_travel_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trailing zeros and long decimals must survive untouched — filters are exact."""
    fake = _FakeClient(routes={"/api/v3/orderList/oco": OCO_RESPONSE})
    _patch(monkeypatch, fake)

    await binance_place_oco_order(
        PlaceOcoOrderInput(
            **{**OCO_SELL, "quantity": "0.00010000", "above_price": "72000.99000000"},
        )
    )

    assert fake.calls[0][2]["params"]["quantity"] == "0.00010000"
    assert fake.calls[0][2]["params"]["abovePrice"] == "72000.99000000"


async def test_place_oco_dead_legs_do_not_read_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every leg EXPIRED means nothing rests on the book — the heading must say so."""
    dead = {
        **OCO_RESPONSE,
        "orderReports": [
            {**OCO_RESPONSE["orderReports"][0], "status": "EXPIRED"},
            {**OCO_RESPONSE["orderReports"][1], "status": "EXPIRED"},
        ],
    }
    fake = _FakeClient(routes={"/api/v3/orderList/oco": dead})
    _patch(monkeypatch, fake)

    result = await binance_place_oco_order(PlaceOcoOrderInput(**OCO_SELL))

    assert result.startswith("# OCO order list NOT live (EXPIRED) on BTCUSDT")
    assert "order list placed" not in result
    assert "The list is **not working**" in result


async def test_place_oco_rejected_list_status_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    rejected = {**OCO_RESPONSE, "listStatusType": "ALL_DONE", "listOrderStatus": "REJECT", "orderReports": []}
    fake = _FakeClient(routes={"/api/v3/orderList/oco": rejected})
    _patch(monkeypatch, fake)

    result = await binance_place_oco_order(PlaceOcoOrderInput(**OCO_SELL))

    assert result.startswith("# OCO order list NOT live (REJECT) on BTCUSDT")
    assert "Binance returned ids only" in result  # fell back to orders[]


async def test_place_oco_kill_switch_message_is_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client's kill-switch surfaces as `Error: … trading is disabled …`, unaltered."""
    message = (
        "POST /api/v3/orderList/oco would move funds or change account state, but trading is disabled. "
        "Set BINANCE_ALLOW_TRADING=1 to enable order placement/cancellation, transfers, convert and "
        "algo orders. Dry-run validation (POST /api/v3/order/test) works without it."
    )
    fake = _FakeClient(routes={"/api/v3/orderList/oco": TradingDisabledError(message)})
    _patch(monkeypatch, fake)

    result = await binance_place_oco_order(PlaceOcoOrderInput(**OCO_SELL))

    assert result == f"Error: {message}"
    assert "trading is disabled" in result


async def test_place_oco_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/orderList/oco": _status_error(
                "POST",
                "/api/v3/orderList/oco",
                400,
                {"code": -2010, "msg": "Stop price would trigger immediately."},
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_place_oco_order(PlaceOcoOrderInput(**OCO_SELL))

    assert result.startswith("Error (400)")
    assert "(code -2010)" in result


# -- binance_place_oto_order ----------------------------------------------------------


OTO_LIMIT: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "working_type": "LIMIT",
    "working_side": "BUY",
    "working_price": "60000.00",
    "working_quantity": "0.001",
    "working_time_in_force": "GTC",
    "pending_type": "LIMIT",
    "pending_side": "SELL",
    "pending_quantity": "0.001",
    "pending_price": "66000.00",
    "pending_time_in_force": "GTC",
}

OTO_LIMIT_PARAMS: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "workingType": "LIMIT",
    "workingSide": "BUY",
    "workingPrice": "60000.00",
    "workingQuantity": "0.001",
    "workingTimeInForce": "GTC",
    "pendingType": "LIMIT",
    "pendingSide": "SELL",
    "pendingQuantity": "0.001",
    "pendingPrice": "66000.00",
    "pendingTimeInForce": "GTC",
}

OTO_RESPONSE: dict[str, Any] = {
    "orderListId": 5,
    "contingencyType": "OTO",
    "listStatusType": "EXEC_STARTED",
    "listOrderStatus": "EXECUTING",
    "listClientOrderId": "entry-then-target-001",
    "transactionTime": 1758500000000,
    "symbol": "BTCUSDT",
    "orders": [
        {"symbol": "BTCUSDT", "orderId": 20, "clientOrderId": "working-1"},
        {"symbol": "BTCUSDT", "orderId": 21, "clientOrderId": "pending-1"},
    ],
    "orderReports": [
        {
            "symbol": "BTCUSDT",
            "orderId": 20,
            "orderListId": 5,
            "clientOrderId": "working-1",
            "price": "60000.00",
            "origQty": "0.001",
            "status": "NEW",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "BUY",
        },
        {
            "symbol": "BTCUSDT",
            "orderId": 21,
            "orderListId": 5,
            "clientOrderId": "pending-1",
            "price": "66000.00",
            "origQty": "0.001",
            "status": "PENDING_NEW",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "SELL",
        },
    ],
}


async def test_place_oto_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/orderList/oto": OTO_RESPONSE})
    _patch(monkeypatch, fake)

    result = await binance_place_oto_order(
        PlaceOtoOrderInput(**OTO_LIMIT, list_client_order_id="entry-then-target-001")
    )

    assert result.startswith("# OTO order list placed on BTCUSDT")
    assert "**contingencyType**: OTO" in result
    assert "| 20 | working-1 | N/A | LIMIT | BUY | NEW |" in result
    assert "| 21 | pending-1 | N/A | LIMIT | SELL | PENDING_NEW |" in result
    assert fake.calls == [
        (
            "POST",
            "/api/v3/orderList/oto",
            {"auth": "signed", "params": {**OTO_LIMIT_PARAMS, "listClientOrderId": "entry-then-target-001"}},
        )
    ]


async def test_place_oto_limit_maker_working_leg_needs_no_time_in_force(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/orderList/oto": OTO_RESPONSE})
    _patch(monkeypatch, fake)

    result = await binance_place_oto_order(
        PlaceOtoOrderInput(
            symbol="BTCUSDT",
            working_type="LIMIT_MAKER",
            working_side="BUY",
            working_price="60000.00",
            working_quantity="0.001",
            working_iceberg_qty="0.0001",
            working_client_order_id="working-1",
            pending_type="STOP_LOSS",
            pending_side="SELL",
            pending_quantity="0.001",
            pending_stop_price="57000.00",
            pending_client_order_id="pending-1",
            pending_strategy_id=3,
            pending_strategy_type=1000003,
        )
    )

    assert "# OTO order list placed on BTCUSDT" in result
    assert fake.calls == [
        (
            "POST",
            "/api/v3/orderList/oto",
            {
                "auth": "signed",
                "params": {
                    "symbol": "BTCUSDT",
                    "workingType": "LIMIT_MAKER",
                    "workingSide": "BUY",
                    "workingPrice": "60000.00",
                    "workingQuantity": "0.001",
                    "workingIcebergQty": "0.0001",
                    "workingClientOrderId": "working-1",
                    "pendingType": "STOP_LOSS",
                    "pendingSide": "SELL",
                    "pendingQuantity": "0.001",
                    "pendingStopPrice": "57000.00",
                    "pendingClientOrderId": "pending-1",
                    "pendingStrategyId": 3,
                    "pendingStrategyType": 1000003,
                },
            },
        )
    ]


async def test_place_oto_market_pending_leg_needs_no_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    """A MARKET pending leg only needs its quantity — never quoteOrderQty, which lists reject."""
    fake = _FakeClient(routes={"/api/v3/orderList/oto": OTO_RESPONSE})
    _patch(monkeypatch, fake)

    await binance_place_oto_order(
        PlaceOtoOrderInput(
            symbol="BTCUSDT",
            working_type="LIMIT",
            working_side="SELL",
            working_price="72000.00",
            working_quantity="0.001",
            working_time_in_force="GTC",
            pending_type="MARKET",
            pending_side="BUY",
            pending_quantity="0.001",
        )
    )

    assert fake.calls[0][2]["params"] == {
        "symbol": "BTCUSDT",
        "workingType": "LIMIT",
        "workingSide": "SELL",
        "workingPrice": "72000.00",
        "workingQuantity": "0.001",
        "workingTimeInForce": "GTC",
        "pendingType": "MARKET",
        "pendingSide": "BUY",
        "pendingQuantity": "0.001",
    }


async def test_place_oto_working_limit_requires_time_in_force(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="working_type=LIMIT requires working_time_in_force"):
        PlaceOtoOrderInput(**{**OTO_LIMIT, "working_time_in_force": None})
    assert fake.calls == []


async def test_place_oto_pending_limit_requires_price_and_time_in_force(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match=r"pending_type=LIMIT requires pending_price"):
        PlaceOtoOrderInput(**{**OTO_LIMIT, "pending_price": None})
    with pytest.raises(ValidationError, match=r"pending_type=LIMIT requires pending_time_in_force"):
        PlaceOtoOrderInput(**{**OTO_LIMIT, "pending_time_in_force": None})
    assert fake.calls == []


async def test_place_oto_pending_stop_family_requires_a_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    """`pendingStopPrice` and/or `pendingTrailingDelta` — either one satisfies it (S2 L3423)."""
    fake = _FakeClient(routes={"/api/v3/orderList/oto": OTO_RESPONSE})
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match=r"pending_type=TAKE_PROFIT requires a trigger"):
        PlaceOtoOrderInput(
            symbol="BTCUSDT",
            working_type="LIMIT",
            working_side="BUY",
            working_price="60000.00",
            working_quantity="0.001",
            working_time_in_force="GTC",
            pending_type="TAKE_PROFIT",
            pending_side="SELL",
            pending_quantity="0.001",
        )

    result = await binance_place_oto_order(
        PlaceOtoOrderInput(
            symbol="BTCUSDT",
            working_type="LIMIT",
            working_side="BUY",
            working_price="60000.00",
            working_quantity="0.001",
            working_time_in_force="GTC",
            pending_type="TAKE_PROFIT",
            pending_side="SELL",
            pending_quantity="0.001",
            pending_trailing_delta="100",
        )
    )

    assert "# OTO order list placed on BTCUSDT" in result
    assert fake.calls[0][2]["params"]["pendingTrailingDelta"] == "100"
    assert "pendingStopPrice" not in fake.calls[0][2]["params"]


async def test_place_oto_pending_stop_limit_requires_price_tif_and_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match=r"pending_type=STOP_LOSS_LIMIT requires pending_price"):
        PlaceOtoOrderInput(
            symbol="BTCUSDT",
            working_type="LIMIT_MAKER",
            working_side="BUY",
            working_price="60000.00",
            working_quantity="0.001",
            pending_type="STOP_LOSS_LIMIT",
            pending_side="SELL",
            pending_quantity="0.001",
            pending_stop_price="57500.00",
            pending_time_in_force="GTC",
        )
    assert fake.calls == []


async def test_place_oto_pending_limit_maker_requires_price(monkeypatch: pytest.MonkeyPatch) -> None:
    """S2's OTO table omits LIMIT_MAKER; its OTOCO twin (L3574) requires the price."""
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match=r"pending_type=LIMIT_MAKER requires pending_price"):
        PlaceOtoOrderInput(
            symbol="BTCUSDT",
            working_type="LIMIT_MAKER",
            working_side="BUY",
            working_price="60000.00",
            working_quantity="0.001",
            pending_type="LIMIT_MAKER",
            pending_side="SELL",
            pending_quantity="0.001",
        )
    assert fake.calls == []


async def test_place_oto_kill_switch_message_is_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    message = (
        "POST /api/v3/orderList/oto would move funds or change account state, but trading is disabled. "
        "Set BINANCE_ALLOW_TRADING=1 to enable order placement/cancellation, transfers, convert and "
        "algo orders. Dry-run validation (POST /api/v3/order/test) works without it."
    )
    fake = _FakeClient(routes={"/api/v3/orderList/oto": TradingDisabledError(message)})
    _patch(monkeypatch, fake)

    result = await binance_place_oto_order(PlaceOtoOrderInput(**OTO_LIMIT))

    assert result == f"Error: {message}"
    assert "trading is disabled" in result


async def test_place_oto_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/orderList/oto": _status_error(
                "POST",
                "/api/v3/orderList/oto",
                400,
                {"code": -1013, "msg": "Filter failure: NOTIONAL"},
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_place_oto_order(PlaceOtoOrderInput(**OTO_LIMIT))

    assert result.startswith("Error (400)")
    assert "NOTIONAL" in result


# -- binance_place_otoco_order --------------------------------------------------------


OTOCO: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "working_type": "LIMIT",
    "working_side": "BUY",
    "working_price": "60000.00",
    "working_quantity": "0.001",
    "working_time_in_force": "GTC",
    "pending_side": "SELL",
    "pending_quantity": "0.001",
    "pending_above_type": "LIMIT_MAKER",
    "pending_above_price": "66000.00",
    "pending_below_type": "STOP_LOSS_LIMIT",
    "pending_below_price": "57000.00",
    "pending_below_stop_price": "57500.00",
    "pending_below_time_in_force": "GTC",
}

OTOCO_PARAMS: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "workingType": "LIMIT",
    "workingSide": "BUY",
    "workingPrice": "60000.00",
    "workingQuantity": "0.001",
    "workingTimeInForce": "GTC",
    "pendingSide": "SELL",
    "pendingQuantity": "0.001",
    "pendingAboveType": "LIMIT_MAKER",
    "pendingAbovePrice": "66000.00",
    "pendingBelowType": "STOP_LOSS_LIMIT",
    "pendingBelowPrice": "57000.00",
    "pendingBelowStopPrice": "57500.00",
    "pendingBelowTimeInForce": "GTC",
}

OTOCO_RESPONSE: dict[str, Any] = {
    "orderListId": 9,
    "contingencyType": "OTO",
    "listStatusType": "EXEC_STARTED",
    "listOrderStatus": "EXECUTING",
    "listClientOrderId": "full-bracket-001",
    "transactionTime": 1758500000000,
    "symbol": "BTCUSDT",
    "orders": [
        {"symbol": "BTCUSDT", "orderId": 30, "clientOrderId": "w"},
        {"symbol": "BTCUSDT", "orderId": 31, "clientOrderId": "pa"},
        {"symbol": "BTCUSDT", "orderId": 32, "clientOrderId": "pb"},
    ],
    "orderReports": [
        {"symbol": "BTCUSDT", "orderId": 30, "clientOrderId": "w", "status": "NEW", "type": "LIMIT", "side": "BUY"},
        {
            "symbol": "BTCUSDT",
            "orderId": 31,
            "clientOrderId": "pa",
            "status": "PENDING_NEW",
            "type": "LIMIT_MAKER",
            "side": "SELL",
        },
        {
            "symbol": "BTCUSDT",
            "orderId": 32,
            "clientOrderId": "pb",
            "status": "PENDING_NEW",
            "type": "STOP_LOSS_LIMIT",
            "side": "SELL",
        },
    ],
}


async def test_place_otoco_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/orderList/otoco": OTOCO_RESPONSE})
    _patch(monkeypatch, fake)

    result = await binance_place_otoco_order(PlaceOtocoOrderInput(**OTOCO, list_client_order_id="full-bracket-001"))

    assert result.startswith("# OTOCO order list placed on BTCUSDT")
    assert "| 30 | w | N/A | LIMIT | BUY | NEW |" in result
    assert "| 31 | pa | N/A | LIMIT_MAKER | SELL | PENDING_NEW |" in result
    assert "| 32 | pb | N/A | STOP_LOSS_LIMIT | SELL | PENDING_NEW |" in result
    assert fake.calls == [
        (
            "POST",
            "/api/v3/orderList/otoco",
            {"auth": "signed", "params": {**OTOCO_PARAMS, "listClientOrderId": "full-bracket-001"}},
        )
    ]


async def test_place_otoco_without_a_below_leg_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binance marks pendingBelowType optional (S2 L3550); nothing below is then sent."""
    fake = _FakeClient(routes={"/api/v3/orderList/otoco": OTOCO_RESPONSE})
    _patch(monkeypatch, fake)

    await binance_place_otoco_order(
        PlaceOtocoOrderInput(
            **{
                **OTOCO,
                "pending_below_type": None,
                "pending_below_price": None,
                "pending_below_stop_price": None,
                "pending_below_time_in_force": None,
            }
        )
    )

    params = fake.calls[0][2]["params"]
    assert params["pendingAboveType"] == "LIMIT_MAKER"
    assert not [key for key in params if key.startswith("pendingBelow")]


async def test_place_otoco_rejects_pending_pair_price_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="pending_above_price .* must be strictly greater than"):
        PlaceOtocoOrderInput(**{**OTOCO, "pending_above_price": "50000.00"})
    assert fake.calls == []


async def test_place_otoco_rejects_pending_pair_from_one_family(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match="exactly one take-profit leg"):
        PlaceOtocoOrderInput(
            **{
                **OTOCO,
                "pending_above_type": "TAKE_PROFIT",
                "pending_above_price": None,
                "pending_above_stop_price": "66000.00",
                "pending_below_type": "TAKE_PROFIT",
                "pending_below_price": None,
                "pending_below_stop_price": "57500.00",
                "pending_below_time_in_force": None,
            }
        )
    assert fake.calls == []


async def test_place_otoco_pending_above_limit_maker_requires_price(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError, match=r"pending_above_type=LIMIT_MAKER requires pending_above_price"):
        PlaceOtocoOrderInput(**{**OTOCO, "pending_above_price": None})
    assert fake.calls == []


async def test_place_otoco_kill_switch_message_is_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    message = (
        "POST /api/v3/orderList/otoco would move funds or change account state, but trading is disabled. "
        "Set BINANCE_ALLOW_TRADING=1 to enable order placement/cancellation, transfers, convert and "
        "algo orders. Dry-run validation (POST /api/v3/order/test) works without it."
    )
    fake = _FakeClient(routes={"/api/v3/orderList/otoco": TradingDisabledError(message)})
    _patch(monkeypatch, fake)

    result = await binance_place_otoco_order(PlaceOtocoOrderInput(**OTOCO))

    assert result == f"Error: {message}"
    assert "trading is disabled" in result


async def test_place_otoco_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/orderList/otoco": _status_error(
                "POST",
                "/api/v3/orderList/otoco",
                400,
                {"code": -2021, "msg": "Order would immediately match and take."},
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_place_otoco_order(PlaceOtocoOrderInput(**OTOCO))

    assert result.startswith("Error (400)")
    assert "(code -2021)" in result


# -- binance_get_order_list -----------------------------------------------------------


QUERY_RESPONSE: dict[str, Any] = {
    "orderListId": 27,
    "contingencyType": "OCO",
    "listStatusType": "EXEC_STARTED",
    "listOrderStatus": "EXECUTING",
    "listClientOrderId": "btc-bracket-001",
    "transactionTime": 1758500000000,
    "symbol": "BTCUSDT",
    "orders": [
        {"symbol": "BTCUSDT", "orderId": 4, "clientOrderId": "leg-a"},
        {"symbol": "BTCUSDT", "orderId": 5, "clientOrderId": "leg-b"},
    ],
}


async def test_get_order_list_by_order_list_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/orderList": QUERY_RESPONSE})
    _patch(monkeypatch, fake)

    result = await binance_get_order_list(GetOrderListInput(order_list_id=27))

    assert "# Order list 27 on BTCUSDT" in result
    assert "**listOrderStatus**: EXECUTING" in result
    assert "| 4 | leg-a | N/A | N/A | N/A | N/A |" in result
    assert "Binance returned ids only" in result
    assert fake.calls == [("GET", "/api/v3/orderList", {"auth": "signed", "params": {"orderListId": 27}})]


async def test_get_order_list_by_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GET spells the client id `origClientOrderId`, unlike the DELETE."""
    fake = _FakeClient(routes={"/api/v3/orderList": QUERY_RESPONSE})
    _patch(monkeypatch, fake)

    result = await binance_get_order_list(GetOrderListInput(orig_client_order_id="btc-bracket-001"))

    assert "btc-bracket-001" in result
    assert fake.calls == [
        ("GET", "/api/v3/orderList", {"auth": "signed", "params": {"origClientOrderId": "btc-bracket-001"}})
    ]


async def test_get_order_list_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/orderList": QUERY_RESPONSE})
    _patch(monkeypatch, fake)

    result = await binance_get_order_list(GetOrderListInput(order_list_id=27, response_format=ResponseFormat.JSON))

    assert '"orderListId": 27' in result


async def test_get_order_list_requires_exactly_one_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    both = await binance_get_order_list(GetOrderListInput(order_list_id=27, orig_client_order_id="x"))
    neither = await binance_get_order_list(GetOrderListInput())

    assert both.startswith("Error: Pass exactly one of order_list_id")
    assert neither.startswith("Error: Pass exactly one of order_list_id")
    assert fake.calls == []


async def test_get_order_list_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/orderList": _status_error(
                "GET", "/api/v3/orderList", 400, {"code": -2013, "msg": "Order does not exist."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_order_list(GetOrderListInput(order_list_id=27))

    assert result.startswith("Error (400)")
    assert "(code -2013)" in result


# -- binance_cancel_order_list --------------------------------------------------------


CANCELLED_LIST: dict[str, Any] = {
    "orderListId": 27,
    "contingencyType": "OCO",
    "listStatusType": "ALL_DONE",
    "listOrderStatus": "ALL_DONE",
    "listClientOrderId": "btc-bracket-001",
    "transactionTime": 1758500000000,
    "symbol": "BTCUSDT",
    "orders": [
        {"symbol": "BTCUSDT", "orderId": 4, "clientOrderId": "leg-a"},
        {"symbol": "BTCUSDT", "orderId": 5, "clientOrderId": "leg-b"},
    ],
    "orderReports": [
        {
            "symbol": "BTCUSDT",
            "origClientOrderId": "leg-a",
            "orderId": 4,
            "orderListId": 27,
            "clientOrderId": "cancel-1",
            "price": "72000.00",
            "origQty": "0.001",
            "status": "CANCELED",
            "type": "LIMIT_MAKER",
            "side": "SELL",
        },
        {
            "symbol": "BTCUSDT",
            "origClientOrderId": "leg-b",
            "orderId": 5,
            "orderListId": 27,
            "clientOrderId": "cancel-1",
            "price": "58000.00",
            "stopPrice": "58500.00",
            "origQty": "0.001",
            "status": "CANCELED",
            "type": "STOP_LOSS_LIMIT",
            "side": "SELL",
        },
    ],
}


async def test_cancel_order_list_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/orderList": CANCELLED_LIST})
    _patch(monkeypatch, fake)

    result = await binance_cancel_order_list(CancelOrderListInput(symbol="BTCUSDT", order_list_id=27))

    assert result.startswith("# Order list cancelled on BTCUSDT")
    assert "**listOrderStatus**: ALL_DONE" in result
    assert "| 4 | cancel-1 | leg-a | LIMIT_MAKER | SELL | CANCELED |" in result
    assert "| 5 | cancel-1 | leg-b | STOP_LOSS_LIMIT | SELL | CANCELED |" in result
    assert fake.calls == [
        ("DELETE", "/api/v3/orderList", {"auth": "signed", "params": {"symbol": "BTCUSDT", "orderListId": 27}})
    ]


async def test_cancel_order_list_by_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DELETE spells the client id `listClientOrderId`, unlike the GET."""
    fake = _FakeClient(routes={"/api/v3/orderList": CANCELLED_LIST})
    _patch(monkeypatch, fake)

    result = await binance_cancel_order_list(
        CancelOrderListInput(symbol="btcusdt", list_client_order_id="btc-bracket-001", new_client_order_id="cancel-1")
    )

    assert "CANCELED" in result
    assert fake.calls == [
        (
            "DELETE",
            "/api/v3/orderList",
            {
                "auth": "signed",
                "params": {
                    "symbol": "BTCUSDT",
                    "listClientOrderId": "btc-bracket-001",
                    "newClientOrderId": "cancel-1",
                },
            },
        )
    ]


async def test_cancel_order_list_requires_exactly_one_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    both = await binance_cancel_order_list(
        CancelOrderListInput(symbol="BTCUSDT", order_list_id=27, list_client_order_id="x")
    )
    neither = await binance_cancel_order_list(CancelOrderListInput(symbol="BTCUSDT"))

    assert both.startswith("Error: Pass exactly one of order_list_id")
    assert neither.startswith("Error: Pass exactly one of order_list_id")
    assert fake.calls == []


async def test_cancel_order_list_kill_switch_message_is_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    message = (
        "DELETE /api/v3/orderList would move funds or change account state, but trading is disabled. "
        "Set BINANCE_ALLOW_TRADING=1 to enable order placement/cancellation, transfers, convert and "
        "algo orders. Dry-run validation (POST /api/v3/order/test) works without it."
    )
    fake = _FakeClient(routes={"/api/v3/orderList": TradingDisabledError(message)})
    _patch(monkeypatch, fake)

    result = await binance_cancel_order_list(CancelOrderListInput(symbol="BTCUSDT", order_list_id=27))

    assert result == f"Error: {message}"
    assert "trading is disabled" in result


async def test_cancel_order_list_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/orderList": _status_error(
                "DELETE", "/api/v3/orderList", 400, {"code": -2011, "msg": "Unknown order sent."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_cancel_order_list(CancelOrderListInput(symbol="BTCUSDT", order_list_id=27))

    assert result.startswith("Error (400)")
    assert "(code -2011)" in result
    assert "already filled or cancelled" in result


# -- binance_get_all_order_lists ------------------------------------------------------


LIST_ROW: dict[str, Any] = {
    "orderListId": 29,
    "contingencyType": "OCO",
    "listStatusType": "EXEC_STARTED",
    "listOrderStatus": "EXECUTING",
    "listClientOrderId": "amEEAXryFzFwYF1FeRpUoZ",
    "transactionTime": 1758500000000,
    "symbol": "BTCUSDT",
    "orders": [
        {"symbol": "BTCUSDT", "orderId": 4, "clientOrderId": "a"},
        {"symbol": "BTCUSDT", "orderId": 5, "clientOrderId": "b"},
    ],
}


async def test_get_all_order_lists_with_iso_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/allOrderList": [LIST_ROW]})
    _patch(monkeypatch, fake)

    result = await binance_get_all_order_lists(
        GetAllOrderListsInput(start_time="2026-09-22T00:00:00Z", end_time="2026-09-22T12:00:00Z", limit=1000)
    )

    assert "# Order lists (all symbols)" in result
    assert "| BTCUSDT | 29 | OCO | EXEC_STARTED | EXECUTING | amEEAXryFzFwYF1FeRpUoZ | 2 |" in result
    assert fake.calls == [
        (
            "GET",
            "/api/v3/allOrderList",
            {"auth": "signed", "params": {"limit": 1000, "startTime": 1790035200000, "endTime": 1790078400000}},
        )
    ]


async def test_get_all_order_lists_with_from_id_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/allOrderList": []})
    _patch(monkeypatch, fake)

    result = await binance_get_all_order_lists(GetAllOrderListsInput(from_id=29))

    assert "_No order lists._" in result
    assert fake.calls == [("GET", "/api/v3/allOrderList", {"auth": "signed", "params": {"limit": 500, "fromId": 29}})]


async def test_get_all_order_lists_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/allOrderList": [LIST_ROW]})
    _patch(monkeypatch, fake)

    result = await binance_get_all_order_lists(GetAllOrderListsInput(response_format=ResponseFormat.JSON))

    assert '"orderListId": 29' in result


async def test_get_all_order_lists_rejects_from_id_with_a_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    result = await binance_get_all_order_lists(GetAllOrderListsInput(from_id=29, start_time="2026-09-22T00:00:00Z"))

    assert result.startswith("Error: from_id cannot be combined with start_time/end_time")
    assert fake.calls == []


async def test_get_all_order_lists_rejects_window_over_24h(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    result = await binance_get_all_order_lists(
        GetAllOrderListsInput(start_time="2026-09-20T00:00:00Z", end_time="2026-09-22T00:00:00Z")
    )

    assert result.startswith("Error: the start_time/end_time window spans ~48.0 hours")
    assert "caps it at 24 hours" in result
    assert fake.calls == []


async def test_get_all_order_lists_rejects_reversed_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    result = await binance_get_all_order_lists(GetAllOrderListsInput(start_time=1789128000000, end_time=1789084800000))

    assert result == "Error: end_time must be after start_time."
    assert fake.calls == []


async def test_get_all_order_lists_rejects_malformed_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)

    result = await binance_get_all_order_lists(GetAllOrderListsInput(start_time="yesterday"))

    assert result.startswith("Error: start_time must be an epoch-ms integer")
    assert fake.calls == []


async def test_get_all_order_lists_display_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{**LIST_ROW, "orderListId": index} for index in range(60)]
    fake = _FakeClient(routes={"/api/v3/allOrderList": rows})
    _patch(monkeypatch, fake)

    result = await binance_get_all_order_lists(GetAllOrderListsInput())

    assert "Showing **50** of **60** order list(s)." in result
    assert "_[10 more order list(s) not shown" in result
    assert "| 49 | OCO |" in result
    assert "| 50 | OCO |" not in result


async def test_get_all_order_lists_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/allOrderList": _status_error(
                "GET",
                "/api/v3/allOrderList",
                400,
                {"code": -1127, "msg": "More than 24 hours between startTime and endTime."},
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_all_order_lists(GetAllOrderListsInput())

    assert result.startswith("Error (400)")
    assert "(code -1127)" in result


# -- binance_get_open_order_lists -----------------------------------------------------


async def test_get_open_order_lists_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/openOrderList": [LIST_ROW]})
    _patch(monkeypatch, fake)

    result = await binance_get_open_order_lists(GetOpenOrderListsInput())

    assert "# Open order lists (all symbols)" in result
    assert "| BTCUSDT | 29 | OCO |" in result
    assert fake.calls == [("GET", "/api/v3/openOrderList", {"auth": "signed", "params": {}})]


async def test_get_open_order_lists_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/api/v3/openOrderList": []})
    _patch(monkeypatch, fake)

    result = await binance_get_open_order_lists(GetOpenOrderListsInput(response_format=ResponseFormat.JSON))

    assert result == "[]"
    assert fake.calls == [("GET", "/api/v3/openOrderList", {"auth": "signed", "params": {}})]


async def test_get_open_order_lists_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/api/v3/openOrderList": _status_error(
                "GET", "/api/v3/openOrderList", 401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_open_order_lists(GetOpenOrderListsInput())

    assert result.startswith("Error (401)")
    assert "allowlist" in result


# -- live smoke ------------------------------------------------------------------------


@pytest.mark.live
async def test_live_smoke_read_tools() -> None:
    """Smoke-test the READ tools against the real configured account.

    Read-only by construction: no placement or cancellation runs here. Both endpoints are
    account-wide, so no symbol is needed and an empty result is a valid answer. Requires
    BINANCE_API_KEY + secret/PEM (see the conftest's live-marker gating).
    """
    open_lists = await binance_get_open_order_lists(GetOpenOrderListsInput())
    all_lists = await binance_get_all_order_lists(GetAllOrderListsInput(limit=5))
    for result in (open_lists, all_lists):
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
    """Place a far-from-market OTO on the spot testnet, then cancel the list.

    Skipped by the conftest unless BINANCE_TESTNET=1 **and** BINANCE_TEST_ALLOW_TRADING=1.

    An OTO with a LIMIT **BUY** working leg is used deliberately: a SELL list would need a
    base-asset balance this test cannot assume, whereas a buy at 50% of the last price
    rests, never fills, and therefore never arms its pending leg. The quantity is the
    smallest that clears LOT_SIZE and NOTIONAL. The list carries a unique
    listClientOrderId and the cancel goes by that id — so the round-trip can only ever
    touch the list this test created.

    BINANCE_ALLOW_TRADING is set here because the client's kill-switch reads that variable,
    while the conftest gate reads BINANCE_TEST_ALLOW_TRADING: opting into the marker is the
    deliberate act, this just wires it through.

    The env-var check is NOT sufficient on its own: `Settings.base_url` only falls back to
    the testnet when BINANCE_API_URL is still the default, so BINANCE_TESTNET=1 plus a
    custom BINANCE_API_URL routes this straight to mainnet — with real money. The effective
    base URL is therefore asserted against TESTNET_API_URL before the first request.
    """
    from binance_mcp.client import get_client
    from binance_mcp.config import TESTNET_API_URL, get_settings

    monkeypatch.setenv("BINANCE_ALLOW_TRADING", "1")
    get_settings.cache_clear()
    monkeypatch.setattr("binance_mcp.client._client", None)
    assert os.environ.get("BINANCE_TESTNET", "").lower() in ("1", "true", "yes")
    assert get_settings().base_url == TESTNET_API_URL, (
        "refusing to place an order list: BINANCE_TESTNET=1 but the effective base URL is "
        f"{get_settings().base_url!r}, not the testnet — a custom BINANCE_API_URL overrides the flag."
    )

    symbol = "BTCUSDT"
    client = get_client()
    info = (await client.request("GET", "/api/v3/exchangeInfo", params={"symbol": symbol})).json()
    filters = {f["filterType"]: f for f in info["symbols"][0]["filters"]}
    tick = Decimal(filters["PRICE_FILTER"]["tickSize"])
    step = Decimal(filters["LOT_SIZE"]["stepSize"])
    min_qty = Decimal(filters["LOT_SIZE"]["minQty"])
    min_notional = Decimal(filters.get("NOTIONAL", {}).get("minNotional", "0"))

    last = Decimal((await client.request("GET", "/api/v3/ticker/price", params={"symbol": symbol})).json()["price"])
    working_price = _quantize(last / 2, tick, ROUND_DOWN)
    pending_price = _quantize(last * 2, tick, ROUND_UP)
    quantity = max(min_qty, _quantize(min_notional / working_price, step, ROUND_UP) if working_price > 0 else min_qty)
    if quantity * working_price < min_notional:
        quantity += step
    list_client_order_id = f"t11-smoke-{int(time.time())}"

    placed = await binance_place_oto_order(
        PlaceOtoOrderInput(
            symbol=symbol,
            working_type="LIMIT",
            working_side="BUY",
            working_price=str(working_price),
            working_quantity=str(quantity),
            working_time_in_force="GTC",
            pending_type="LIMIT",
            pending_side="SELL",
            pending_quantity=str(quantity),
            pending_price=str(pending_price),
            pending_time_in_force="GTC",
            list_client_order_id=list_client_order_id,
        )
    )
    assert "# OTO order list placed on BTCUSDT" in placed, placed
    assert list_client_order_id in placed

    cancelled = await binance_cancel_order_list(
        CancelOrderListInput(symbol=symbol, list_client_order_id=list_client_order_id)
    )
    assert "# Order list cancelled on BTCUSDT" in cancelled, cancelled
    assert "CANCELED" in cancelled
