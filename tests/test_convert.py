"""Unit tests for tools/convert.py against a fake client.

Every tool asserts the captured `(method, path, auth, params)` as well as the rendered
string, because a convert tool that signs the wrong path or drops a mandatory parameter
still "returns a string". The walk tests script `tradeFlow` per (startTime, endTime) so
the window boundaries themselves are asserted, not just the row count.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from binance_mcp.client import TradingDisabledError
from binance_mcp.tools.convert import (
    CONVERT_ACCEPT_QUOTE_PATH,
    CONVERT_ASSET_INFO_PATH,
    CONVERT_EXCHANGE_INFO_PATH,
    CONVERT_GET_QUOTE_PATH,
    CONVERT_LIMIT_CANCEL_PATH,
    CONVERT_LIMIT_OPEN_ORDERS_PATH,
    CONVERT_LIMIT_PLACE_PATH,
    CONVERT_ORDER_STATUS_PATH,
    CONVERT_TRADE_FLOW_PATH,
    MAX_CONVERT_WINDOW_MS,
    WALK_WINDOW_MS,
    AcceptConvertQuoteInput,
    CancelConvertLimitOrderInput,
    ConvertAssetInfoInput,
    ConvertHistoryInput,
    ConvertOpenLimitOrdersInput,
    ConvertOrderStatusInput,
    ConvertPairsInput,
    ConvertQuoteInput,
    PlaceConvertLimitOrderInput,
    _dedupe_by_order_id,
    _envelope_error,
    _normalize_asset,
    _oldest_create_time,
    _to_ms,
    binance_accept_convert_quote,
    binance_cancel_convert_limit_order,
    binance_get_convert_asset_info,
    binance_get_convert_history,
    binance_get_convert_open_limit_orders,
    binance_get_convert_order_status,
    binance_get_convert_pairs,
    binance_get_convert_quote,
    binance_place_convert_limit_order,
)

UNTIL = 1_800_000_000_000
NOW = 1_800_000_000_000


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Records every call; routes by path, and `tradeFlow` by (startTime, endTime).

    A routed value that is an `Exception` is raised instead of returned, so an error can
    be scripted for one specific window of a walk.
    """

    def __init__(
        self,
        routes: dict[str, Any] | None = None,
        trade_flow: dict[tuple[int, int], Any] | None = None,
    ) -> None:
        self._routes = routes or {}
        self._trade_flow = trade_flow
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, path, kwargs))
        if self._trade_flow is not None and path == CONVERT_TRADE_FLOW_PATH:
            params = kwargs.get("params") or {}
            outcome = self._trade_flow.get((params.get("startTime"), params.get("endTime")))
            if outcome is None:
                outcome = {"list": [], "moreData": False}
            if isinstance(outcome, Exception):
                raise outcome
            return _FakeResponse(outcome)
        outcome = self._routes.get(path)
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(outcome)

    @property
    def params(self) -> dict[str, Any]:
        """Params of the last recorded call."""
        return dict(self.calls[-1][2].get("params") or {})

    @property
    def windows(self) -> list[tuple[int, int]]:
        """(startTime, endTime) of every tradeFlow call, in order."""
        return [
            (call[2]["params"]["startTime"], call[2]["params"]["endTime"])
            for call in self.calls
            if call[1] == CONVERT_TRADE_FLOW_PATH
        ]


def _patch(monkeypatch: pytest.MonkeyPatch, fake: _FakeClient) -> None:
    monkeypatch.setattr("binance_mcp.tools.convert.get_client", lambda: fake)


def _pin_now(monkeypatch: pytest.MonkeyPatch, now_ms: int = NOW) -> None:
    monkeypatch.setattr("binance_mcp.tools.convert._now_ms", lambda: now_ms)


def _status_error(status: int = 400, body: dict[str, Any] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.binance.com/sapi/v1/convert/tradeFlow")
    response = httpx.Response(status, json=body or {"code": -1121, "msg": "Invalid symbol."}, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def _trade(order_id: int, *, created: int, from_amount: str = "0.01", to_amount: str = "600") -> dict[str, Any]:
    return {
        "quoteId": f"q{order_id}",
        "orderId": order_id,
        "orderStatus": "SUCCESS",
        "fromAsset": "BTC",
        "fromAmount": from_amount,
        "toAsset": "USDT",
        "toAmount": to_amount,
        "ratio": "60000",
        "inverseRatio": "0.0000166",
        "createTime": created,
    }


def _page(rows: list[dict[str, Any]], *, more: bool = False) -> dict[str, Any]:
    return {"list": rows, "startTime": 0, "endTime": 0, "limit": 100, "moreData": more}


# -- pure helpers ---------------------------------------------------------------------


def test_to_ms_passes_through_ints_and_none() -> None:
    assert _to_ms(1_700_000_000_000) == 1_700_000_000_000
    assert _to_ms(None) is None


def test_to_ms_parses_digit_strings_and_iso() -> None:
    assert _to_ms("1700000000000") == 1_700_000_000_000
    assert _to_ms("2023-11-14T22:13:20Z") == int(datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC).timestamp() * 1000)


def test_to_ms_rejects_short_digit_strings_and_garbage() -> None:
    with pytest.raises(ValueError):
        _to_ms("12345")
    with pytest.raises(ValueError):
        _to_ms("not-a-date")


def test_normalize_asset_uppercases_and_allows_one_letter_assets() -> None:
    assert _normalize_asset("btc") == "BTC"
    assert _normalize_asset(" w ") == "W"  # real one-letter asset: {1,20}, never {2,20}
    with pytest.raises(ValueError):
        _normalize_asset("BTC/USDT")
    with pytest.raises(ValueError):
        _normalize_asset("")
    # `str.isalnum()` would accept these; Binance would not.
    with pytest.raises(ValueError):
        _normalize_asset("BT\u0421")  # Cyrillic ES
    with pytest.raises(ValueError):
        _normalize_asset("\u00b2")  # superscript two


def test_amount_fields_reject_exponent_sign_and_leading_dot() -> None:
    for bad in ("1E+5", "+5", ".5", "1e5", "NaN", "Infinity", "-1"):
        with pytest.raises(ValidationError) as exc_info:
            ConvertQuoteInput(from_asset="BTC", to_asset="USDT", from_amount=bad)
        assert "from_amount" in str(exc_info.value)
    # The plain shapes still pass through verbatim.
    assert ConvertQuoteInput(from_asset="BTC", to_asset="USDT", from_amount="0.010").from_amount == "0.010"
    assert ConvertQuoteInput(from_asset="BTC", to_asset="USDT", from_amount="500").from_amount == "500"


async def test_sort_key_create_time_survives_a_garbage_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric createTime must not turn a successful walk into an error string."""
    _pin_now(monkeypatch)
    start, end = NOW - 86_400_000, NOW
    rows = [{**_trade(1, created=NOW - 1000), "createTime": "later"}, _trade(2, created=NOW - 2000)]
    fake = _FakeClient(trade_flow={(start, end): _page(rows)})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(ConvertHistoryInput(start_time=start, end_time=end))

    assert "Found **2** conversion(s)" in result
    assert not result.startswith("Error")


def test_envelope_error_only_fires_on_a_real_error_code() -> None:
    assert _envelope_error({"code": 200, "msg": "ok"}) is None
    assert _envelope_error({"code": "000000", "data": []}) is None
    assert _envelope_error([{"fromAsset": "BTC"}]) is None
    assert _envelope_error({"quoteId": "1"}) is None
    assert _envelope_error({"code": -1121, "msg": "Invalid symbol."}) == "Error: Invalid symbol. (code -1121)"


def test_oldest_create_time_ignores_unusable_rows() -> None:
    assert _oldest_create_time([]) is None
    assert _oldest_create_time([{"createTime": None}]) is None
    assert _oldest_create_time([{"createTime": 5}, {"createTime": 3}, {"createTime": "bad"}]) == 3


def test_dedupe_by_order_id_keeps_first_occurrence() -> None:
    rows = [_trade(1, created=10), _trade(2, created=9), _trade(1, created=10)]
    assert [row["orderId"] for row in _dedupe_by_order_id(rows)] == [1, 2]


# -- model validation (nothing reaches the network) -----------------------------------


def test_quote_rejects_both_amounts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)
    with pytest.raises(ValidationError) as exc_info:
        ConvertQuoteInput(from_asset="BTC", to_asset="USDT", from_amount="0.01", to_amount="600")
    assert "exactly one of from_amount" in str(exc_info.value)
    assert fake.calls == []


def test_quote_rejects_neither_amount(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)
    with pytest.raises(ValidationError):
        ConvertQuoteInput(from_asset="BTC", to_asset="USDT")
    assert fake.calls == []


def test_quote_rejects_non_decimal_amount() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ConvertQuoteInput(from_asset="BTC", to_asset="USDT", from_amount="lots")
    assert "must be a plain decimal string" in str(exc_info.value)


def test_quote_rejects_same_asset_and_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ConvertQuoteInput(from_asset="BTC", to_asset="BTC", from_amount="1")
    with pytest.raises(ValidationError):
        ConvertQuoteInput(from_asset="BTC", to_asset="USDT", from_amount="1", bogus=1)  # type: ignore[call-arg]


def test_limit_order_rejects_both_and_neither_amount(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)
    base = {"base_asset": "BTC", "quote_asset": "USDT", "limit_price": "50000", "side": "BUY", "expired_type": "7_D"}
    with pytest.raises(ValidationError) as exc_info:
        PlaceConvertLimitOrderInput(**base, base_amount="0.01", quote_amount="500")
    assert "exactly one of base_amount or quote_amount" in str(exc_info.value)
    with pytest.raises(ValidationError):
        PlaceConvertLimitOrderInput(**base)
    assert fake.calls == []


def test_order_status_requires_exactly_one_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)
    with pytest.raises(ValidationError) as exc_info:
        ConvertOrderStatusInput(order_id="1", quote_id="2")
    assert "exactly one of order_id or quote_id" in str(exc_info.value)
    with pytest.raises(ValidationError):
        ConvertOrderStatusInput()
    assert fake.calls == []


def test_history_rejects_mixing_window_and_walk_params(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient()
    _patch(monkeypatch, fake)
    with pytest.raises(ValidationError) as exc_info:
        ConvertHistoryInput(start_time=1_700_000_000_000, since=1_600_000_000_000)
    assert "do not mix the single-window parameters" in str(exc_info.value)
    assert fake.calls == []


def test_history_rejects_bad_time_string() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ConvertHistoryInput(since="12345")
    assert "since must be epoch ms or ISO-8601" in str(exc_info.value)


def test_history_rejects_limit_over_1000() -> None:
    with pytest.raises(ValidationError):
        ConvertHistoryInput(limit=1001)


# -- binance_get_convert_pairs --------------------------------------------------------


async def test_convert_pairs_happy_path_is_unauthenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [
        {
            "fromAsset": "BTC",
            "toAsset": "USDT",
            "fromAssetMinAmount": "0.0001",
            "fromAssetMaxAmount": "50",
            "toAssetMinAmount": "5",
            "toAssetMaxAmount": "2000000",
        }
    ]
    fake = _FakeClient(routes={CONVERT_EXCHANGE_INFO_PATH: payload})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_pairs(ConvertPairsInput(from_asset="btc", to_asset="usdt"))

    assert "# Binance Convert Pairs" in result
    assert "**BTC → USDT**" in result
    assert "0.0001–50 BTC" in result
    assert "IP weight 3000" in result
    assert fake.calls == [
        ("GET", CONVERT_EXCHANGE_INFO_PATH, {"params": {"fromAsset": "BTC", "toAsset": "USDT"}, "auth": "none"})
    ]


async def test_convert_pairs_drops_none_filters_and_caps_display(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [{"fromAsset": f"A{i}", "toAsset": "USDT"} for i in range(150)]
    fake = _FakeClient(routes={CONVERT_EXCHANGE_INFO_PATH: payload})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_pairs(ConvertPairsInput())

    assert fake.calls[-1][2] == {"params": {}, "auth": "none"}
    assert "Showing **100** of **150** pair(s)" in result
    assert "pass `from_asset`" in result


async def test_convert_pairs_json_carries_every_row(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [{"fromAsset": f"A{i}", "toAsset": "USDT"} for i in range(150)]
    fake = _FakeClient(routes={CONVERT_EXCHANGE_INFO_PATH: payload})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_pairs(ConvertPairsInput(response_format="json"))

    assert len(json.loads(result)["items"]) == 150


async def test_convert_pairs_surfaces_envelope_code(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={CONVERT_EXCHANGE_INFO_PATH: {"code": -1121, "msg": "Invalid symbol."}})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_pairs(ConvertPairsInput(from_asset="NOPE"))

    assert result == "Error: Invalid symbol. (code -1121)"


# -- binance_get_convert_asset_info ---------------------------------------------------


async def test_convert_asset_info_is_signed_and_filters_client_side(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [{"asset": "BTC", "fraction": 8}, {"asset": "USDT", "fraction": 4}]
    fake = _FakeClient(routes={CONVERT_ASSET_INFO_PATH: payload})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_asset_info(ConvertAssetInfoInput(asset="btc"))

    assert "**BTC** — 8 decimal place(s)" in result
    assert "USDT" not in result
    assert fake.calls == [("GET", CONVERT_ASSET_INFO_PATH, {"auth": "signed"})]


# -- binance_get_convert_quote --------------------------------------------------------


async def test_convert_quote_sends_camel_case_and_renders_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "quoteId": "12415572564",
        "ratio": "60000.00",
        "inverseRatio": "0.0000166",
        "validTimestamp": 1_700_000_030_000,
        "toAmount": "600.00",
        "fromAmount": "0.01",
    }
    fake = _FakeClient(routes={CONVERT_GET_QUOTE_PATH: payload})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_quote(
        ConvertQuoteInput(from_asset="btc", to_asset="usdt", from_amount="0.01", wallet_type="SPOT", valid_time="30s")
    )

    assert "# Convert quote — BTC → USDT" in result
    assert "`12415572564`" in result
    assert "1 BTC = 60000 USDT" in result
    assert "1 USDT = 0.0000166 BTC" in result
    assert "- **fromAmount**: 0.01 BTC (from the SPOT wallet)" in result
    assert "- **toAmount**: 600 USDT" in result
    assert "2023-11-14 22:13:50 UTC" in result
    assert "accept with `binance_accept_convert_quote`" in result.lower()
    assert "Nothing has been converted" in result
    assert fake.calls == [
        (
            "POST",
            CONVERT_GET_QUOTE_PATH,
            {
                "params": {
                    "fromAsset": "BTC",
                    "toAsset": "USDT",
                    "fromAmount": "0.01",
                    "walletType": "SPOT",
                    "validTime": "30s",
                },
                "auth": "signed",
            },
        )
    ]


async def test_convert_quote_sends_to_amount_and_drops_optionals(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={CONVERT_GET_QUOTE_PATH: {"quoteId": "1", "ratio": "1", "inverseRatio": "1"}})
    _patch(monkeypatch, fake)

    await binance_get_convert_quote(ConvertQuoteInput(from_asset="USDT", to_asset="BTC", to_amount="0.5"))

    assert fake.params == {"fromAsset": "USDT", "toAsset": "BTC", "toAmount": "0.5"}


# -- binance_accept_convert_quote (GATED) ---------------------------------------------


async def test_accept_quote_echoes_process_without_claiming_a_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"orderId": 933256278426274426, "createTime": 1_700_000_000_000, "orderStatus": "PROCESS"}
    fake = _FakeClient(routes={CONVERT_ACCEPT_QUOTE_PATH: payload})
    _patch(monkeypatch, fake)

    result = await binance_accept_convert_quote(AcceptConvertQuoteInput(quote_id="12415572564"))

    assert "# Convert quote accepted" in result
    assert "`933256278426274426`" in result
    assert "- **orderStatus**: **PROCESS**" in result
    assert "not** confirmed" in result
    assert "The conversion completed" not in result
    assert fake.calls == [("POST", CONVERT_ACCEPT_QUOTE_PATH, {"params": {"quoteId": "12415572564"}, "auth": "signed"})]


async def test_accept_quote_reports_success_and_failure_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={CONVERT_ACCEPT_QUOTE_PATH: {"orderId": 7, "orderStatus": "SUCCESS"}})
    _patch(monkeypatch, fake)
    ok = await binance_accept_convert_quote(AcceptConvertQuoteInput(quote_id="q"))
    assert "- **orderStatus**: **SUCCESS**" in ok
    assert "The conversion completed" in ok

    fake = _FakeClient(routes={CONVERT_ACCEPT_QUOTE_PATH: {"orderId": 8, "orderStatus": "FAIL"}})
    _patch(monkeypatch, fake)
    bad = await binance_accept_convert_quote(AcceptConvertQuoteInput(quote_id="q"))
    assert "- **orderStatus**: **FAIL**" in bad
    assert "no assets were exchanged" in bad
    assert "completed" not in bad


async def test_accept_quote_without_a_status_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={CONVERT_ACCEPT_QUOTE_PATH: {"orderId": 9}})
    _patch(monkeypatch, fake)

    result = await binance_accept_convert_quote(AcceptConvertQuoteInput(quote_id="q"))

    assert "- **orderStatus**: **not reported**" in result
    assert "is unknown from this response" in result


# -- binance_get_convert_order_status -------------------------------------------------


async def test_order_status_by_order_id(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "orderId": 933256278426274426,
        "orderStatus": "SUCCESS",
        "fromAsset": "BTC",
        "fromAmount": "0.01",
        "toAsset": "USDT",
        "toAmount": "600",
        "ratio": "60000",
        "inverseRatio": "0.0000166",
        "createTime": 1_700_000_000_000,
    }
    fake = _FakeClient(routes={CONVERT_ORDER_STATUS_PATH: payload})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_order_status(ConvertOrderStatusInput(order_id="933256278426274426"))

    assert "- **status**: **SUCCESS**" in result
    assert "- **from**: 0.01 BTC" in result
    assert fake.calls == [
        ("GET", CONVERT_ORDER_STATUS_PATH, {"params": {"orderId": "933256278426274426"}, "auth": "signed"})
    ]


async def test_order_status_by_quote_id_drops_order_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={CONVERT_ORDER_STATUS_PATH: {"orderId": 1, "orderStatus": "PROCESS"}})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_order_status(ConvertOrderStatusInput(quote_id="12415572564"))

    assert fake.params == {"quoteId": "12415572564"}
    assert "still processing" in result


# -- binance_get_convert_history: single window ---------------------------------------


async def test_history_defaults_to_the_last_30_days(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_now(monkeypatch)
    fake = _FakeClient(trade_flow={(NOW - MAX_CONVERT_WINDOW_MS, NOW): _page([_trade(1, created=NOW - 1000)])})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(ConvertHistoryInput())

    assert fake.calls == [
        (
            "GET",
            CONVERT_TRADE_FLOW_PATH,
            {
                "params": {"startTime": NOW - MAX_CONVERT_WINDOW_MS, "endTime": NOW, "limit": 100},
                "auth": "signed",
            },
        )
    ]
    assert "Found **1** conversion(s)" in result
    assert "0.01 BTC → 600 USDT" in result
    assert "status **SUCCESS**" in result


async def test_history_fills_the_missing_bound_with_the_30_day_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_now(monkeypatch)
    end = NOW - 100 * 86_400_000
    fake = _FakeClient(trade_flow={})
    _patch(monkeypatch, fake)

    await binance_get_convert_history(ConvertHistoryInput(end_time=end))
    assert fake.windows[-1] == (end - MAX_CONVERT_WINDOW_MS, end)

    start = NOW - 200 * 86_400_000
    await binance_get_convert_history(ConvertHistoryInput(start_time=start))
    assert fake.windows[-1] == (start, start + MAX_CONVERT_WINDOW_MS)


async def test_history_start_only_never_runs_past_now(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_now(monkeypatch)
    fake = _FakeClient(trade_flow={})
    _patch(monkeypatch, fake)

    await binance_get_convert_history(ConvertHistoryInput(start_time=NOW - 10 * 86_400_000))

    assert fake.windows[-1] == (NOW - 10 * 86_400_000, NOW)


async def test_history_rejects_a_span_over_30_days_without_calling(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_now(monkeypatch)
    fake = _FakeClient(trade_flow={})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(ConvertHistoryInput(start_time=NOW - 40 * 86_400_000, end_time=NOW))

    assert result.startswith("Error: the startTime/endTime span")
    assert "cannot exceed 30 days" in result
    assert "40.0 days" in result
    assert fake.calls == []


async def test_history_rejects_reversed_bounds_without_calling(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_now(monkeypatch)
    fake = _FakeClient(trade_flow={})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(ConvertHistoryInput(start_time=NOW, end_time=NOW - 1000))

    assert result == "Error: start_time must be before end_time."
    assert fake.calls == []


async def test_history_narrows_a_window_on_more_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """`moreData` is the only continuation tradeFlow offers: re-ask with endTime = oldest.

    The boundary is INCLUSIVE, so the re-ask returns the row that sat on it again — which
    the orderId dedupe absorbs.
    """
    _pin_now(monkeypatch)
    start, end = NOW - 10 * 86_400_000, NOW
    oldest_first_page = NOW - 3 * 86_400_000
    fake = _FakeClient(
        trade_flow={
            (start, end): _page([_trade(1, created=NOW - 1000), _trade(2, created=oldest_first_page)], more=True),
            (start, oldest_first_page): _page(
                [_trade(2, created=oldest_first_page), _trade(3, created=NOW - 5 * 86_400_000)], more=False
            ),
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(ConvertHistoryInput(start_time=start, end_time=end, limit=2))

    assert fake.windows == [(start, end), (start, oldest_first_page)]
    assert [call[2]["params"]["limit"] for call in fake.calls] == [2, 2]
    assert "Found **3** conversion(s)" in result  # orderId 2 came back twice, counted once
    assert "2 API call(s) spent" in result
    assert "resume_before" not in result  # the window completed


async def test_history_flags_more_data_it_cannot_follow(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_now(monkeypatch)
    start, end = NOW - 86_400_000, NOW
    fake = _FakeClient(trade_flow={(start, end): _page([{"orderId": 1}], more=True)})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(ConvertHistoryInput(start_time=start, end_time=end))

    assert "no way to continue past" in result
    assert "no usable `createTime`" in result
    assert fake.windows == [(start, end)]


async def test_history_keeps_rows_tied_on_the_page_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tie straddling the `limit` cut must not vanish.

    Five conversions, limit=3, and rows 3+4 share one instant: the page returns 1-3, so the
    exclusive `oldest - 1` boundary used to skip row 4 forever. The inclusive boundary
    re-reads the tie and the dedupe absorbs row 3.
    """
    _pin_now(monkeypatch)
    start, end = NOW - 86_400_000, NOW
    tie = NOW - 30_000
    fake = _FakeClient(
        trade_flow={
            (start, end): _page(
                [_trade(1, created=NOW - 10_000), _trade(2, created=NOW - 20_000), _trade(3, created=tie)],
                more=True,
            ),
            # Binance re-answers the inclusive boundary with BOTH tied rows plus the tail.
            (start, tie): _page([_trade(3, created=tie), _trade(4, created=tie), _trade(5, created=NOW - 40_000)]),
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(
        ConvertHistoryInput(start_time=start, end_time=end, limit=3, response_format="json")
    )

    payload = json.loads(result)
    assert fake.windows == [(start, end), (start, tie)]
    assert sorted(row["orderId"] for row in payload["items"]) == [1, 2, 3, 4, 5]
    assert payload["possibly_incomplete"] is False


async def test_history_flags_a_full_page_inside_one_millisecond(monkeypatch: pytest.MonkeyPatch) -> None:
    """`limit` rows sharing the window's end instant cannot be narrowed past — say so.

    Reported honestly instead of returning the partial page as if it were complete: there
    is no `endTime` between `oldest` and `page_end` when they are the same millisecond.
    """
    _pin_now(monkeypatch)
    start, end = NOW - 86_400_000, NOW
    fake = _FakeClient(trade_flow={(start, end): _page([_trade(i, created=end) for i in range(1, 4)], more=True)})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(
        ConvertHistoryInput(start_time=start, end_time=end, limit=3, response_format="json")
    )

    payload = json.loads(result)
    assert fake.windows == [(start, end)]  # no second call: narrowing cannot advance
    assert payload["possibly_incomplete"] is True
    assert len(payload["items"]) == 3

    markdown = await binance_get_convert_history(ConvertHistoryInput(start_time=start, end_time=end, limit=3))
    assert "share one millisecond" in markdown


# -- binance_get_convert_history: the walk --------------------------------------------


async def test_walk_slices_newest_first_into_29_day_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    since = UNTIL - 2 * WALK_WINDOW_MS
    fake = _FakeClient(
        trade_flow={
            (UNTIL - WALK_WINDOW_MS, UNTIL): _page([_trade(1, created=UNTIL - 5000)]),
            (since, UNTIL - WALK_WINDOW_MS - 1): _page([_trade(2, created=since + 5000)]),
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(ConvertHistoryInput(since=since, until=UNTIL))

    assert fake.windows == [
        (UNTIL - WALK_WINDOW_MS, UNTIL),
        (since, UNTIL - WALK_WINDOW_MS - 1),
    ]
    assert all(call[2]["auth"] == "signed" for call in fake.calls)
    assert "(Windowed Walk)" in result
    assert "Found **2** conversion(s)" in result
    assert "2 API call(s) spent" in result
    assert "Stopped early" not in result
    # Newest first.
    assert result.index("orderId `1`") < result.index("orderId `2`")


async def test_walk_stops_at_a_window_boundary_and_emits_the_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Budget exhausted exactly between two windows: the cursor is the next window's end."""
    since = UNTIL - 2 * WALK_WINDOW_MS
    fake = _FakeClient(
        trade_flow={
            (UNTIL - WALK_WINDOW_MS, UNTIL): _page([_trade(1, created=UNTIL - 5000)]),
            (since, UNTIL - WALK_WINDOW_MS - 1): _page([_trade(2, created=since + 5000)]),
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(
        ConvertHistoryInput(since=since, until=UNTIL, max_calls=1, response_format="json")
    )

    payload = json.loads(result)
    assert fake.windows == [(UNTIL - WALK_WINDOW_MS, UNTIL)]
    assert payload["calls_used"] == 1
    assert payload["stop_reason"] == "budget"
    assert payload["resume_before"] == UNTIL - WALK_WINDOW_MS - 1
    assert payload["no_progress"] is False
    assert [row["orderId"] for row in payload["items"]] == [1]


async def test_walk_stops_mid_window_with_the_narrowed_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Budget exhausted inside a window: the cursor is the endTime the next call would use."""
    since = UNTIL - WALK_WINDOW_MS
    oldest = UNTIL - 5_000_000
    fake = _FakeClient(trade_flow={(since, UNTIL): _page([_trade(1, created=oldest)], more=True)})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(
        ConvertHistoryInput(since=since, until=UNTIL, max_calls=1, response_format="json")
    )

    payload = json.loads(result)
    assert payload["calls_used"] == 1
    # Inclusive, like the narrowing itself: an exclusive cursor would skip rows tied on
    # `oldest` that this page never reached.
    assert payload["resume_before"] == oldest
    assert payload["no_progress"] is False
    assert [row["orderId"] for row in payload["items"]] == [1]


async def test_walk_resume_leaves_no_gap_and_no_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Feed the emitted cursor back in: the two runs together cover the whole range once."""
    since = UNTIL - 2 * WALK_WINDOW_MS
    boundary = UNTIL - WALK_WINDOW_MS - 1
    fake = _FakeClient(
        trade_flow={
            (UNTIL - WALK_WINDOW_MS, UNTIL): _page([_trade(1, created=UNTIL - 5000)]),
            (since, boundary): _page([_trade(2, created=since + 5000)]),
            # A resumed walk re-derives its window from the cursor; that range is the
            # still-unfetched tail, so it must answer with the same rows.
            (max(boundary - WALK_WINDOW_MS, since), boundary): _page([_trade(2, created=since + 5000)]),
        }
    )
    _patch(monkeypatch, fake)

    first = json.loads(
        await binance_get_convert_history(
            ConvertHistoryInput(since=since, until=UNTIL, max_calls=1, response_format="json")
        )
    )
    cursor = first["resume_before"]
    assert cursor == boundary

    second = json.loads(
        await binance_get_convert_history(
            ConvertHistoryInput(since=since, resume_before=cursor, response_format="json")
        )
    )

    assert second["resume_before"] is None
    assert second["stop_reason"] is None
    ids = [row["orderId"] for row in first["items"]] + [row["orderId"] for row in second["items"]]
    assert sorted(ids) == [1, 2]  # no gap (2 was fetched) and no duplicate (1 was not refetched)


async def test_walk_error_returns_partial_rows_plus_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    since = UNTIL - 2 * WALK_WINDOW_MS
    fake = _FakeClient(
        trade_flow={
            (UNTIL - WALK_WINDOW_MS, UNTIL): _page([_trade(1, created=UNTIL - 5000)]),
            (since, UNTIL - WALK_WINDOW_MS - 1): _status_error(429, {"code": -1003, "msg": "Too many requests."}),
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(ConvertHistoryInput(since=since, until=UNTIL))

    assert "Stopped early" in result
    assert "Error (429)" in result
    assert "orderId `1`" in result  # the rows already collected survive the failure
    assert "Found **1** conversion(s)" in result
    assert "2 API call(s) spent" in result  # the failed request counts against the budget
    assert f"resume_before={UNTIL - WALK_WINDOW_MS - 1}" in result


async def test_walk_error_on_the_first_call_reports_no_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    since = UNTIL - WALK_WINDOW_MS
    fake = _FakeClient(trade_flow={(since, UNTIL): _status_error(500, {"code": -1000, "msg": "Unknown."})})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(ConvertHistoryInput(since=since, until=UNTIL, response_format="json"))
    payload = json.loads(result)

    assert payload["calls_used"] == 1
    assert payload["items"] == []
    # No cursor at all: one that does not advance past the upper bound would just make a
    # JSON caller repeat the identical UID-3000 call.
    assert payload["resume_before"] is None
    assert payload["no_progress"] is True

    markdown = await binance_get_convert_history(ConvertHistoryInput(since=since, until=UNTIL))
    assert "No cursor is offered" in markdown
    assert "resume_before=" not in markdown


async def test_walk_dedupes_overlapping_rows_across_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    since = UNTIL - 2 * WALK_WINDOW_MS
    duplicate = _trade(1, created=UNTIL - WALK_WINDOW_MS)
    fake = _FakeClient(
        trade_flow={
            (UNTIL - WALK_WINDOW_MS, UNTIL): _page([duplicate]),
            (since, UNTIL - WALK_WINDOW_MS - 1): _page([duplicate, _trade(2, created=since + 1)]),
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(ConvertHistoryInput(since=since, until=UNTIL, response_format="json"))

    assert [row["orderId"] for row in json.loads(result)["items"]] == [1, 2]


async def test_walk_with_since_after_until_does_not_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(trade_flow={})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_history(ConvertHistoryInput(since=UNTIL, until=UNTIL - 1000))

    assert "No window to walk" in result
    assert fake.calls == []


# -- limit orders ---------------------------------------------------------------------


async def test_place_limit_order_sends_camel_case_and_never_claims_a_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient(routes={CONVERT_LIMIT_PLACE_PATH: {"orderId": 1603680255057330400, "status": "SUCCESS"}})
    _patch(monkeypatch, fake)

    result = await binance_place_convert_limit_order(
        PlaceConvertLimitOrderInput(
            base_asset="btc",
            quote_asset="usdt",
            limit_price="50000",
            side="BUY",
            expired_type="7_D",
            quote_amount="500",
            wallet_type="SPOT_FUNDING",
        )
    )

    assert "# Convert limit order placed — BUY BTC/USDT" in result
    assert "`1603680255057330400`" in result
    assert "- **status**: **SUCCESS**" in result
    assert "Nothing has been converted yet" in result
    assert fake.calls == [
        (
            "POST",
            CONVERT_LIMIT_PLACE_PATH,
            {
                "params": {
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "limitPrice": "50000",
                    "side": "BUY",
                    "expiredType": "7_D",
                    "quoteAmount": "500",
                    "walletType": "SPOT_FUNDING",
                },
                "auth": "signed",
            },
        )
    ]


async def test_place_limit_order_sends_base_amount_and_drops_wallet_type(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={CONVERT_LIMIT_PLACE_PATH: {"orderId": 2, "status": "SUCCESS"}})
    _patch(monkeypatch, fake)

    await binance_place_convert_limit_order(
        PlaceConvertLimitOrderInput(
            base_asset="BTC",
            quote_asset="USDT",
            limit_price="0.00000100",
            side="SELL",
            expired_type="30_D",
            base_amount="0.010",
        )
    )

    # Decimal strings travel verbatim — no float round-trip, no trailing-zero rewrite.
    assert fake.params == {
        "baseAsset": "BTC",
        "quoteAsset": "USDT",
        "limitPrice": "0.00000100",
        "side": "SELL",
        "expiredType": "30_D",
        "baseAmount": "0.010",
    }


async def test_cancel_limit_order_echoes_status(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={CONVERT_LIMIT_CANCEL_PATH: {"orderId": 1603680255057330400, "status": "CANCELED"}})
    _patch(monkeypatch, fake)

    result = await binance_cancel_convert_limit_order(CancelConvertLimitOrderInput(order_id="1603680255057330400"))

    assert "# Convert limit order cancellation" in result
    assert "- **status**: **CANCELED**" in result
    assert fake.calls == [
        ("POST", CONVERT_LIMIT_CANCEL_PATH, {"params": {"orderId": "1603680255057330400"}, "auth": "signed"})
    ]


async def test_open_limit_orders_renders_rows_and_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {**_trade(5, created=1_700_000_000_000), "orderStatus": "PENDING", "expiredTimestamp": 1_700_600_000_000}
    fake = _FakeClient(routes={CONVERT_LIMIT_OPEN_ORDERS_PATH: {"list": [row]}})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_open_limit_orders(ConvertOpenLimitOrdersInput())

    assert "**1** resting order(s)" in result
    assert "status **PENDING**" in result
    assert "expires 2023-11-21" in result
    assert fake.calls == [("GET", CONVERT_LIMIT_OPEN_ORDERS_PATH, {"auth": "signed"})]


async def test_open_limit_orders_handles_an_empty_book(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={CONVERT_LIMIT_OPEN_ORDERS_PATH: {"list": []}})
    _patch(monkeypatch, fake)

    result = await binance_get_convert_open_limit_orders(ConvertOpenLimitOrdersInput())

    assert "No convert limit orders are resting" in result


# -- the kill-switch, verbatim, on all three gated tools ------------------------------


async def test_accept_quote_kill_switch_message_is_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client's kill-switch surfaces as `Error: … trading is disabled …`, unaltered."""
    message = (
        "POST /sapi/v1/convert/acceptQuote would move funds or change account state, but trading is disabled. "
        "Set BINANCE_ALLOW_TRADING=1 to enable order placement/cancellation, transfers, convert and "
        "algo orders. Dry-run validation (POST /api/v3/order/test) works without it."
    )
    fake = _FakeClient(routes={CONVERT_ACCEPT_QUOTE_PATH: TradingDisabledError(message)})
    _patch(monkeypatch, fake)

    result = await binance_accept_convert_quote(AcceptConvertQuoteInput(quote_id="q"))

    assert result == f"Error: {message}"
    assert "trading is disabled" in result


async def test_place_limit_order_kill_switch_message_is_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    message = (
        "POST /sapi/v1/convert/limit/placeOrder would move funds or change account state, but trading is "
        "disabled. Set BINANCE_ALLOW_TRADING=1 to enable order placement/cancellation, transfers, convert "
        "and algo orders. Dry-run validation (POST /api/v3/order/test) works without it."
    )
    fake = _FakeClient(routes={CONVERT_LIMIT_PLACE_PATH: TradingDisabledError(message)})
    _patch(monkeypatch, fake)

    result = await binance_place_convert_limit_order(
        PlaceConvertLimitOrderInput(
            base_asset="BTC",
            quote_asset="USDT",
            limit_price="50000",
            side="BUY",
            expired_type="1_D",
            quote_amount="500",
        )
    )

    assert result == f"Error: {message}"


async def test_cancel_limit_order_kill_switch_message_is_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    message = (
        "POST /sapi/v1/convert/limit/cancelOrder would move funds or change account state, but trading is "
        "disabled. Set BINANCE_ALLOW_TRADING=1 to enable order placement/cancellation, transfers, convert "
        "and algo orders. Dry-run validation (POST /api/v3/order/test) works without it."
    )
    fake = _FakeClient(routes={CONVERT_LIMIT_CANCEL_PATH: TradingDisabledError(message)})
    _patch(monkeypatch, fake)

    result = await binance_cancel_convert_limit_order(CancelConvertLimitOrderInput(order_id="1"))

    assert result == f"Error: {message}"


# -- one error path per tool ----------------------------------------------------------


async def test_every_tool_surfaces_a_binance_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error = _status_error(400, {"code": -1121, "msg": "Invalid symbol."})
    calls: list[tuple[str, Any]] = [
        (CONVERT_EXCHANGE_INFO_PATH, lambda: binance_get_convert_pairs(ConvertPairsInput(from_asset="BTC"))),
        (CONVERT_ASSET_INFO_PATH, lambda: binance_get_convert_asset_info(ConvertAssetInfoInput())),
        (
            CONVERT_GET_QUOTE_PATH,
            lambda: binance_get_convert_quote(ConvertQuoteInput(from_asset="BTC", to_asset="USDT", from_amount="0.01")),
        ),
        (CONVERT_ACCEPT_QUOTE_PATH, lambda: binance_accept_convert_quote(AcceptConvertQuoteInput(quote_id="q"))),
        (
            CONVERT_ORDER_STATUS_PATH,
            lambda: binance_get_convert_order_status(ConvertOrderStatusInput(order_id="1")),
        ),
        (
            CONVERT_LIMIT_PLACE_PATH,
            lambda: binance_place_convert_limit_order(
                PlaceConvertLimitOrderInput(
                    base_asset="BTC",
                    quote_asset="USDT",
                    limit_price="50000",
                    side="BUY",
                    expired_type="1_D",
                    quote_amount="500",
                )
            ),
        ),
        (
            CONVERT_LIMIT_CANCEL_PATH,
            lambda: binance_cancel_convert_limit_order(CancelConvertLimitOrderInput(order_id="1")),
        ),
        (
            CONVERT_LIMIT_OPEN_ORDERS_PATH,
            lambda: binance_get_convert_open_limit_orders(ConvertOpenLimitOrdersInput()),
        ),
    ]
    for path, call in calls:
        fake = _FakeClient(routes={path: error})
        _patch(monkeypatch, fake)
        result = await call()
        assert result.startswith("Error (400)"), path
        assert "Invalid symbol." in result, path

    # The history walk reports the failure too, alongside whatever it had collected.
    fake = _FakeClient(trade_flow={(NOW - MAX_CONVERT_WINDOW_MS, NOW): error})
    _pin_now(monkeypatch)
    _patch(monkeypatch, fake)
    assert "Error (400)" in await binance_get_convert_history(ConvertHistoryInput())


# -- live smokes: READ tools only -----------------------------------------------------
#
# There is deliberately NO live test for the three gated tools (acceptQuote,
# limit/placeOrder, limit/cancelOrder): they move real funds and `/sapi` does not exist
# on the spot testnet, so there is nowhere safe to run them.


@pytest.mark.live
async def test_convert_pairs_live() -> None:
    """Unauthenticated, but IP weight 3000 — filtered to one pair to stay cheap."""
    result = await binance_get_convert_pairs(ConvertPairsInput(from_asset="BTC", to_asset="USDT"))
    assert isinstance(result, str)


@pytest.mark.live
async def test_convert_asset_info_live() -> None:
    """Signed /sapi: real account only — the spot testnet has no /sapi endpoints."""
    result = await binance_get_convert_asset_info(ConvertAssetInfoInput(asset="BTC"))
    assert isinstance(result, str)


@pytest.mark.live
async def test_convert_history_live() -> None:
    """Signed /sapi (no testnet): one window, one call, UID weight 3000."""
    result = await binance_get_convert_history(ConvertHistoryInput(max_calls=1))
    assert isinstance(result, str)


@pytest.mark.live
async def test_convert_open_limit_orders_live() -> None:
    """Signed /sapi (no testnet): UID weight 3000, read-only."""
    result = await binance_get_convert_open_limit_orders(ConvertOpenLimitOrdersInput())
    assert isinstance(result, str)
