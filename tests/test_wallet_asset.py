"""Unit tests for the wallet-asset tools against a fake client.

Every tool asserts BOTH the rendered string and the captured
`(method, path, auth, params)` — the exact compact dict, with `None` fields dropped and
amounts travelling as strings. The validation tests additionally assert
`fake.calls == []`: a rejected window or a missing isolated-margin leg must die locally,
before anything is signed or sent.

The two fund-moving tools (`binance_transfer_between_wallets`,
`binance_convert_dust_to_bnb`) have NO live/trading smoke test, by design: `/sapi` does
not exist on the spot testnet, so there is nowhere to exercise them that is not the real
account with real money.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from binance_mcp.client import Repeat, TradingDisabledError, build_query
from binance_mcp.formatters import ResponseFormat
from binance_mcp.tools.wallet_asset import (
    AssetDetailInput,
    AssetDividendsInput,
    ConvertDustInput,
    DustConvertibleInput,
    DustLogInput,
    FundingWalletInput,
    TradeFeesInput,
    TransferBetweenWalletsInput,
    TransferHistoryInput,
    TransferType,
    UserAssetsInput,
    WalletBalancesInput,
    binance_convert_dust_to_bnb,
    binance_get_asset_detail,
    binance_get_asset_dividends,
    binance_get_dust_convertible,
    binance_get_dust_log,
    binance_get_funding_wallet,
    binance_get_trade_fees,
    binance_get_transfer_history,
    binance_get_user_assets,
    binance_get_wallet_balances,
    binance_transfer_between_wallets,
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
    monkeypatch.setattr("binance_mcp.tools.wallet_asset.get_client", lambda: fake)


FUNDING_ASSETS: list[dict[str, Any]] = [
    {
        "asset": "USDT",
        "free": "12.50000000",
        "locked": "0",
        "freeze": "0",
        "withdrawing": "0",
        "btcValuation": "0.00020000",
    },
    {
        "asset": "BNB",
        "free": "0.10000000",
        "locked": "0",
        "freeze": "0",
        "withdrawing": "0",
        "btcValuation": "0.00090000",
    },
]

USER_ASSETS: list[dict[str, Any]] = [
    {
        "asset": "BTC",
        "free": "0.00500000",
        "locked": "0",
        "freeze": "0",
        "withdrawing": "0",
        "ipoable": "0",
        "btcValuation": "0.00500000",
    }
]


# -- binance_get_funding_wallet ---------------------------------------------------------


async def test_funding_wallet_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/get-funding-asset": FUNDING_ASSETS})
    _patch(monkeypatch, fake)

    result = await binance_get_funding_wallet(FundingWalletInput(need_btc_valuation=True))

    assert "# Binance Funding wallet" in result
    assert "Binance Pay, Binance Card and Binance Gift Card" in result
    assert "| BNB | 0.1 |" in result  # highest BTC valuation sorts first
    assert "Total across all returned assets: 0.0011 BTC" in result
    method, path, kwargs = fake.calls[0]
    assert (method, path) == ("POST", "/sapi/v1/asset/get-funding-asset")
    assert kwargs["auth"] == "signed"
    assert kwargs["params"] == {"needBtcValuation": True}


async def test_funding_wallet_asset_filter_is_uppercased(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/get-funding-asset": []})
    _patch(monkeypatch, fake)

    result = await binance_get_funding_wallet(FundingWalletInput(asset="usdt"))

    assert "_No balances returned" in result
    assert fake.calls[0][2]["params"] == {"needBtcValuation": False, "asset": "USDT"}


async def test_funding_wallet_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/get-funding-asset": FUNDING_ASSETS})
    _patch(monkeypatch, fake)

    result = await binance_get_funding_wallet(FundingWalletInput(response_format=ResponseFormat.JSON))

    assert result.strip().startswith("[")
    assert '"asset": "USDT"' in result


async def test_funding_wallet_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/get-funding-asset": _status_error(
                "POST",
                "/sapi/v1/asset/get-funding-asset",
                401,
                {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."},
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_funding_wallet(FundingWalletInput())

    assert result.startswith("Error (401)")
    assert "(code -2015)" in result


# -- binance_get_user_assets ------------------------------------------------------------


async def test_user_assets_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v3/asset/getUserAsset": USER_ASSETS})
    _patch(monkeypatch, fake)

    result = await binance_get_user_assets(UserAssetsInput(asset="btc", need_btc_valuation=True))

    assert "# Binance Spot wallet assets" in result
    assert "| BTC | 0.005 |" in result
    assert "ipoable" in result
    method, path, kwargs = fake.calls[0]
    assert (method, path) == ("POST", "/sapi/v3/asset/getUserAsset")
    assert kwargs["auth"] == "signed"
    assert kwargs["params"] == {"needBtcValuation": True, "asset": "BTC"}


async def test_user_assets_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v3/asset/getUserAsset": _status_error(
                "POST", "/sapi/v3/asset/getUserAsset", 404, {"code": -1121, "msg": "Invalid symbol."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_user_assets(UserAssetsInput())

    assert result.startswith("Error (404)")
    assert "/sapi" in result


# -- binance_get_wallet_balances --------------------------------------------------------


async def test_wallet_balances_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/wallet/balance": [
                {"activate": True, "balance": "10.5", "walletName": "Spot"},
                {"activate": True, "balance": "1.5", "walletName": "Funding"},
                {"activate": False, "balance": "0", "walletName": "Options"},
            ]
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_wallet_balances(WalletBalancesInput(quote_asset="usdt"))

    assert "# Binance wallet balances (valued in USDT)" in result
    assert "| Funding | True | 1.5 |" in result
    assert "Total across all wallets: 12 USDT" in result
    method, path, kwargs = fake.calls[0]
    assert (method, path) == ("GET", "/sapi/v1/asset/wallet/balance")
    assert kwargs["auth"] == "signed"
    assert kwargs["params"] == {"quoteAsset": "USDT"}


async def test_wallet_balances_without_quote_asset_sends_no_params(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/wallet/balance": []})
    _patch(monkeypatch, fake)

    result = await binance_get_wallet_balances(WalletBalancesInput())

    assert "_Binance returned no wallets._" in result
    assert fake.calls[0][2]["params"] is None


async def test_wallet_balances_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/wallet/balance": _status_error(
                "GET", "/sapi/v1/asset/wallet/balance", 429, {"code": -1003, "msg": "Too many requests."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_wallet_balances(WalletBalancesInput())

    assert result.startswith("Error (429)")


# -- binance_get_transfer_history -------------------------------------------------------


TRANSFER_HISTORY: dict[str, Any] = {
    "total": 42,
    "rows": [
        {
            "asset": "USDT",
            "amount": "25.50000000",
            "type": "MAIN_FUNDING",
            "status": "CONFIRMED",
            "tranId": 11945860693,
            "timestamp": 1758500000000,
        },
        {
            "asset": "USDT",
            "amount": "10.00000000",
            "type": "MAIN_FUNDING",
            "status": "CONFIRMED",
            "tranId": 11945860694,
            "timestamp": 1758400000000,
        },
    ],
}


async def test_transfer_history_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/transfer": TRANSFER_HISTORY})
    _patch(monkeypatch, fake)

    result = await binance_get_transfer_history(
        TransferHistoryInput(type=TransferType.MAIN_FUNDING, start_time="2026-09-01", page=2, limit=100)
    )

    assert "# Binance transfers — MAIN_FUNDING" in result
    assert "of total **42**" in result
    assert "| 11945860693 |" in result
    assert "Total moved in the rows above: 35.5 USDT" in result
    method, path, kwargs = fake.calls[0]
    assert (method, path) == ("GET", "/sapi/v1/asset/transfer")
    assert kwargs["auth"] == "signed"
    assert kwargs["params"] == {
        "type": "MAIN_FUNDING",
        "current": 2,
        "size": 100,
        "startTime": 1788220800000,  # 2026-09-01T00:00:00Z
    }


async def test_transfer_history_defaults_and_more_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/transfer": TRANSFER_HISTORY})
    _patch(monkeypatch, fake)

    result = await binance_get_transfer_history(TransferHistoryInput(type=TransferType.FUNDING_MAIN))

    assert "More available — request page 2." in result
    assert fake.calls[0][2]["params"] == {"type": "FUNDING_MAIN", "current": 1, "size": 20}


async def test_transfer_history_rejects_inverted_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/transfer": TRANSFER_HISTORY})
    _patch(monkeypatch, fake)

    result = await binance_get_transfer_history(
        TransferHistoryInput(type=TransferType.MAIN_FUNDING, start_time="2026-09-10", end_time="2026-09-01")
    )

    assert result == "Error: end_time must be after start_time."
    assert fake.calls == []


async def test_transfer_history_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/transfer": {"total": 0, "rows": []}})
    _patch(monkeypatch, fake)

    result = await binance_get_transfer_history(TransferHistoryInput(type=TransferType.MAIN_MARGIN))

    assert "last 7 days" in result
    assert "6 months" in result


async def test_transfer_history_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/transfer": _status_error(
                "GET",
                "/sapi/v1/asset/transfer",
                400,
                {"code": -1102, "msg": "Mandatory parameter 'type' was not sent."},
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_transfer_history(TransferHistoryInput(type=TransferType.MAIN_FUNDING))

    assert result.startswith("Error (400)")
    assert "(code -1102)" in result


# -- binance_get_dust_log ---------------------------------------------------------------


DUST_LOG: dict[str, Any] = {
    "total": 1,
    "userAssetDribblets": [
        {
            "operateTime": 1758500000000,
            "totalTransferedAmount": "0.00132256",
            "totalServiceChargeAmount": "0.00002699",
            "transId": 45178372831,
            "userAssetDribbletDetails": [
                {
                    "transId": 4359321,
                    "serviceChargeAmount": "0.000009",
                    "amount": "0.0009",
                    "operateTime": 1758500000000,
                    "transferedAmount": "0.000441",
                    "fromAsset": "USDT",
                },
                {
                    "transId": 4359322,
                    "serviceChargeAmount": "0.00001799",
                    "amount": "0.0009",
                    "operateTime": 1758500000000,
                    "transferedAmount": "0.00088156",
                    "fromAsset": "ETH",
                },
            ],
        }
    ],
}


async def test_dust_log_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/dribblet": DUST_LOG})
    _patch(monkeypatch, fake)

    result = await binance_get_dust_log(DustLogInput(account_type="SPOT", start_time=1756684800000))

    assert "# Binance dust conversion log" in result
    assert "transId `45178372831`" in result
    assert "| USDT | 0.0009 | 0.000441 |" in result.replace(" 0.000009 |", " 0.000009 |")
    assert "| ETH |" in result
    method, path, kwargs = fake.calls[0]
    assert (method, path) == ("GET", "/sapi/v1/asset/dribblet")
    assert kwargs["auth"] == "signed"
    assert kwargs["params"] == {"accountType": "SPOT", "startTime": 1756684800000}


async def test_dust_log_empty_sends_no_params(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/dribblet": {"total": 0, "userAssetDribblets": []}})
    _patch(monkeypatch, fake)

    result = await binance_get_dust_log(DustLogInput())

    assert "last 100 records" in result
    assert fake.calls[0][2]["params"] is None


async def test_dust_log_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/dribblet": _status_error(
                "GET", "/sapi/v1/asset/dribblet", 401, {"code": -2015, "msg": "Invalid API-key."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_dust_log(DustLogInput())

    assert result.startswith("Error (401)")


# -- binance_get_dust_convertible -------------------------------------------------------


DUST_BTC: dict[str, Any] = {
    "details": [
        {
            "asset": "ADA",
            "assetFullName": "ADA",
            "amountFree": "6.21",
            "toBTC": "0.00016848",
            "toBNB": "0.01777302",
            "toBNBOffExchange": "0.01741756",
            "exchange": "0.00035546",
        }
    ],
    "totalTransferBtc": "0.00016848",
    "totalTransferBNB": "0.01777302",
    "dribbletPercentage": "0.02",
}


async def test_dust_convertible_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/dust-btc": DUST_BTC})
    _patch(monkeypatch, fake)

    result = await binance_get_dust_convertible(DustConvertibleInput(account_type="MARGIN"))

    assert "# Binance convertible dust" in result
    assert "nothing was converted by this call" in result
    assert "| ADA | ADA | 6.21 | 0.00016848 | 0.01777302 | 0.01741756 |" in result
    assert "total BNB you would receive**: 0.01777302" in result
    assert "irreversible" in result
    method, path, kwargs = fake.calls[0]
    assert (method, path) == ("POST", "/sapi/v1/asset/dust-btc")
    assert kwargs["auth"] == "signed"
    assert kwargs["params"] == {"accountType": "MARGIN"}


async def test_dust_convertible_empty_sends_no_params(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/dust-btc": {"details": []}})
    _patch(monkeypatch, fake)

    result = await binance_get_dust_convertible(DustConvertibleInput())

    assert "No balances currently qualify" in result
    assert fake.calls[0][2]["params"] is None


async def test_dust_convertible_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/dust-btc": _status_error(
                "POST", "/sapi/v1/asset/dust-btc", 400, {"code": -1100, "msg": "Illegal characters found."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_dust_convertible(DustConvertibleInput())

    assert result.startswith("Error (400)")


# -- binance_get_asset_detail -----------------------------------------------------------


async def test_asset_detail_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/assetDetail": {
                "CTR": {
                    "minWithdrawAmount": "70.00000000",
                    "depositStatus": False,
                    "withdrawFee": 35,
                    "withdrawStatus": True,
                    "depositTip": "Delisted, Deposit Suspended",
                },
                "BTC": {
                    "minWithdrawAmount": "0.001",
                    "depositStatus": True,
                    "withdrawFee": 0.0005,
                    "withdrawStatus": True,
                },
            }
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_asset_detail(AssetDetailInput())

    assert "# Binance asset detail" in result
    assert "of **2** asset(s)" in result
    assert "| BTC | True | True | 0.0005 | 0.001 |  |" in result
    assert "Delisted, Deposit Suspended" in result
    method, path, kwargs = fake.calls[0]
    assert (method, path) == ("GET", "/sapi/v1/asset/assetDetail")
    assert kwargs["auth"] == "signed"
    assert kwargs["params"] is None


async def test_asset_detail_caps_at_fifty(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {f"A{i:03d}": {"depositStatus": True, "withdrawStatus": True} for i in range(60)}
    fake = _FakeClient(routes={"/sapi/v1/asset/assetDetail": payload})
    _patch(monkeypatch, fake)

    result = await binance_get_asset_detail(AssetDetailInput(asset="a000"))

    assert "Showing **50** of **60** asset(s)" in result
    assert "10 more asset(s) not shown" in result
    assert fake.calls[0][2]["params"] == {"asset": "A000"}


async def test_asset_detail_envelope_is_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 200 body carrying `{code, msg}` is an error here too, not an empty asset map.

    The response is normally a map keyed by asset, but Binance asset codes are
    uppercase, so a lowercase `code` key can only ever be the error envelope.
    """
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/assetDetail": {
                "code": -2015,
                "msg": "Invalid API-key, IP, or permissions for action.",
            }
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_asset_detail(AssetDetailInput())

    assert result == "Error: Invalid API-key, IP, or permissions for action. (code -2015)"
    assert "Binance returned no assets" not in result


async def test_asset_detail_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/assetDetail": _status_error(
                "GET", "/sapi/v1/asset/assetDetail", 500, {"code": -1000, "msg": "Unknown error."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_asset_detail(AssetDetailInput())

    assert result.startswith("Error (500)")
    assert "UNKNOWN" in result


# -- binance_get_trade_fees -------------------------------------------------------------


async def test_trade_fees_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/tradeFee": [{"symbol": "BTCUSDT", "makerCommission": "0.001", "takerCommission": "0.001"}]
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_trade_fees(TradeFeesInput(symbol="btcusdt"))

    assert "| BTCUSDT | 0.1% | 0.1% |" in result
    method, path, kwargs = fake.calls[0]
    assert (method, path) == ("GET", "/sapi/v1/asset/tradeFee")
    assert kwargs["auth"] == "signed"
    assert kwargs["params"] == {"symbol": "BTCUSDT"}


async def test_trade_fees_caps_at_fifty(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"symbol": f"SYM{i}USDT", "makerCommission": "0.001", "takerCommission": "0.001"} for i in range(55)]
    fake = _FakeClient(routes={"/sapi/v1/asset/tradeFee": rows})
    _patch(monkeypatch, fake)

    result = await binance_get_trade_fees(TradeFeesInput())

    assert "Showing **50** of **55** symbol(s)" in result
    assert "5 more symbol(s) not shown" in result
    assert fake.calls[0][2]["params"] is None


async def test_trade_fees_single_object_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """With `symbol` set Binance answers with a bare object, not a one-element array."""
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/tradeFee": {
                "symbol": "ETHUSDT",
                "makerCommission": "0.00075",
                "takerCommission": "0.001",
            }
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_trade_fees(TradeFeesInput(symbol="ETHUSDT"))

    assert "of **1** symbol(s)" in result
    assert "| ETHUSDT | 0.075% | 0.1% |" in result


async def test_trade_fees_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/tradeFee": _status_error(
                "GET", "/sapi/v1/asset/tradeFee", 400, {"code": -1121, "msg": "Invalid symbol."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_trade_fees(TradeFeesInput(symbol="NOPEUSDT"))

    assert result.startswith("Error (400)")
    assert "(code -1121)" in result


# -- binance_get_asset_dividends --------------------------------------------------------


DIVIDENDS: dict[str, Any] = {
    "rows": [
        {
            "id": 1637366104,
            "amount": "10.00000000",
            "asset": "BHFT",
            "divTime": 1563189166000,
            "enInfo": "BHFT distribution",
            "tranId": 2968885920,
        },
        {
            "id": 1631750237,
            "amount": "0.00092003",
            "asset": "BNB",
            "divTime": 1563189165000,
            "enInfo": "Trading rebate",
            "tranId": 2968885920,
        },
    ],
    "total": 2,
}


async def test_asset_dividends_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/assetDividend": DIVIDENDS})
    _patch(monkeypatch, fake)

    result = await binance_get_asset_dividends(
        AssetDividendsInput(asset="bnb", start_time="2026-01-01", end_time="2026-03-01", limit=500)
    )

    assert "# Binance asset dividend record" in result
    assert "Trading rebate" in result
    assert "Total credited in the rows above: 10 BHFT, 0.00092003 BNB" in result
    method, path, kwargs = fake.calls[0]
    assert (method, path) == ("GET", "/sapi/v1/asset/assetDividend")
    assert kwargs["auth"] == "signed"
    assert kwargs["params"] == {
        "limit": 500,
        "asset": "BNB",
        "startTime": 1767225600000,
        "endTime": 1772323200000,
    }


async def test_asset_dividends_rejects_window_over_180_days(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 180-day cap is enforced locally — nothing is signed or sent."""
    fake = _FakeClient(routes={"/sapi/v1/asset/assetDividend": DIVIDENDS})
    _patch(monkeypatch, fake)

    result = await binance_get_asset_dividends(AssetDividendsInput(start_time="2026-01-01", end_time="2026-09-01"))

    assert result.startswith("Error: the start_time/end_time window spans")
    assert "at most 180 days" in result
    assert fake.calls == []


async def test_asset_dividends_accepts_exactly_180_days(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap is inclusive: 180 days exactly must still go through."""
    fake = _FakeClient(routes={"/sapi/v1/asset/assetDividend": {"rows": [], "total": 0}})
    _patch(monkeypatch, fake)

    start = 1767225600000
    result = await binance_get_asset_dividends(
        AssetDividendsInput(start_time=start, end_time=start + 180 * 24 * 60 * 60 * 1000)
    )

    assert "_No distributions in range._" in result
    assert len(fake.calls) == 1


async def test_asset_dividends_rejects_inverted_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/assetDividend": DIVIDENDS})
    _patch(monkeypatch, fake)

    result = await binance_get_asset_dividends(AssetDividendsInput(start_time="2026-09-10", end_time="2026-09-01"))

    assert result == "Error: end_time must be after start_time."
    assert fake.calls == []


async def test_asset_dividends_rejects_one_sided_window_over_180_days(monkeypatch: pytest.MonkeyPatch) -> None:
    """With only `start_time`, `end_time` defaults to now — and the cap still applies."""
    fake = _FakeClient(routes={"/sapi/v1/asset/assetDividend": DIVIDENDS})
    _patch(monkeypatch, fake)

    result = await binance_get_asset_dividends(AssetDividendsInput(start_time="2020-01-01"))

    assert result.startswith("Error: the start_time/end_time window spans")
    assert "at most 180 days" in result
    assert fake.calls == []


async def test_asset_dividends_rejects_unparseable_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_to_ms` fails with a readable message naming the field, before any request."""
    fake = _FakeClient(routes={"/sapi/v1/asset/assetDividend": DIVIDENDS})
    _patch(monkeypatch, fake)

    result = await binance_get_asset_dividends(AssetDividendsInput(start_time="last tuesday"))

    assert result.startswith("Error: start_time must be an epoch-ms integer")
    assert "ISO-8601" in result
    assert "'last tuesday'" in result
    assert fake.calls == []


async def test_asset_dividends_short_numeric_string_is_not_epoch_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    """A numeric string under 12 digits is a typo, not a timestamp — say so, don't send it."""
    fake = _FakeClient(routes={"/sapi/v1/asset/assetDividend": DIVIDENDS})
    _patch(monkeypatch, fake)

    result = await binance_get_asset_dividends(AssetDividendsInput(start_time="17672256"))

    assert result.startswith("Error: start_time must be an epoch-ms integer")
    assert fake.calls == []


async def test_asset_dividends_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/assetDividend": _status_error(
                "GET", "/sapi/v1/asset/assetDividend", 400, {"code": -1127, "msg": "More than 180 days."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_get_asset_dividends(AssetDividendsInput())

    assert result.startswith("Error (400)")
    assert "(code -1127)" in result


# -- binance_transfer_between_wallets (MOVES FUNDS) -------------------------------------


async def test_transfer_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/transfer": {"tranId": 13526853623}})
    _patch(monkeypatch, fake)

    result = await binance_transfer_between_wallets(
        TransferBetweenWalletsInput(type=TransferType.MAIN_FUNDING, asset="usdt", amount="25.50000000")
    )

    assert result.startswith("# Transfer accepted")
    assert "**tranId**: 13526853623" in result
    assert "no status and no resulting" in result
    assert "MAIN_FUNDING" in result
    # The confirmation echoes the amount VERBATIM — the exact string that was signed,
    # not a normalised rendering of it.
    assert "transfer of **25.50000000 USDT**" in result
    assert "25.5 USDT" not in result
    method, path, kwargs = fake.calls[0]
    assert (method, path) == ("POST", "/sapi/v1/asset/transfer")
    assert kwargs["auth"] == "signed"
    assert kwargs["params"] == {"type": "MAIN_FUNDING", "asset": "USDT", "amount": "25.50000000"}


async def test_transfer_amount_string_travels_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trailing zeros and long decimals must survive untouched — amounts are exact."""
    fake = _FakeClient(routes={"/sapi/v1/asset/transfer": {"tranId": 1}})
    _patch(monkeypatch, fake)

    await binance_transfer_between_wallets(
        TransferBetweenWalletsInput(type=TransferType.FUNDING_MAIN, asset="BNB", amount="0.10000000")
    )

    assert fake.calls[0][2]["params"]["amount"] == "0.10000000"
    assert isinstance(fake.calls[0][2]["params"]["amount"], str)


async def test_transfer_isolated_margin_legs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/transfer": {"tranId": 7}})
    _patch(monkeypatch, fake)

    result = await binance_transfer_between_wallets(
        TransferBetweenWalletsInput(
            type=TransferType.ISOLATEDMARGIN_ISOLATEDMARGIN,
            asset="USDT",
            amount="100",
            from_symbol="btcusdt",
            to_symbol="ethusdt",
        )
    )

    assert "**fromSymbol**: BTCUSDT" in result
    assert "**toSymbol**: ETHUSDT" in result
    assert fake.calls[0][2]["params"] == {
        "type": "ISOLATEDMARGIN_ISOLATEDMARGIN",
        "asset": "USDT",
        "amount": "100",
        "fromSymbol": "BTCUSDT",
        "toSymbol": "ETHUSDT",
    }


@pytest.mark.parametrize(
    ("transfer_type", "kwargs", "missing"),
    [
        (TransferType.ISOLATEDMARGIN_MARGIN, {}, "from_symbol"),
        (TransferType.ISOLATEDMARGIN_ISOLATEDMARGIN, {"to_symbol": "BTCUSDT"}, "from_symbol"),
        (TransferType.MARGIN_ISOLATEDMARGIN, {}, "to_symbol"),
        (TransferType.ISOLATEDMARGIN_ISOLATEDMARGIN, {"from_symbol": "BTCUSDT"}, "to_symbol"),
    ],
)
async def test_transfer_rejects_missing_isolated_margin_leg(
    monkeypatch: pytest.MonkeyPatch, transfer_type: TransferType, kwargs: dict[str, str], missing: str
) -> None:
    """A missing fromSymbol/toSymbol dies locally: nothing is signed or sent."""
    fake = _FakeClient(routes={"/sapi/v1/asset/transfer": {"tranId": 1}})
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError) as excinfo:
        TransferBetweenWalletsInput(type=transfer_type, asset="USDT", amount="10", **kwargs)

    assert missing in str(excinfo.value)
    assert fake.calls == []


@pytest.mark.parametrize("amount", ["0", "0.00", "-1", "abc", "nan"])
async def test_transfer_rejects_non_positive_amount(monkeypatch: pytest.MonkeyPatch, amount: str) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/transfer": {"tranId": 1}})
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError):
        TransferBetweenWalletsInput(type=TransferType.MAIN_FUNDING, asset="USDT", amount=amount)

    assert fake.calls == []


@pytest.mark.parametrize("amount", ["1E+2", "1e2", "+5", "Infinity", "1_000", "1.5.2", ".5", "1,5"])
async def test_transfer_rejects_non_plain_decimal_amount(monkeypatch: pytest.MonkeyPatch, amount: str) -> None:
    """Only `123` / `123.45` shapes may be signed.

    `Decimal('1E+2')` is a perfectly valid 100, which is exactly the problem: it is a
    hundred times what an operator reading '1E+2' in a confirmation might assume, and
    what Binance does with the literal is not something to find out on a live transfer.
    """
    fake = _FakeClient(routes={"/sapi/v1/asset/transfer": {"tranId": 1}})
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError) as excinfo:
        TransferBetweenWalletsInput(type=TransferType.MAIN_FUNDING, asset="USDT", amount=amount)

    assert "plain decimal string" in str(excinfo.value)
    assert fake.calls == []


async def test_transfer_kill_switch_message_is_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client's kill-switch surfaces as `Error: … trading is disabled …`, unaltered."""
    message = (
        "POST /sapi/v1/asset/transfer would move funds or change account state, but trading is disabled. "
        "Set BINANCE_ALLOW_TRADING=1 to enable order placement/cancellation, transfers, convert and "
        "algo orders. Dry-run validation (POST /api/v3/order/test) works without it."
    )
    fake = _FakeClient(routes={"/sapi/v1/asset/transfer": TradingDisabledError(message)})
    _patch(monkeypatch, fake)

    result = await binance_transfer_between_wallets(
        TransferBetweenWalletsInput(type=TransferType.MAIN_FUNDING, asset="USDT", amount="1")
    )

    assert result == f"Error: {message}"
    assert "trading is disabled" in result


async def test_transfer_error_path_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/transfer": _status_error(
                "POST",
                "/sapi/v1/asset/transfer",
                401,
                {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."},
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_transfer_between_wallets(
        TransferBetweenWalletsInput(type=TransferType.MAIN_FUNDING, asset="USDT", amount="1")
    )

    assert result.startswith("Error (401)")
    assert "(code -2015)" in result


async def test_transfer_type_enum_has_the_31_documented_values() -> None:
    """Inventory E lists exactly 31 transfer directions (S3)."""
    assert len(list(TransferType)) == 31
    assert TransferType.MAIN_FUNDING.value == "MAIN_FUNDING"
    assert TransferType.PORTFOLIO_MARGIN_MAIN.value == "PORTFOLIO_MARGIN_MAIN"


# -- binance_convert_dust_to_bnb (MOVES FUNDS, irreversible) ----------------------------


DUST_RESULT: dict[str, Any] = {
    "totalServiceCharge": "0.02102542",
    "totalTransfered": "1.05127099",
    "transferResult": [
        {
            "tranId": 2970932918,
            "serviceChargeAmount": "0.00500000",
            "amount": "0.1",
            "operateTime": 1563368549307,
            "transferedAmount": "0.25000000",
            "fromAsset": "ADA",
        },
        {
            "tranId": 2970932919,
            "serviceChargeAmount": "0.01602542",
            "amount": "0.2",
            "operateTime": 1563368549307,
            "transferedAmount": "0.80127099",
            "fromAsset": "DOT",
        },
    ],
}


async def test_convert_dust_happy_path_sends_repeated_asset_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/dust": DUST_RESULT})
    _patch(monkeypatch, fake)

    result = await binance_convert_dust_to_bnb(ConvertDustInput(assets=["btc", "eth"], account_type="SPOT"))

    assert result.startswith("# Dust converted to BNB")
    assert "This cannot be undone." in result
    assert "BNB received (totalTransfered)**: 1.05127099" in result
    assert "service charge (totalServiceCharge)**: 0.02102542" in result
    assert "| ADA | 0.1 | 0.25 | 0.005 |" in result
    assert "| DOT |" in result
    assert "binance_get_dust_log" in result

    method, path, kwargs = fake.calls[0]
    assert (method, path) == ("POST", "/sapi/v1/asset/dust")
    assert kwargs["auth"] == "signed"
    sent = kwargs["params"]
    assert set(sent) == {"asset", "accountType"}
    assert sent["accountType"] == "SPOT"
    # The asset list must be the client's Repeat marker, not a plain list: a plain list
    # would be JSON-encoded (`asset=["BTC","ETH"]`) and Binance rejects that here.
    assert isinstance(sent["asset"], Repeat)
    assert list(sent["asset"]) == ["BTC", "ETH"]
    assert build_query(sent) == "asset=BTC&asset=ETH&accountType=SPOT"


async def test_convert_dust_without_account_type_drops_it(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/dust": DUST_RESULT})
    _patch(monkeypatch, fake)

    await binance_convert_dust_to_bnb(ConvertDustInput(assets=["ADA"]))

    sent = fake.calls[0][2]["params"]
    assert set(sent) == {"asset"}
    assert build_query(sent) == "asset=ADA"


async def test_convert_dust_requires_at_least_one_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/dust": DUST_RESULT})
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError):
        ConvertDustInput(assets=[])

    assert fake.calls == []


async def test_convert_dust_caps_the_asset_list_at_a_hundred(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 100-asset cap is a client-side guard on an irreversible call, not a Binance limit."""
    fake = _FakeClient(routes={"/sapi/v1/asset/dust": DUST_RESULT})
    _patch(monkeypatch, fake)

    at_the_cap = ConvertDustInput(assets=[f"A{i:03d}" for i in range(100)])
    assert len(at_the_cap.assets) == 100

    with pytest.raises(ValidationError):
        ConvertDustInput(assets=[f"A{i:03d}" for i in range(101)])

    assert fake.calls == []


async def test_convert_dust_rejects_a_trading_pair_as_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(routes={"/sapi/v1/asset/dust": DUST_RESULT})
    _patch(monkeypatch, fake)

    with pytest.raises(ValidationError) as excinfo:
        ConvertDustInput(assets=["BTC-USDT"])

    assert "bare asset code" in str(excinfo.value)
    assert fake.calls == []


async def test_convert_dust_with_no_transfer_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never claim more than the response says: no rows means no rows."""
    fake = _FakeClient(
        routes={"/sapi/v1/asset/dust": {"totalServiceCharge": "0", "totalTransfered": "0", "transferResult": []}}
    )
    _patch(monkeypatch, fake)

    result = await binance_convert_dust_to_bnb(ConvertDustInput(assets=["ADA"]))

    assert "_Binance returned no per-asset results for this conversion._" in result


async def test_convert_dust_kill_switch_message_is_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client's kill-switch surfaces as `Error: … trading is disabled …`, unaltered."""
    message = (
        "POST /sapi/v1/asset/dust would move funds or change account state, but trading is disabled. "
        "Set BINANCE_ALLOW_TRADING=1 to enable order placement/cancellation, transfers, convert and "
        "algo orders. Dry-run validation (POST /api/v3/order/test) works without it."
    )
    fake = _FakeClient(routes={"/sapi/v1/asset/dust": TradingDisabledError(message)})
    _patch(monkeypatch, fake)

    result = await binance_convert_dust_to_bnb(ConvertDustInput(assets=["ADA"]))

    assert result == f"Error: {message}"
    assert "trading is disabled" in result


async def test_convert_dust_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        routes={
            "/sapi/v1/asset/dust": _status_error(
                "POST", "/sapi/v1/asset/dust", 400, {"code": -1100, "msg": "The asset BTC has no dust balance."}
            )
        }
    )
    _patch(monkeypatch, fake)

    result = await binance_convert_dust_to_bnb(ConvertDustInput(assets=["BTC"]))

    assert result.startswith("Error (400)")
    assert "no dust balance" in result


# -- envelope guard ---------------------------------------------------------------------


async def test_sapi_code_envelope_is_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 200 body carrying `{code, msg}` (no `success` key) is an error, not empty data."""
    fake = _FakeClient(routes={"/sapi/v1/asset/wallet/balance": {"code": -9000, "msg": "internal failure"}})
    _patch(monkeypatch, fake)

    result = await binance_get_wallet_balances(WalletBalancesInput())

    assert result == "Error: internal failure (code -9000)"


# -- live smoke (READ tools only) --------------------------------------------------------


@pytest.mark.live
async def test_live_smoke_read_tools() -> None:
    """Smoke-test the READ tools against the real configured account.

    Read-only by construction. The spot testnet serves no `/sapi` endpoints at all, so
    there is nowhere but the real account to exercise these — which is also why the two
    fund-moving tools in this module (`binance_transfer_between_wallets`,
    `binance_convert_dust_to_bnb`) have no live or trading smoke test and never will.
    Requires BINANCE_API_KEY + secret/PEM (see the conftest's live-marker gating).
    """
    results = [
        await binance_get_funding_wallet(FundingWalletInput(need_btc_valuation=True)),
        await binance_get_user_assets(UserAssetsInput(need_btc_valuation=True)),
        await binance_get_wallet_balances(WalletBalancesInput()),
        await binance_get_transfer_history(TransferHistoryInput(type=TransferType.MAIN_FUNDING)),
        await binance_get_dust_log(DustLogInput()),
        await binance_get_dust_convertible(DustConvertibleInput()),
        await binance_get_asset_detail(AssetDetailInput(asset="BTC")),
        await binance_get_trade_fees(TradeFeesInput(symbol="BTCUSDT")),
        await binance_get_asset_dividends(AssetDividendsInput(limit=5)),
    ]
    for result in results:
        assert isinstance(result, str)
        assert result
        assert not result.startswith("Error")
