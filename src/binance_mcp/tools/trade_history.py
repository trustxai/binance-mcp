"""Your own executed trades — `GET /api/v3/myTrades` plus the "everything I ever traded" walk.

Binance has **no endpoint that returns every trade across every symbol**: `myTrades`
takes a mandatory `symbol`. Binance staff answer the question twice on their own forum
by pointing at the user-data websocket stream, or at inferring the pairs from the
account's assets — there is no REST shortcut:

- https://dev.binance.vision/t/get-the-list-of-all-trades-from-binance-api-without-passing-the-trade-pair-symbol-to-api-v3-mytrades/4810
- https://dev.binance.vision/t/api-endpoint-to-retrieve-all-traded-symbols/4329

So this module reconstructs the answer in three steps:

1. `binance_get_my_trades` — one symbol, one page (the raw endpoint).
2. `binance_discover_traded_symbols` — guess the *candidate* symbol set from the assets
   the account has ever held (spot balances + funding/user assets + dust conversions),
   crossed with every `exchangeInfo` symbol. Cheap (~46 IP weight) and far narrower than
   the 3707 listed symbols (1370 of them TRADING).
3. `binance_get_all_my_trades` — walk each candidate symbol by `fromId`, under an
   explicit weight budget, and hand back a `cursor` so the next run is incremental.

The known blind spot of step 2: an asset that was bought and then fully sold inside
spot, with no deposit, withdrawal or dust trace, leaves nothing to discover from. Pass
it in `extra_assets` (or the pair in `symbols`) when you know it happened.

All `/api/v3` calls here are `auth="signed"` (USER_DATA); `exchangeInfo` is public.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from re import compile as re_compile
from typing import Any

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from binance_mcp.client import get_client
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import ResponseFormat, clip_response, epoch_to_human, fmt_num, to_json
from binance_mcp.server import mcp

# -- context-window guards -----------------------------------------------------------
MAX_TRADES_DISPLAY = 100
MAX_ALL_TRADES_DISPLAY = 200
MAX_DISCOVERED_SYMBOLS_DISPLAY = 400

# -- endpoint constants --------------------------------------------------------------
# GET /api/v3/myTrades: IP weight 20 (5 when `orderId` is supplied) — S2, inventory B.
MY_TRADES_WEIGHT = 20
MY_TRADES_MAX_LIMIT = 1000
# The `fromId` walk pages at the endpoint's maximum; a short page means "symbol done".
WALK_PAGE_LIMIT = 1000
# startTime..endTime on myTrades is capped at 24 h by Binance (-1127 otherwise).
MY_TRADES_WINDOW_MS = 24 * 60 * 60 * 1000
# Default ceiling for one walk: 150 symbols' worth of calls, a quarter of the 6000/min
# IP budget. `weight_ceiling` is the second brake, read from the response headers.
DEFAULT_MAX_WEIGHT = 3000
DEFAULT_WEIGHT_CEILING = 5000
# The quote assets a retail spot account realistically traded against.
DEFAULT_QUOTE_ASSETS: tuple[str, ...] = ("USDT", "USDC", "FDUSD", "BUSD", "BTC", "ETH", "BNB", "EUR", "TUSD")

_SYMBOL_PATTERN = re_compile(r"^[A-Z0-9]{2,20}$")
# Bare asset codes: Binance really does list one-letter assets (`W`, `S`).
_ASSET_PATTERN = re_compile(r"^[A-Z0-9]{1,20}$")

# `exchangeInfo` is 3707 symbols and IP weight 20 — fetched once per process and kept
# here (symbol -> the exchangeInfo entry). Tests clear it between cases.
_SYMBOL_CACHE: dict[str, dict[str, Any]] = {}


# -- shared helpers ------------------------------------------------------------------


def _normalize_symbol(value: str) -> str:
    """Uppercase and validate a Binance symbol (e.g. `btcusdt` -> `BTCUSDT`)."""
    upper = value.strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(upper):
        raise ValueError(f"Invalid symbol {value!r} — expected 2-20 uppercase letters/digits (e.g. BTCUSDT).")
    return upper


def _normalize_asset(value: str) -> str:
    """Uppercase and validate a bare asset code (`btc` -> `BTC`; one-letter assets exist)."""
    upper = value.strip().upper()
    if not _ASSET_PATTERN.fullmatch(upper):
        raise ValueError(f"Invalid asset {value!r} — expected 1-20 uppercase letters/digits (e.g. BTC).")
    return upper


def _to_ms(value: Any, field_name: str = "timestamp") -> int | None:
    """Accept an epoch-ms int or an ISO-8601 string on the tool surface; always send ms.

    A digits-only string is treated as an epoch in milliseconds only when it has 12+
    digits (2001-09-09 onward); shorter digit strings are ambiguous (seconds? a year?)
    and are rejected rather than guessed at.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be epoch ms or ISO-8601 — got {value!r}.")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() and len(text) >= 12:
            return int(text)
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be epoch ms or ISO-8601 — got {value!r} "
                "(e.g. 1700000000000 or 2024-01-01T00:00:00Z)."
            ) from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    raise ValueError(f"{field_name} must be epoch ms or ISO-8601 — got {value!r}.")


def _dec(value: Any) -> Decimal:
    """Parse a Binance decimal string; a missing/garbage value contributes nothing."""
    if value in (None, ""):
        return Decimal(0)
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return Decimal(0)


@dataclass
class _Totals:
    """Per-symbol (or per-call) trade totals, accumulated as exact decimals."""

    trades: int = 0
    bought_qty: Decimal = Decimal(0)
    bought_quote: Decimal = Decimal(0)
    sold_qty: Decimal = Decimal(0)
    sold_quote: Decimal = Decimal(0)
    fees: dict[str, Decimal] = field(default_factory=dict)

    def add(self, trade: dict[str, Any]) -> None:
        """Fold one `myTrades` row in. For MY trades `isBuyer` is MY side of the fill."""
        self.trades += 1
        qty, quote = _dec(trade.get("qty")), _dec(trade.get("quoteQty"))
        if trade.get("isBuyer"):
            self.bought_qty += qty
            self.bought_quote += quote
        else:
            self.sold_qty += qty
            self.sold_quote += quote
        commission, asset = _dec(trade.get("commission")), str(trade.get("commissionAsset") or "")
        if asset and commission:
            self.fees[asset] = self.fees.get(asset, Decimal(0)) + commission


def _fees_text(fees: dict[str, Decimal]) -> str:
    """ "0.001 BNB, 0.5 USDT" — fees are charged in whichever asset Binance picked."""
    if not fees:
        return "none"
    return ", ".join(f"{fmt_num(amount)} {asset}" for asset, amount in sorted(fees.items()))


def _totals_lines(totals: _Totals) -> list[str]:
    """Markdown bullets for one `_Totals` block."""
    return [
        f"- bought: {fmt_num(totals.bought_qty)} base for {fmt_num(totals.bought_quote)} quote",
        f"- sold: {fmt_num(totals.sold_qty)} base for {fmt_num(totals.sold_quote)} quote",
        f"- fees: {_fees_text(totals.fees)}",
    ]


def _totals_json(totals: _Totals) -> dict[str, Any]:
    """The same block for `response_format=json` (decimals rendered as strings)."""
    return {
        "trades": totals.trades,
        "bought_qty": fmt_num(totals.bought_qty),
        "bought_quote": fmt_num(totals.bought_quote),
        "sold_qty": fmt_num(totals.sold_qty),
        "sold_quote": fmt_num(totals.sold_quote),
        "fees": {asset: fmt_num(amount) for asset, amount in sorted(totals.fees.items())},
    }


def _trade_row(trade: dict[str, Any], *, with_symbol: bool) -> str:
    """One markdown table row. `isBuyer` is MY side (unlike public trades' isBuyerMaker)."""
    side = "buy" if trade.get("isBuyer") else "sell"
    role = "maker" if trade.get("isMaker") else "taker"
    prefix = f"| {epoch_to_human(trade.get('time'))} |"
    if with_symbol:
        prefix += f" {trade.get('symbol')} |"
    return (
        f"{prefix} {side} | {fmt_num(trade.get('price'))} | {fmt_num(trade.get('qty'))} | "
        f"{fmt_num(trade.get('quoteQty'))} | {fmt_num(trade.get('commission'))} "
        f"{trade.get('commissionAsset') or ''} | {role} |"
    )


async def _load_exchange_symbols(client: Any) -> dict[str, dict[str, Any]]:
    """`GET /api/v3/exchangeInfo` once per process (IP weight 20), cached module-level.

    3707 symbols is far too much to dump at an LLM and far too expensive to re-fetch per
    tool call, but the base/quote mapping is exactly what symbol discovery needs.
    """
    if not _SYMBOL_CACHE:
        resp = await client.request("GET", "/api/v3/exchangeInfo", params={}, auth="none")
        for entry in resp.json().get("symbols", []):
            symbol = entry.get("symbol")
            if symbol:
                _SYMBOL_CACHE[str(symbol)] = entry
    return _SYMBOL_CACHE


# -- binance_get_my_trades -----------------------------------------------------------


class MyTradesInput(BaseModel):
    """Input for `binance_get_my_trades`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str = Field(description="Trading pair symbol, e.g. BTCUSDT. Mandatory — Binance has no all-symbol form.")
    order_id: int | None = Field(
        default=None, ge=0, description="Only fills of this order. Drops the endpoint's weight from 20 to 5."
    )
    from_id: int | None = Field(
        default=None, ge=0, description="Return trades with id >= this one. Cannot combine with the time window."
    )
    start_time: int | str | None = Field(
        default=None,
        description="Window start, epoch ms or ISO-8601. With `end_time` the span must be <= 24 h.",
    )
    end_time: int | str | None = Field(
        default=None,
        description="Window end, epoch ms or ISO-8601. With `start_time` the span must be <= 24 h.",
    )
    limit: int = Field(
        default=500, ge=1, le=MY_TRADES_MAX_LIMIT, description="Trades per page (Binance max 1000, default 500)."
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        return _normalize_symbol(v)

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _validate_times(cls, v: Any, info: ValidationInfo) -> int | None:
        return _to_ms(v, info.field_name or "timestamp")


def _check_my_trades_combo(params: MyTradesInput) -> str | None:
    """Reject the parameter combinations Binance does not accept, before spending a call.

    Inventory B lists the allowed sets exactly: symbol; +orderId; +fromId; +startTime;
    +endTime; +startTime+endTime; +orderId+fromId. Anything else is -1128, and the 24 h
    span cap is -1127 — both are cheaper to catch here.
    """
    has_window = params.start_time is not None or params.end_time is not None
    if params.order_id is not None and has_window:
        return (
            "Error: `order_id` cannot combine with `start_time`/`end_time` on myTrades. Binance accepts only: "
            "symbol; symbol+order_id; symbol+from_id; symbol+start_time[+end_time]; symbol+order_id+from_id."
        )
    if params.from_id is not None and has_window:
        return (
            "Error: `from_id` cannot combine with `start_time`/`end_time` on myTrades. Use `from_id` to walk by "
            "trade id, or the time window on its own."
        )
    if params.start_time is not None and params.end_time is not None:
        # The field is annotated `int | str | None` so the LLM sees both forms in the
        # schema; the mode="before" validator has already normalised it to epoch ms.
        span = int(params.end_time) - int(params.start_time)
        if span < 0:
            return "Error: `end_time` is before `start_time`."
        if span > MY_TRADES_WINDOW_MS:
            hours = span / 3_600_000
            return (
                f"Error: myTrades caps start_time..end_time at 24 hours; the requested window spans {hours:.1f} h. "
                "Slice it into <= 24 h calls, or use `binance_get_all_my_trades` which walks by trade id instead."
            )
    return None


@mcp.tool(
    name="binance_get_my_trades",
    annotations=ToolAnnotations(
        title="Binance My Trades",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_my_trades(params: MyTradesInput) -> str:
    """Fetch YOUR executed trades (fills) for one symbol.

    Calls `GET /api/v3/myTrades` (signed; IP weight **20**, or 5 when `order_id` is
    given). `symbol` is mandatory — Binance has no endpoint that returns trades across
    every symbol, which is what `binance_get_all_my_trades` exists to work around.

    Unlike the public trade endpoints, `isBuyer` here is **your own side** of the fill
    (public trades expose `isBuyerMaker`, the aggressor's side, instead).

    When to Use:
    - To see the fills of one pair, or of one order (`order_id`).
    - To check the exact price, fee and fee asset of a known trade.

    When NOT to Use:
    - For the whole account's history — use `binance_get_all_my_trades`.
    - For orders that never filled — use `binance_get_all_orders` (spot_orders).
    - For anonymous market trades — use `binance_get_recent_trades` (market_data).

    Returns:
    Markdown: a table of up to 100 fills (time, side, price, qty, quote qty, commission,
    maker/taker) plus totals for the page — bought/sold base and quote, and fees broken
    down by fee asset. JSON: the full page plus the same totals.

    Pagination / Windows:
    Binance accepts only these combinations: `symbol`; `symbol`+`order_id`;
    `symbol`+`from_id`; `symbol`+`start_time`[+`end_time`]; `symbol`+`order_id`+`from_id`.
    `start_time`..`end_time` must span **at most 24 hours** — a wider window is rejected
    here, with no call spent. To page, re-call with `from_id` = the last id + 1.

    Examples:
    params = {"symbol": "BTCUSDT", "limit": 100}
    params = {"symbol": "BTCUSDT", "order_id": 987654321}
    params = {"symbol": "ETHUSDT", "start_time": "2024-01-01T00:00:00Z", "end_time": "2024-01-01T23:59:59Z"}
    params = {"symbol": "ETHUSDT", "from_id": 4211999}

    Error Handling:
    An over-wide window or an illegal combination is refused locally with an `Error: …`
    string and no API call. -2015 means the key lacks Reading or the IP is not
    allowlisted; -1121 means the symbol does not exist.
    """
    try:
        if (rejection := _check_my_trades_combo(params)) is not None:
            return rejection
        client = get_client()
        query: dict[str, Any] = {"symbol": params.symbol, "limit": params.limit}
        if params.order_id is not None:
            query["orderId"] = params.order_id
        if params.from_id is not None:
            query["fromId"] = params.from_id
        if params.start_time is not None:
            query["startTime"] = params.start_time
        if params.end_time is not None:
            query["endTime"] = params.end_time
        resp = await client.request("GET", "/api/v3/myTrades", params=query, auth="signed")
        trades: list[dict[str, Any]] = resp.json()

        totals = _Totals()
        for trade in trades:
            totals.add(trade)

        if params.response_format is ResponseFormat.JSON:
            return clip_response(
                to_json(
                    {
                        "symbol": params.symbol,
                        "count": len(trades),
                        "totals": _totals_json(totals),
                        "trades": trades,
                    }
                )
            )

        shown = trades[:MAX_TRADES_DISPLAY]
        lines = [
            f"# My trades — {params.symbol}",
            "",
            f"Showing {len(shown)} of {len(trades)} fill(s).",
            "",
            "| time | side | price | qty | quote qty | commission | role |",
            "|---|---|---|---|---|---|---|",
        ]
        lines.extend(_trade_row(trade, with_symbol=False) for trade in shown)
        if not trades:
            lines.append("_No fills for this symbol with those filters._")
        elif len(trades) > MAX_TRADES_DISPLAY:
            lines.append("")
            lines.append(f"_...{len(trades) - MAX_TRADES_DISPLAY} more not shown — use response_format=json._")
        if trades:
            lines.extend(["", "## Totals for this page", *_totals_lines(totals)])
            last_id = trades[-1].get("id")
            if len(trades) == params.limit and last_id is not None:
                lines.append(f"- page is full — continue with `from_id`: **{int(last_id) + 1}**")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


# -- binance_discover_traded_symbols -------------------------------------------------


class DiscoverTradedSymbolsInput(BaseModel):
    """Input for `binance_discover_traded_symbols`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    quote_assets: list[str] = Field(
        default_factory=lambda: list(DEFAULT_QUOTE_ASSETS),
        description="Quote assets to pair candidates against. Defaults to the nine a retail spot account uses.",
    )
    extra_assets: list[str] = Field(
        default_factory=list,
        description="Assets you know you traded but that hold no balance and left no dust trace (e.g. ['SOL']).",
    )
    include_break: bool = Field(
        default=False,
        description="Include non-TRADING symbols (BREAK/HALT — delisted pairs). Roughly triples the symbol count.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("quote_assets", "extra_assets")
    @classmethod
    def _validate_assets(cls, v: list[str]) -> list[str]:
        return [_normalize_asset(item) for item in v]


async def _discover_candidate_assets(client: Any, extra_assets: list[str]) -> tuple[set[str], dict[str, int]]:
    """Union the assets the account has ever *visibly* held, and say where each came from.

    Three cheap signed reads (IP weight 20 + 5 + 1): current spot balances, the funding /
    user-asset list, and the dust-conversion log (which is the only trace left by an asset
    that was swept to BNB). `getUserAsset` is a POST that only reads — it is on the
    client's POST-read allowlist, so the trading kill-switch does not block it.
    """
    sources: dict[str, int] = {}

    account = await client.request("GET", "/api/v3/account", params={"omitZeroBalances": True}, auth="signed")
    balances = [str(b.get("asset")) for b in account.json().get("balances", []) if b.get("asset")]
    sources["spot balances"] = len(set(balances))

    user_assets = await client.request("POST", "/sapi/v3/asset/getUserAsset", params={}, auth="signed")
    funding = [str(a.get("asset")) for a in user_assets.json() or [] if a.get("asset")]
    sources["user assets"] = len(set(funding))

    dribblet = await client.request("GET", "/sapi/v1/asset/dribblet", params={}, auth="signed")
    dusted: list[str] = []
    for batch in dribblet.json().get("userAssetDribblets", []) or []:
        for detail in batch.get("userAssetDribbletDetails", []) or []:
            if detail.get("fromAsset"):
                dusted.append(str(detail["fromAsset"]))
    sources["dust conversions"] = len(set(dusted))
    sources["extra_assets"] = len(set(extra_assets))

    candidates = {asset.upper() for asset in [*balances, *funding, *dusted, *extra_assets] if asset}
    return candidates, sources


def _select_symbols(
    symbols: dict[str, dict[str, Any]], candidates: set[str], quote_assets: list[str], include_break: bool
) -> list[str]:
    """Every listed symbol whose base OR quote is a candidate asset, restricted to the
    whitelisted quote assets and (by default) to symbols still TRADING."""
    quotes = set(quote_assets)
    selected = [
        name
        for name, entry in symbols.items()
        if str(entry.get("quoteAsset", "")) in quotes
        and (str(entry.get("baseAsset", "")) in candidates or str(entry.get("quoteAsset", "")) in candidates)
        and (include_break or str(entry.get("status")) == "TRADING")
    ]
    return sorted(selected)


@mcp.tool(
    name="binance_discover_traded_symbols",
    annotations=ToolAnnotations(
        title="Binance Discover Traded Symbols",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_discover_traded_symbols(params: DiscoverTradedSymbolsInput) -> str:
    """Work out which symbols this account plausibly traded — Binance will not tell you.

    There is **no endpoint that lists an account's traded pairs**. Binance staff answered
    the question twice on their own developer forum, and both answers are workarounds:
    keep a local record from the user-data websocket stream, or infer the pairs from the
    account's assets (dev.binance.vision threads 4810 and 4329). This tool is the second
    answer, made cheap.

    It unions the assets the account visibly holds or held — `GET /api/v3/account`
    (weight 20, zero balances omitted), `POST /sapi/v3/asset/getUserAsset` (weight 5, the
    funding/spot asset list) and `GET /sapi/v1/asset/dribblet` (weight 1, the last 100
    dust conversions) — then crosses them with every `exchangeInfo` symbol (weight 20,
    fetched once per process and cached) whose **base OR quote** is a candidate asset.
    Total discovery cost: ~46 IP weight of the 6000/min budget.

    Blind spot, stated plainly: an asset bought and then fully sold within spot, never
    deposited, withdrawn or dust-converted, leaves no trace to discover. Name it in
    `extra_assets`.

    Cost trap, equally plainly: the match is base **or** quote, so a candidate asset that
    is also a quote asset — USDT is, and almost every account holds some — selects every
    pair quoted in it (~490 for USDT). Read the estimated weight before walking, and
    narrow `quote_assets` when it is larger than you want to spend.

    When to Use:
    - Before `binance_get_all_my_trades`, to see (and prune) the symbol list and its cost.
    - To answer "which pairs have I ever traded?" without spending 74,000 weight.

    When NOT to Use:
    - When you already know the pairs — pass them to `binance_get_all_my_trades` directly.
    - To read balances — use `binance_get_spot_account` / `binance_get_user_assets`.

    Returns:
    Markdown: the candidate assets with their source counts, the sorted symbol list, and
    the estimated cost of walking it (20 IP weight per symbol). JSON: the same, as
    `{"assets": [...], "symbols": [...], "estimated_weight": N}`.

    Examples:
    params = {}
    params = {"extra_assets": ["SOL", "ADA"], "quote_assets": ["USDT", "BTC"]}
    params = {"include_break": true}

    Error Handling:
    /sapi endpoints do not exist on the spot testnet (404). -2015 means the key lacks
    Reading or the IP is not allowlisted.
    """
    try:
        client = get_client()
        candidates, sources = await _discover_candidate_assets(client, params.extra_assets)
        symbols = await _load_exchange_symbols(client)
        selected = _select_symbols(symbols, candidates, params.quote_assets, params.include_break)
        estimated_weight = MY_TRADES_WEIGHT * len(selected)
        assets = sorted(candidates)

        if params.response_format is ResponseFormat.JSON:
            return clip_response(
                to_json(
                    {
                        "assets": assets,
                        "asset_sources": sources,
                        "quote_assets": params.quote_assets,
                        "include_break": params.include_break,
                        "symbols": selected,
                        "symbol_count": len(selected),
                        "estimated_weight": estimated_weight,
                    }
                )
            )

        shown = selected[:MAX_DISCOVERED_SYMBOLS_DISPLAY]
        lines = [
            "# Traded-symbol candidates",
            "",
            f"**{len(assets)}** candidate asset(s) → **{len(selected)}** symbol(s) "
            f"(of {len(symbols):,} listed), quoted in {', '.join(params.quote_assets)}"
            f"{'' if params.include_break else ', TRADING only'}.",
            "",
            "- sources: " + ", ".join(f"{name} {count}" for name, count in sources.items()),
            f"- assets: {', '.join(assets) if assets else '_none_'}",
            f"- estimated cost of a full walk: **{estimated_weight:,}** IP weight "
            f"({MY_TRADES_WEIGHT} per symbol, 6000/min budget)",
            "",
            "## Symbols",
            "",
            ", ".join(shown) if shown else "_No candidate symbols — the account holds nothing discoverable._",
        ]
        if len(selected) > MAX_DISCOVERED_SYMBOLS_DISPLAY:
            lines.append("")
            lines.append(
                f"_...{len(selected) - MAX_DISCOVERED_SYMBOLS_DISPLAY} more not shown — "
                "use response_format=json, or narrow `quote_assets`._"
            )
        if not params.include_break:
            lines.append("")
            lines.append("_Delisted (BREAK/HALT) pairs are excluded; pass `include_break: true` to add them._")
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)


# -- binance_get_all_my_trades -------------------------------------------------------


class AllMyTradesInput(BaseModel):
    """Input for `binance_get_all_my_trades`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbols: list[str] | None = Field(
        default=None,
        description="Symbols to walk. Omit to discover them (same params as binance_discover_traded_symbols).",
    )
    cursor: dict[str, int] | None = Field(
        default=None,
        description="Per-symbol last trade id from a previous run ({'BTCUSDT': 4211999}) — resumes incrementally.",
    )
    quote_assets: list[str] = Field(
        default_factory=lambda: list(DEFAULT_QUOTE_ASSETS),
        description="Quote assets used by discovery when `symbols` is omitted.",
    )
    extra_assets: list[str] = Field(
        default_factory=list, description="Extra assets for discovery when `symbols` is omitted."
    )
    include_break: bool = Field(
        default=False, description="Let discovery include non-TRADING (delisted) symbols when `symbols` is omitted."
    )
    max_weight: int = Field(
        default=DEFAULT_MAX_WEIGHT,
        ge=MY_TRADES_WEIGHT,
        le=60_000,
        description="Weight this walk may spend (20 per page). Default 3000 = 150 pages, half a minute's budget.",
    )
    weight_ceiling: int = Field(
        default=DEFAULT_WEIGHT_CEILING,
        ge=1,
        le=6_000,
        description="Stop when Binance's reported used IP weight (1m) exceeds this. Default 5000 of the 6000 budget.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format.")

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, v: list[str] | None) -> list[str] | None:
        return None if v is None else [_normalize_symbol(item) for item in v]

    @field_validator("quote_assets", "extra_assets")
    @classmethod
    def _validate_assets(cls, v: list[str]) -> list[str]:
        return [_normalize_asset(item) for item in v]

    @field_validator("cursor")
    @classmethod
    def _validate_cursor(cls, v: dict[str, int] | None) -> dict[str, int] | None:
        if v is None:
            return None
        cleaned: dict[str, int] = {}
        for symbol, last_id in v.items():
            if last_id < 0:
                raise ValueError(f"Invalid cursor for {symbol!r} — a trade id cannot be negative.")
            cleaned[_normalize_symbol(symbol)] = last_id
        return cleaned


@dataclass
class _WalkResult:
    """What one multi-symbol `fromId` walk produced, finished or not."""

    trades: list[dict[str, Any]] = field(default_factory=list)
    cursor: dict[str, int] = field(default_factory=dict)
    calls: int = 0
    symbols_done: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    error: str | None = None


async def _walk_my_trades(
    client: Any, symbols: list[str], cursor: dict[str, int], max_weight: int, weight_ceiling: int
) -> _WalkResult:
    """Page every symbol by `fromId` until a short page, under two independent brakes.

    Brake 1 is the caller's `max_weight`: checked BEFORE every request, counting the call
    about to be made, so the reported call count is always the real one. Brake 2 is
    Binance's own `X-MBX-USED-WEIGHT-1M` as captured by the client after the previous
    call — it sees traffic this walk did not generate (other tools, other sessions).

    The cursor is seeded from the caller's so that a symbol the budget never reached does
    not lose the position a previous run established; symbols with no known position at
    all stay absent, and a walk starts them at `fromId=0`.
    """
    result = _WalkResult(cursor=dict(cursor))
    seen: set[tuple[str, int]] = set()

    try:
        for symbol in symbols:
            last_id = result.cursor.get(symbol)
            while True:
                if MY_TRADES_WEIGHT * result.calls + MY_TRADES_WEIGHT > max_weight:
                    result.stop_reason = (
                        f"weight budget exhausted ({MY_TRADES_WEIGHT * result.calls} of max_weight {max_weight} "
                        f"spent over {result.calls} call(s))"
                    )
                    return result
                used = client.last_used_weight_1m
                if used is not None and used > weight_ceiling:
                    result.stop_reason = (
                        f"Binance reports {used} used IP weight in the last minute, over weight_ceiling "
                        f"{weight_ceiling} — backing off"
                    )
                    return result

                from_id = 0 if last_id is None else last_id + 1
                # Counted BEFORE the await: a call that errors still cost its weight, and
                # the reported count must be the real one (walk-tool rule a).
                result.calls += 1
                resp = await client.request(
                    "GET",
                    "/api/v3/myTrades",
                    params={"symbol": symbol, "fromId": from_id, "limit": WALK_PAGE_LIMIT},
                    auth="signed",
                )
                page: list[dict[str, Any]] = resp.json()
                for trade in page:
                    trade_id = trade.get("id")
                    if trade_id is None:
                        continue
                    key = (symbol, int(trade_id))
                    if key in seen:
                        continue
                    seen.add(key)
                    result.trades.append(trade)
                    if last_id is None or int(trade_id) > last_id:
                        last_id = int(trade_id)
                if last_id is not None:
                    result.cursor[symbol] = last_id
                if len(page) < WALK_PAGE_LIMIT:
                    result.symbols_done.append(symbol)
                    break
    except Exception as exc:  # noqa: BLE001 — partial rows + cursor beat a bare error string
        result.error = handle_api_error(exc)
    return result


def _all_trades_markdown(result: _WalkResult, symbols: list[str], discovered: bool) -> str:
    """Render the walk: status, the trade table, per-symbol totals, and the next cursor."""
    remaining = [s for s in symbols if s not in result.symbols_done]
    lines = [
        "# All my trades",
        "",
        f"Walked **{len(result.symbols_done)}** of **{len(symbols)}** symbol(s) in {result.calls} call(s) "
        f"(~{MY_TRADES_WEIGHT * result.calls:,} IP weight)"
        f"{' — symbols discovered automatically' if discovered else ''}.",
        f"Collected **{len(result.trades):,}** fill(s).",
    ]
    if result.error is not None:
        lines += [
            "",
            f"⚠️ **Stopped on an error after {result.calls} call(s)** — the fills below are what was collected "
            f"before it: {result.error}",
            "Resume from the cursor at the bottom once the cause is fixed.",
        ]
    elif result.stop_reason is not None:
        lines += [
            "",
            f"⚠️ **Stopped early: {result.stop_reason}.** {len(remaining)} symbol(s) not finished "
            f"({', '.join(remaining[:20])}{'…' if len(remaining) > 20 else ''}).",
            "Resume with the `cursor` block at the bottom: re-call this tool with the same `symbols` and that "
            "`cursor` to pick up exactly where this run stopped.",
        ]

    ordered = sorted(result.trades, key=lambda t: (int(t.get("time") or 0), str(t.get("symbol"))))
    shown = ordered[:MAX_ALL_TRADES_DISPLAY]
    lines += [
        "",
        "| time | symbol | side | price | qty | quote qty | commission | role |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(_trade_row(trade, with_symbol=True) for trade in shown)
    if not ordered:
        lines.append("_No fills found for these symbols._")
    elif len(ordered) > MAX_ALL_TRADES_DISPLAY:
        lines += ["", f"_...{len(ordered) - MAX_ALL_TRADES_DISPLAY} more not shown — use response_format=json._"]

    per_symbol: dict[str, _Totals] = {}
    for trade in ordered:
        per_symbol.setdefault(str(trade.get("symbol")), _Totals()).add(trade)
    if per_symbol:
        lines += ["", "## Totals per symbol", ""]
        for symbol, totals in sorted(per_symbol.items()):
            lines.append(f"### {symbol} — {totals.trades} fill(s)")
            lines.extend(_totals_lines(totals))

    lines += [
        "",
        "## Cursor (paste back as `cursor` to continue where this run stopped)",
        "",
        "```json",
        to_json(result.cursor),
        "```",
    ]
    return clip_response("\n".join(lines))


@mcp.tool(
    name="binance_get_all_my_trades",
    annotations=ToolAnnotations(
        title="Binance All My Trades",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_all_my_trades(params: AllMyTradesInput) -> str:
    """Collect every fill across every symbol this account traded — the "all my trades" answer.

    Binance has no such endpoint (`myTrades` needs a `symbol`; staff confirm the gap on
    dev.binance.vision threads 4810 and 4329), so this tool does the only thing that
    works over REST: take a symbol list — yours, or one from
    `binance_discover_traded_symbols` — and walk each symbol by `fromId` at 1000 fills a
    page until a short page says that symbol is exhausted.

    Two brakes keep it inside the 6000/min IP budget. `max_weight` (default 3000 = 150
    pages) is checked **before** every request, and the walk also stops when Binance's own
    reported used weight passes `weight_ceiling` (default 5000). When either fires — or
    when a request errors mid-walk — the fills collected so far are returned **together
    with** a `cursor`, so nothing is lost and the next run resumes exactly there.

    Cost, so nobody is surprised: 20 IP weight per page per symbol. 150 symbols with no
    trades still costs 3000 weight. Walking all 1370 TRADING symbols would cost 27,400 —
    about five minutes of full budget — which is why discovery narrows the list first.

    When to Use:
    - "Show me every trade I have ever made", tax/portfolio reconstruction, a full export.
    - Incremental top-ups: keep the returned `cursor` and pass it back next time.

    When NOT to Use:
    - For one pair — `binance_get_my_trades` is one call.
    - For orders that never filled, deposits, withdrawals, converts or Pay/Card spending:
      those are different endpoints (spot_orders, wallet_capital, convert, pay).

    Returns:
    Markdown: the walk status (symbols finished, calls, weight), a time-sorted table of up
    to 200 fills, per-symbol totals (fills, bought/sold base and quote, fees by asset) and
    the next `cursor` as a JSON block to paste back. JSON: the same data with every fill.

    Pagination:
    `cursor` maps symbol → the last trade id already collected; the walk restarts each
    symbol at that id + 1, so re-running is cheap and never duplicates a fill. Symbols the
    budget never reached keep whatever position the cursor already held.

    Examples:
    params = {}
    params = {"symbols": ["BTCUSDT", "ETHUSDT"], "max_weight": 200}
    params = {"symbols": ["BTCUSDT"], "cursor": {"BTCUSDT": 4211999}}

    Error Handling:
    An error mid-walk never discards work: the partial fills plus the cursor come back
    alongside the error text. -2015 means the key lacks Reading or the IP is not
    allowlisted; a 429/418 means the budget was already spent elsewhere — lower
    `weight_ceiling` and wait a minute.
    """
    try:
        client = get_client()
        discovered = params.symbols is None
        if params.symbols is not None:
            symbols = params.symbols
        else:
            candidates, _sources = await _discover_candidate_assets(client, params.extra_assets)
            exchange_symbols = await _load_exchange_symbols(client)
            symbols = _select_symbols(exchange_symbols, candidates, params.quote_assets, params.include_break)
        if not symbols:
            return (
                "Error: no symbols to walk. Discovery found no candidate pair for this account — pass `symbols` "
                "explicitly, or add the assets you traded to `extra_assets`."
            )

        cursor = dict(params.cursor or {})
        result = await _walk_my_trades(client, symbols, cursor, params.max_weight, params.weight_ceiling)

        if params.response_format is ResponseFormat.JSON:
            ordered = sorted(result.trades, key=lambda t: (int(t.get("time") or 0), str(t.get("symbol"))))
            per_symbol: dict[str, _Totals] = {}
            for trade in ordered:
                per_symbol.setdefault(str(trade.get("symbol")), _Totals()).add(trade)
            return clip_response(
                to_json(
                    {
                        "symbols": symbols,
                        "symbols_done": result.symbols_done,
                        "discovered": discovered,
                        "calls": result.calls,
                        "estimated_weight": MY_TRADES_WEIGHT * result.calls,
                        "stopped_early": result.stop_reason,
                        "error": result.error,
                        "count": len(ordered),
                        "cursor": result.cursor,
                        "totals": {symbol: _totals_json(t) for symbol, t in sorted(per_symbol.items())},
                        "trades": ordered,
                    }
                )
            )
        return _all_trades_markdown(result, symbols, discovered)
    except Exception as exc:
        return handle_api_error(exc)
