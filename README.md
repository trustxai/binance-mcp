# Binance MCP Server (amazing-binance-mcp)

A [Model Context Protocol](https://modelcontextprotocol.io) server for the
[Binance Spot + Wallet REST API](https://developers.binance.com/docs/binance-spot-api-docs),
built on the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
(FastMCP, stdio transport). It authenticates with a plain **API key** — Ed25519 or HMAC,
**no OAuth** — so an LLM can read your deposits, withdrawals, trades, balances, fiat and
Pay history and live market data, and — only when you flip the kill-switch — place and
cancel spot orders, order lists, TWAP algo orders and conversions.

> The PyPI distribution and console script are both **`amazing-binance-mcp`** (the bare
> `binance-mcp` and `binance-mcp-server` names on PyPI belong to unrelated packages).
> Always run `uvx amazing-binance-mcp`.

## Safety model

Three rails, all enforced in the HTTP client — not in the tools — so no tool can forget them:

- **No withdrawals, ever.** There is no withdrawal tool, and the client refuses the
  withdrawal and fiat-rail endpoints under any configuration. Create the API key with
  "Enable Withdrawals" **off** and an IP allowlist as well; `binance_health_check` warns
  if the key can withdraw.
- **Trading is off by default.** Anything that moves funds or changes account state —
  orders, cancels, order lists, transfers between your wallets, convert, TWAP, dust
  conversion — returns `Error: … trading is disabled` until `BINANCE_ALLOW_TRADING=1`.
  The order dry-run (`binance_test_order`) always works. Those tools are marked 🔒 below.
- **Confirmations never claim more than Binance said.** A placed order echoes the
  status Binance returned (an `EXPIRED` FOK is reported as *not live*); a TWAP is
  *accepted, not executed*; a convert is *converted* only when the status is `SUCCESS`.
  A 5xx or timeout on a mutating call is reported as **execution UNKNOWN** with the tool
  to query before retrying — the server never retries a mutation on its own.

Everything else is read-only and works with a key that only has "Enable Reading".

## Features

- **Your whole history, since the account was created.** Deposits, withdrawals, fiat
  orders and payments, Binance Pay, convert trades and — per symbol, since Binance has
  no cross-symbol endpoint — every spot trade, via budgeted **walk tools** that slice the
  90-day / 30-day / 24-hour windows the API imposes and hand you a resume cursor when a
  budget runs out (see [History walks](#history-walks)).
- **Spot trading for algotrading.** Dry-run validation, market/limit/stop orders,
  cancel and cancel-replace, OCO / OTO / OTOCO lists, TWAP algo orders, convert quotes —
  numbers travel as strings exactly as you typed them, and every per-type mandatory
  parameter set is validated locally before anything is signed.
- **Market data without a key.** Exchange info with the filters an order must respect,
  order book, trades, klines, tickers.
- **Spot, funding and earn balances**, wallet-to-wallet transfers, dust, fees, dividends,
  and the account's API-key permissions.
- **Dual-format responses** — markdown for humans and LLMs (default) or JSON for
  programmatic use, per call via `response_format`.
- **Strict inputs, uniform errors.** One pydantic model per tool (`extra="forbid"`), and
  Binance's `{code, msg}` errors normalised into readable strings with a hint for the
  codes that have a known fix (`-1021` clock drift, `-1022` signature, `-2015`
  permissions/IP, `-2010` filters, `-1127` window too wide …).
- **stdio-only transport.** No HTTP/SSE and no logging to stdout — stdout is the
  protocol channel.

## Available Tools

<!-- TOOL TABLE START -->
All **83 tools**, grouped by module. 🔒 = refused unless `BINANCE_ALLOW_TRADING=1`.

| Tool | Description |
|---|---|
| **Health** | |
| `binance_health_check` | Verify connectivity, clock drift, and API-key permissions against Binance. |
| **Market data** | |
| `binance_get_agg_trades` | Fetch compressed/aggregate trades (same price, same taker order, same timestamp merged). |
| `binance_get_avg_price` | Fetch the current average price over Binance's configured window (typically 5 min). |
| `binance_get_book_ticker` | Fetch the best bid/ask price and quantity for one, several, or all symbols. |
| `binance_get_exchange_info` | Look up trading rules, symbol status, and order filters for spot symbols. |
| `binance_get_klines` | Fetch OHLCV candlestick data for a symbol. |
| `binance_get_order_book` | Fetch the current order book (bids/asks) for a symbol. |
| `binance_get_recent_trades` | Fetch the most recent public trades for a symbol. |
| `binance_get_rolling_ticker` | Fetch price change statistics over an arbitrary rolling window. |
| `binance_get_ticker_24h` | Fetch 24-hour rolling price change statistics. |
| `binance_get_ticker_price` | Fetch the latest price for one, several, or all symbols. |
| `binance_get_trading_day_ticker` | Fetch price change statistics for the current trading day (a fixed calendar window). |
| `binance_get_ui_klines` | Fetch presentation-adjusted candlestick data, matching Binance's own chart UI. |
| **Spot account** | |
| `binance_get_allocations` | List Smart Order Routing (SOR) allocations — the per-symbol fills behind an SOR order. |
| `binance_get_commission_rates` | Get the account's standard/special/tax commission rates for one symbol. |
| `binance_get_order_rate_limits` | Get the account's current order-rate-limit usage (per-second/day order counts). |
| `binance_get_prevented_matches` | List orders rejected by Self-Trade Prevention (STP) for a symbol. |
| `binance_get_spot_account` | Get spot account state: trade/withdraw/deposit flags, commission rates, balances. |
| **Spot orders** | |
| `binance_cancel_all_open_orders` 🔒 | Cancel EVERY open order on one symbol, including order-list legs. |
| `binance_cancel_order` 🔒 | Cancel one open spot order by id. |
| `binance_cancel_replace_order` 🔒 | Cancel one order and place its replacement in a single request. |
| `binance_get_all_orders` | List a symbol's orders — open, filled, cancelled and expired alike. |
| `binance_get_open_orders` | List the orders currently resting on the book. |
| `binance_get_order` | Look up one order — open, filled, cancelled or expired — by id. |
| `binance_place_order` 🔒 | Place a REAL spot order on Binance. This spends real money. |
| `binance_test_order` | Validate an order against Binance's filters WITHOUT sending it to the order book. |
| **Order lists (OCO / OTO / OTOCO)** | |
| `binance_cancel_order_list` 🔒 | Cancel an ENTIRE order list — every leg of it — by id. |
| `binance_get_all_order_lists` | List this account's order lists — working, completed and cancelled — across symbols. |
| `binance_get_open_order_lists` | List the order lists that are still working, across every symbol. |
| `binance_get_order_list` | Look up one order list — OCO, OTO or OTOCO — by id. |
| `binance_place_oco_order` 🔒 | Place a REAL one-cancels-the-other pair (take-profit + stop). This spends real money. |
| `binance_place_oto_order` 🔒 | Place a REAL one-triggers-the-other pair (entry, then follow-up). Real money. |
| `binance_place_otoco_order` 🔒 | Place a REAL entry that arms a take-profit/stop pair when it fills. Real money. |
| **Trade history** | |
| `binance_discover_traded_symbols` | Work out which symbols this account plausibly traded — Binance will not tell you. |
| `binance_get_all_my_trades` | Collect every fill across every symbol this account traded — the "all my trades" answer. |
| `binance_get_my_trades` | Fetch YOUR executed trades (fills) for one symbol. |
| **Wallet — deposits & withdrawals** | |
| `binance_get_all_deposits` | Every crypto deposit since `since` — the 90-day cap walked for you. |
| `binance_get_all_withdrawals` | Every crypto withdrawal since `since` — the 90-day cap walked for you. |
| `binance_get_coin_config` | Per-coin deposit/withdraw switches, networks, fees and minimums. |
| `binance_get_deposit_address` | Get the deposit address for one coin on one network. |
| `binance_get_deposit_addresses` | List every deposit address issued for one coin, across networks. |
| `binance_get_deposit_history` | List crypto deposits into the account for one window (up to 90 days). |
| `binance_get_withdraw_history` | List crypto withdrawals out of the account for one window (up to 90 days). |
| **Wallet — assets, funding & transfers** | |
| `binance_convert_dust_to_bnb` 🔒 | Convert small balances to BNB. **Irreversible** — the assets are sold for BNB. |
| `binance_get_asset_detail` | Report per-asset deposit/withdraw status, withdraw fee and minimum. |
| `binance_get_asset_dividends` | List asset distributions credited to the account (airdrops, rebates, interest). |
| `binance_get_dust_convertible` | Preview which small balances can be converted to BNB, and what they are worth. |
| `binance_get_dust_log` | List past dust-to-BNB conversions, with the per-asset detail of each one. |
| `binance_get_funding_wallet` | Read the Funding wallet — the wallet behind Binance Pay, Card and Gift Card. |
| `binance_get_trade_fees` | Report the maker/taker commission rates that apply to this account. |
| `binance_get_transfer_history` | List past transfers between the account's own wallets, one direction at a time. |
| `binance_get_user_assets` | List the Spot wallet's non-zero balances, optionally valued in BTC. |
| `binance_get_wallet_balances` | Show one total per wallet: Spot, Funding, Cross/Isolated Margin, Futures, Earn… |
| `binance_transfer_between_wallets` 🔒 | Move funds between the account's OWN wallets (Spot ⇄ Funding ⇄ Margin ⇄ Futures). |
| **Wallet — account status** | |
| `binance_get_account_info` | Report the account's VIP tier and which product lines are enabled. |
| `binance_get_account_snapshot` | Return daily balance snapshots for the SPOT, MARGIN or FUTURES wallet. |
| `binance_get_account_status` | Report whether the account is in good standing with Binance. |
| `binance_get_api_restrictions` | Report the full permission flag set on the configured API key. |
| `binance_get_api_trading_status` | Report whether spot trading is locked and what triggered it. |
| `binance_get_delist_schedule` | List symbols scheduled to be delisted, with their delisting date. |
| `binance_get_system_status` | Report whether the Binance system is up or under maintenance. |
| **Fiat** | |
| `binance_get_fiat_history` | Walk fiat deposit/withdraw or buy/sell history across pages and time windows. |
| `binance_get_fiat_orders` | List fiat-rail deposit or withdraw orders (bank transfer/card top-up of the fiat wallet). |
| `binance_get_fiat_payments` | List crypto buy/sell payments made with fiat (bank transfer or bank-issued card). |
| **Binance Pay** | |
| `binance_get_pay_history` | Walk up to 18 months of Binance Pay history, past the 90-day / 100-row API caps. |
| `binance_get_pay_transactions` | Fetch Binance Pay transactions (merchant payments, C2C, refunds, payouts) for the account. |
| **Convert** | |
| `binance_accept_convert_quote` 🔒 | Accept a convert quote and EXECUTE the conversion. This moves real funds. |
| `binance_cancel_convert_limit_order` 🔒 | Cancel a resting convert limit order. |
| `binance_get_convert_asset_info` | Show the decimal precision (`fraction`) Convert accepts for each asset. |
| `binance_get_convert_history` | List past conversions, either for one <=30-day window or across a walked range. |
| `binance_get_convert_open_limit_orders` | List the convert limit orders currently resting on the account. |
| `binance_get_convert_order_status` | Check one conversion's status by orderId or by the quoteId it came from. |
| `binance_get_convert_pairs` | List the convertible asset pairs and their per-pair minimum/maximum amounts. |
| `binance_get_convert_quote` | Request a convert quote: a reserved ratio, valid for 10 s to 2 minutes. |
| `binance_place_convert_limit_order` 🔒 | Place a convert LIMIT order: convert automatically if the ratio is reached. |
| **Spot algo (TWAP)** | |
| `binance_cancel_algo_order` 🔒 | Cancel a working spot TWAP algo order. |
| `binance_get_algo_order_history` | List finished spot TWAP algo orders — filled, cancelled or expired. |
| `binance_get_algo_sub_orders` | List the individual orders a TWAP placed on the book, with fills and fees. |
| `binance_get_open_algo_orders` | List the spot TWAP algo orders that are still working, across every symbol. |
| `binance_place_twap_order` 🔒 | Place a REAL spot TWAP algo order on Binance. This spends real money. |
| **Simple Earn** | |
| `binance_get_earn_account` | Summarize total Simple Earn holdings (Flexible + Locked) in BTC and USDT. |
| `binance_get_earn_flexible_positions` | List the caller's Simple Earn Flexible subscriptions and their live APR. |
| `binance_get_earn_locked_positions` | List the caller's Simple Earn Locked subscriptions and their APY/redeem dates. |
<!-- TOOL TABLE END -->

Read tools return markdown (default) or JSON; mutating tools return a confirmation that
echoes exactly what Binance returned.

## Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — for the zero-install `uvx` path and for local
  development (Python 3.13+ is only needed for the clone path; `uvx` brings its own).
- **A Binance API key.** Binance → Account → **API Management**:
  1. Choose **Self-generated** and paste an **Ed25519 public key** (recommended — Binance
     has deprecated HMAC keys, and an unrestricted HMAC key may only hold "Enable
     Reading"). Generate the pair locally and keep the private key out of the repo:
     ```bash
     openssl genpkey -algorithm ed25519 -out ~/.config/binance/ed25519.pem
     openssl pkey -pubout -in ~/.config/binance/ed25519.pem     # paste this one into Binance
     ```
     A "System generated" HMAC key (key + secret) also works.
  2. Permissions: **Enable Reading** on; **Enable Spot & Margin Trading** only if you
     will place orders; **Enable Withdrawals off**; **Permits Universal Transfer** only
     if you want `binance_transfer_between_wallets`.
  3. **Restrict access to trusted IPs** — the IP of the machine that runs this server.
- **(Optional) Spot testnet keys** from <https://testnet.binance.vision> to exercise the
  trading tools with virtual funds. The testnet serves only `/api/v3` (market data,
  account, orders) — wallet, fiat, Pay, convert and algo tools return `404` there.
- **(Optional) Docker** if you prefer the container path.

## Quickstart

```bash
git clone https://github.com/trustxai/binance-mcp.git
cd binance-mcp
uv sync --group dev
cp .env.example .env          # set BINANCE_API_KEY and BINANCE_PRIVATE_KEY_PATH (or BINANCE_API_SECRET)
uv run amazing-binance-mcp    # starts the stdio server
```

The server speaks MCP over stdio, so it is normally launched by an MCP client (see
[Client Configuration](#client-configuration)) rather than run by hand. Call
`binance_health_check` first: it reports connectivity, clock drift against Binance's
server time, and the key's permission flags — with a ⚠️ if withdrawals are enabled or
the key has no IP allowlist.

## Run with uvx (zero install)

```bash
BINANCE_API_KEY=your_key BINANCE_PRIVATE_KEY_PATH=~/.config/binance/ed25519.pem uvx amazing-binance-mcp
```

Public market data needs no credentials at all: `uvx amazing-binance-mcp` with nothing
set serves the market-data and health tools.

> Use `amazing-binance-mcp`, not `binance-mcp`. The bare name on PyPI is an unrelated
> package and will **not** run this server.

## Client Configuration

Every client launches the server as a subprocess and passes credentials through `env`.
Replace `your_key` and the PEM path; for an HMAC key use `BINANCE_API_SECRET` instead of
`BINANCE_PRIVATE_KEY_PATH`. Add `"BINANCE_ALLOW_TRADING": "1"` only for a client you
want to be able to trade.

### Cursor

Add to `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per project):

```json
{
  "mcpServers": {
    "binance": {
      "command": "uvx",
      "args": ["amazing-binance-mcp"],
      "env": {
        "BINANCE_API_KEY": "your_key",
        "BINANCE_PRIVATE_KEY_PATH": "/Users/you/.config/binance/ed25519.pem"
      }
    }
  }
}
```

### Claude Desktop

Edit `claude_desktop_config.json` (Settings → Developer → Edit Config; on macOS it lives
at `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "binance": {
      "command": "uvx",
      "args": ["amazing-binance-mcp"],
      "env": {
        "BINANCE_API_KEY": "your_key",
        "BINANCE_PRIVATE_KEY_PATH": "/Users/you/.config/binance/ed25519.pem"
      }
    }
  }
}
```

Restart Claude Desktop after saving.

### Claude Code

```bash
claude mcp add binance \
  --env BINANCE_API_KEY=your_key \
  --env BINANCE_PRIVATE_KEY_PATH=$HOME/.config/binance/ed25519.pem \
  -- uvx amazing-binance-mcp
```

Or add the equivalent block to `~/.claude.json` under `mcpServers` (same shape as the
Cursor example above).

### MCP Inspector

```bash
BINANCE_API_KEY=your_key BINANCE_PRIVATE_KEY_PATH=~/.config/binance/ed25519.pem \
  npx @modelcontextprotocol/inspector uvx amazing-binance-mcp
```

If you cloned the repo, `uv run mcp dev src/binance_mcp/server.py` does the same (the
`mcp` dev CLI ships in the dev dependency group).

### Docker

Build the image from the repo `Dockerfile`, then point any client at `docker run`. The
PEM has to be mounted into the container; with an HMAC key you can pass the secret through
`env` instead and drop the volume.

```bash
docker build -t amazing-binance-mcp .
```

```json
{
  "mcpServers": {
    "binance": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "/Users/you/.config/binance/ed25519.pem:/keys/ed25519.pem:ro",
        "-e", "BINANCE_API_KEY",
        "-e", "BINANCE_PRIVATE_KEY_PATH",
        "amazing-binance-mcp"
      ],
      "env": {
        "BINANCE_API_KEY": "your_key",
        "BINANCE_PRIVATE_KEY_PATH": "/keys/ed25519.pem"
      }
    }
  }
}
```

The `-i` flag is required — the server communicates over stdin/stdout.

## Authentication

Every signed request carries the key in the `X-MBX-APIKEY` header plus a `timestamp`,
a `recvWindow` and a `signature` over the query string:

- **Ed25519 key** (recommended): the private key PEM at `BINANCE_PRIVATE_KEY_PATH`
  (optionally encrypted — `BINANCE_PRIVATE_KEY_PASSPHRASE`) signs the payload; the
  signature is base64 and percent-encoded, as Binance requires. RSA PEMs work the same way.
- **HMAC key**: `BINANCE_API_SECRET` produces the HMAC-SHA256 hex signature. Supported,
  but Binance has deprecated HMAC keys.

The key is never logged or echoed. Missing credentials fail lazily at the first signed
request with a clear message; public market data never needs them. The signing code is
unit-tested against the example vectors Binance publishes in its own documentation.

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `BINANCE_API_KEY` | for account/trading tools | — | The API key (`X-MBX-APIKEY`). Public market data works without it. |
| `BINANCE_PRIVATE_KEY_PATH` | one of the two | — | Path to the Ed25519 (or RSA) private-key PEM of a "Self-generated" key. Takes precedence over the HMAC secret. |
| `BINANCE_PRIVATE_KEY_PASSPHRASE` | no | — | Passphrase of an encrypted PEM. |
| `BINANCE_API_SECRET` | one of the two | — | HMAC secret of a "System generated" key. |
| `BINANCE_ALLOW_TRADING` | no | off | Kill-switch. `1` allows orders, cancels, order lists, transfers, convert, TWAP and dust conversion. Never withdrawals. |
| `BINANCE_TESTNET` | no | off | `1` routes `/api/v3` to the spot testnet (`https://testnet.binance.vision`). Use testnet keys. `/sapi` tools return `404` there. |
| `BINANCE_API_URL` | no | `https://api.binance.com` | REST base URL (alternatives: `api-gcp`, `api1`…`api4`). Overrides the testnet switch when set. |
| `BINANCE_RECV_WINDOW_MS` | no | `5000` | Signed-request validity window in ms (max 60000). Raise on `-1021`. |
| `BINANCE_REQUEST_TIMEOUT_SECONDS` | no | `30` | Per-request HTTP timeout. |

## Enabling trading

1. Give the key **Enable Spot & Margin Trading** (and an IP allowlist).
2. Set `BINANCE_ALLOW_TRADING=1` in that client's `env` — and only there.
3. Validate with `binance_test_order` first: it runs Binance's filter checks (lot size,
   price filter, notional) without touching the book. `binance_get_exchange_info` shows
   the filters for a symbol.

Start on the testnet: `BINANCE_TESTNET=1` with keys from <https://testnet.binance.vision>
lets you place, query and cancel orders with virtual funds. The suite's own trading
smoke test refuses to run anywhere but the testnet.

## History walks

Binance caps history queries per call — deposits/withdrawals and Pay at 90 days,
convert at 30, spot trades and orders at 24 hours when you pass a time range — and
`myTrades` needs a symbol. The `binance_get_all_*` and `*_history` tools walk those
windows newest-first from `until` (default now) down to `since`, within a call budget
sized to the endpoint's weight (withdraw history costs 18 000 UID weight per call, so its
default budget is 10 calls; deposits cost 1 and get 60).

When a budget runs out before `since`, the response says so and returns
`resume_before` — the boundary of the next unfetched range. Call the tool again with the
same `since` and that `resume_before` to continue; rows are deduplicated by id, so an
overlap is harmless. If the budget could not finish even the newest window the tool says
so explicitly instead of handing you a cursor that would repeat the same calls.

`binance_get_all_my_trades` first discovers which symbols you may have traded (assets
you hold or held, crossed with the exchange's symbol list — Binance has no endpoint for
this) and walks each one with `fromId`; it returns a `cursor` of the last trade id per
symbol that you can paste back for incremental runs.

## What is not available

- **Binance Card spending.** Binance exposes no API for card transactions. The closest
  reads are `binance_get_funding_wallet` (the wallet behind Pay/Card), Pay transactions
  with `walletType` 4/6, and spot⇄funding transfer history.
- **Withdrawals**, by design — use the Binance app.
- **Futures, margin trading, sub-accounts** — out of scope for this server.

## Running Manually

```bash
uv run amazing-binance-mcp     # console script (recommended)
uv run python -m binance_mcp    # module entry point
```

## Troubleshooting

- **`-1021 Timestamp for this request is outside of the recvWindow`.** Your clock drifts
  from Binance's. `binance_health_check` prints the drift; sync the clock (NTP) or raise
  `BINANCE_RECV_WINDOW_MS`.
- **`-1022 Signature for this request is not valid`.** The secret does not match the key,
  or the key type is wrong (an HMAC secret with a self-generated key, or vice versa).
- **`-2015 Invalid API-key, IP, or permissions for action`.** The key lacks the
  permission the tool needs (Reading, Spot & Margin Trading, Permits Universal
  Transfer), or this machine's IP is not on the key's allowlist.
- **`Error: … trading is disabled`.** Expected: set `BINANCE_ALLOW_TRADING=1` for the
  client that should trade. The dry-run `binance_test_order` never needs it.
- **`-1127` / "window too wide".** You passed a time range wider than the endpoint allows
  (24 h for trades/orders, 90 d for deposits/withdrawals/Pay, 30 d for convert). Narrow
  it, or use the walk tool for that history.
- **`429`, `418`, or "used weight" warnings.** `/api` calls share a 6000-weight-per-minute
  budget per IP; `/sapi` has separate per-IP and per-UID budgets. Heavy tools say so in
  their descriptions (account snapshot 2400, fiat orders 45 000 UID, withdraw history
  18 000 UID). Back off; a `418` is a temporary IP ban.
- **`404` on wallet/fiat/Pay/convert/algo tools.** You are on the testnet, which serves
  only `/api/v3`. Those tools need the real API with a read-only key.
- **`1 validation error for …Arguments` when calling a tool by hand.** Every tool takes a
  single `params` object: `{"params": {"symbol": "BTCUSDT"}}`, not the bare fields. MCP
  clients read that from the schema; scripts and the Inspector must nest it.
- **`uvx binance-mcp` runs the wrong thing.** This server is `amazing-binance-mcp`.
- **"Why is my card spend missing?"** See [What is not available](#what-is-not-available).

## Contributing

Conventional Commits drive releases (release-please). Before pushing, run the full gate —
it must exit `0`:

```bash
uv run pytest -m "not live and not trading" && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/
```

Live smokes run with a read-only key in `.env` (`uv run pytest -m live`); the trading
smoke needs `BINANCE_TESTNET=1 BINANCE_TEST_ALLOW_TRADING=1` and testnet keys. Regenerate
the tool table with `uv run python scripts/gen_tool_table.py --write`.

## License

[Apache-2.0](LICENSE)
