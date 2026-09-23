# Binance MCP Server (amazing-binance-mcp)

A [Model Context Protocol](https://modelcontextprotocol.io) server for the
[Binance Spot + Wallet REST API](https://developers.binance.com/docs/binance-spot-api-docs),
built on the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
(FastMCP, stdio transport). It authenticates with a plain **API key** (HMAC or Ed25519 —
no OAuth), so an LLM can read your deposits, withdrawals, trades, balances and market
data, and — only when you flip the kill-switch — place and cancel spot orders.

> The PyPI distribution and console script are both **`amazing-binance-mcp`** (the bare
> `binance-mcp` name is taken on PyPI by an unrelated package). Always run
> `uvx amazing-binance-mcp`.

> **Status: under construction.** The tool table, quickstart and client configuration
> land with the first release.

## Safety model

- **No withdrawals, ever.** The server has no withdrawal tool and the HTTP client refuses
  the withdrawal endpoint under any configuration. Create the API key with "Enable
  Withdrawals" OFF and an IP allowlist as well.
- **Trading is off by default.** Anything that moves funds or changes account state
  (orders, cancels, transfers, convert, algo orders, dust conversion) returns an error
  until `BINANCE_ALLOW_TRADING=1`. The order dry-run (`POST /api/v3/order/test`) always works.
- **Testnet first.** `BINANCE_TESTNET=1` routes `/api/v3` to the spot testnet so
  strategies can be exercised without real funds.

## License

Apache-2.0 — see [LICENSE](LICENSE).
