"""Allow running the server with `python -m binance_mcp`."""

from __future__ import annotations

from binance_mcp.server import main_stdio

if __name__ == "__main__":
    main_stdio()
