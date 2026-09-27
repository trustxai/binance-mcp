"""FastMCP server definition and entry points for binance_mcp."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("binance_mcp")

# FastMCP configures root logging at INFO to stderr, which MCP clients persist to log
# files. At INFO, httpx/httpcore log one "HTTP Request: ..." line per call — the full
# signed query string (timestamp + signature), so quiet that down to WARNING.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Register all tools (side-effect imports via decorators)
from binance_mcp.tools import register_all  # noqa: E402

register_all(mcp)


def main_stdio() -> None:
    """Entry point for local / Docker stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main_stdio()
