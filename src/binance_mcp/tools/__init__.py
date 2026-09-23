"""Tool registration for binance_mcp.

ALL wave modules are listed here from day one (even while some are still empty
stubs) so that parallel worktrees implementing individual modules never need to
touch this file — eliminating merge conflicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register_all(mcp: FastMCP) -> None:
    """Import every tool module; @mcp.tool decorators self-register on import."""
    from binance_mcp.tools import (  # noqa: F401
        convert,
        fiat,
        health,
        market_data,
        order_lists,
        pay,
        simple_earn,
        spot_account,
        spot_algo,
        spot_orders,
        trade_history,
        wallet_account,
        wallet_asset,
        wallet_capital,
    )
