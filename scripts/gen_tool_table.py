"""Print the README's "Available Tools" table from the live registry.

Usage: `uv run python scripts/gen_tool_table.py [--check | --write]`

Groups tools by module in board order and takes each tool's first docstring
line as its description, so the README can never drift from the code. The table
lives between `<!-- TOOL TABLE START -->` / `<!-- TOOL TABLE END -->` markers in
README.md: `--write` replaces it in place, `--check` exits 1 when it is stale.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from binance_mcp.server import mcp

# Board order (wave 1 → 3), with the README group titles.
MODULE_ORDER: list[tuple[str, str]] = [
    ("health", "Health"),
    ("market_data", "Market data"),
    ("spot_account", "Spot account"),
    ("spot_orders", "Spot orders"),
    ("order_lists", "Order lists (OCO / OTO / OTOCO)"),
    ("trade_history", "Trade history"),
    ("wallet_capital", "Wallet — deposits & withdrawals"),
    ("wallet_asset", "Wallet — assets, funding & transfers"),
    ("wallet_account", "Wallet — account status"),
    ("fiat", "Fiat"),
    ("pay", "Binance Pay"),
    ("convert", "Convert"),
    ("spot_algo", "Spot algo (TWAP)"),
    ("simple_earn", "Simple Earn"),
]

MUTATING_PREFIXES = (
    "binance_place_",
    "binance_cancel_",
    "binance_accept_",
    "binance_transfer_",
    "binance_convert_dust",
)


def _first_line(doc: str | None) -> str:
    return (doc or "").strip().splitlines()[0].strip() if doc else ""


def build_table() -> str:
    tools = asyncio.run(mcp.list_tools())
    by_module: dict[str, list[tuple[str, str, bool]]] = {}
    manager = mcp._tool_manager  # noqa: SLF001 — the SDK exposes no public per-tool module lookup
    for tool in tools:
        registered = manager.get_tool(tool.name)
        fn = getattr(registered, "fn", None)
        module = (getattr(fn, "__module__", "") or "").rsplit(".", 1)[-1]
        gated = tool.name.startswith(MUTATING_PREFIXES) and not tool.name.startswith("binance_test_")
        by_module.setdefault(module, []).append((tool.name, _first_line(tool.description), gated))

    lines = ["| Tool | Description |", "|---|---|"]
    total = 0
    for module, title in MODULE_ORDER:
        rows = sorted(by_module.pop(module, []))
        if not rows:
            continue
        lines.append(f"| **{title}** | |")
        for name, desc, gated in rows:
            flag = " 🔒" if gated else ""
            lines.append(f"| `{name}`{flag} | {desc} |")
            total += 1
    for module, rows in by_module.items():  # anything not in MODULE_ORDER — should be empty
        lines.append(f"| **{module}** | |")
        for name, desc, _gated in sorted(rows):
            lines.append(f"| `{name}` | {desc} |")
            total += 1
    header = f"All **{total} tools**, grouped by module. 🔒 = refused unless `BINANCE_ALLOW_TRADING=1`."
    return header + "\n\n" + "\n".join(lines) + "\n"


START = "<!-- TOOL TABLE START -->"
END = "<!-- TOOL TABLE END -->"


def _splice(readme: str, table: str) -> str:
    head, _, rest = readme.partition(START)
    _, _, tail = rest.partition(END)
    if not head or not tail:
        raise SystemExit(f"README.md lacks the {START} / {END} markers")
    return f"{head}{START}\n{table}{END}{tail}"


def main() -> int:
    table = build_table()
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    if "--check" in sys.argv:
        if _splice(readme_path.read_text(), table) != readme_path.read_text():
            sys.stderr.write("README.md tool table is stale — run scripts/gen_tool_table.py --write\n")
            return 1
        return 0
    if "--write" in sys.argv:
        readme_path.write_text(_splice(readme_path.read_text(), table))
        return 0
    sys.stdout.write(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
