"""Response formatting shared by every tool (markdown / JSON dual output)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

# Context-window guard: MCP responses cap at 1 MB; stay well under it.
MAX_RESPONSE_BYTES = 900_000


class ResponseFormat(StrEnum):
    """Output format selector present on list/get tools."""

    MARKDOWN = "markdown"
    JSON = "json"


def to_json(data: Any) -> str:
    """Serialize any payload for the LLM (stable, human-readable)."""
    return json.dumps(data, indent=2, default=str)


def epoch_to_human(value: Any) -> str:
    """Render a Binance timestamp as UTC ISO-8601.

    Binance timestamps are unix epochs in *milliseconds*; second-resolution
    epochs are auto-detected and handled too.
    """
    if value in (None, "", 0, "0"):
        return "N/A"
    try:
        epoch = float(value)
    except (TypeError, ValueError):
        return str(value)
    if epoch > 1e11:  # milliseconds
        epoch /= 1000.0
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt_num(value: Any) -> str:
    """Render a Binance decimal string without trailing zeros ("0.00100000" → "0.001")."""
    if value in (None, ""):
        return "N/A"
    try:
        dec = Decimal(str(value))
    except InvalidOperation:
        return str(value)
    if dec == dec.to_integral_value():
        return str(dec.quantize(Decimal(1)))
    return format(dec.normalize(), "f")


def clip_response(text: str, max_bytes: int = MAX_RESPONSE_BYTES) -> str:
    """Truncate an oversized response with an explicit note instead of failing the call."""
    encoded = text.encode()
    if len(encoded) <= max_bytes:
        return text
    cut = encoded[:max_bytes].decode(errors="ignore")
    return f"{cut}\n\n_[truncated: response exceeded {max_bytes:,} bytes — narrow the query]_"


def paginated_response(
    *,
    items: list[dict[str, Any]],
    limit: int,
    offset: int,
    fmt: ResponseFormat,
    item_formatter: Callable[[dict[str, Any]], str],
    title: str,
    total: int | None = None,
) -> str:
    """Uniform paginated output for list tools.

    `has_more` is computed from `total` when the API provides one, otherwise
    from the page being full (`len(items) == limit`).
    """
    count = len(items)
    if total is not None:
        has_more = total > offset + count
    else:
        has_more = count == limit

    if fmt is ResponseFormat.JSON:
        return clip_response(
            to_json(
                {
                    "title": title,
                    "count": count,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "has_more": has_more,
                    "items": items,
                }
            )
        )

    lines = [f"# {title}", ""]
    if total is not None:
        lines.append(f"Showing **{count:,}** of total **{total:,}** (offset {offset:,}).")
    else:
        lines.append(f"Showing **{count:,}** item(s) (offset {offset:,}).")
    if has_more:
        lines.append(f"More available — next offset → **{offset + count:,}**.")
    lines.append("")
    for item in items:
        lines.append(item_formatter(item))
    if not items:
        lines.append("_No items._")
    return clip_response("\n".join(lines))
