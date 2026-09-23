"""Unit tests for the shared response formatters."""

from __future__ import annotations

import json

from binance_mcp.formatters import (
    ResponseFormat,
    clip_response,
    epoch_to_human,
    fmt_num,
    paginated_response,
    to_json,
)


def test_epoch_to_human_milliseconds() -> None:
    assert epoch_to_human(1720000000000) == "2024-07-03 09:46:40 UTC"


def test_epoch_to_human_string_milliseconds() -> None:
    assert epoch_to_human("1720000000000") == "2024-07-03 09:46:40 UTC"


def test_epoch_to_human_seconds_autodetected() -> None:
    assert epoch_to_human(1720000000) == "2024-07-03 09:46:40 UTC"


def test_epoch_to_human_empty() -> None:
    assert epoch_to_human(None) == "N/A"
    assert epoch_to_human("") == "N/A"
    assert epoch_to_human(0) == "N/A"


def test_epoch_to_human_non_numeric_passthrough() -> None:
    assert epoch_to_human("not-a-date") == "not-a-date"


def test_fmt_num_strips_trailing_zeros() -> None:
    assert fmt_num("0.00100000") == "0.001"
    assert fmt_num("12.50000000") == "12.5"
    assert fmt_num("100.00000000") == "100"
    assert fmt_num("0E-8") == "0"


def test_fmt_num_passthrough_and_empty() -> None:
    assert fmt_num("abc") == "abc"
    assert fmt_num(None) == "N/A"
    assert fmt_num("") == "N/A"


def test_clip_response_short_untouched() -> None:
    assert clip_response("hello", max_bytes=10) == "hello"


def test_clip_response_truncates_with_note() -> None:
    out = clip_response("x" * 100, max_bytes=10)
    assert out.startswith("x" * 10)
    assert "truncated" in out


def test_to_json_stable() -> None:
    assert json.loads(to_json({"a": 1})) == {"a": 1}


def _fmt(item: dict[str, object]) -> str:
    return f"- **{item['symbol']}**"


def test_paginated_response_markdown_with_total() -> None:
    output = paginated_response(
        items=[{"symbol": "BTCUSDT"}],
        total=10,
        limit=1,
        offset=0,
        fmt=ResponseFormat.MARKDOWN,
        item_formatter=_fmt,
        title="Symbols",
    )
    assert "# Symbols" in output
    assert "total **10**" in output
    assert "next offset → **1**" in output
    assert "- **BTCUSDT**" in output


def test_paginated_response_markdown_full_page_has_more() -> None:
    output = paginated_response(
        items=[{"symbol": "A"}, {"symbol": "B"}],
        limit=2,
        offset=0,
        fmt=ResponseFormat.MARKDOWN,
        item_formatter=_fmt,
        title="Things",
    )
    assert "More available" in output


def test_paginated_response_json() -> None:
    output = paginated_response(
        items=[{"symbol": "A"}],
        limit=20,
        offset=0,
        fmt=ResponseFormat.JSON,
        item_formatter=_fmt,
        title="Things",
    )
    payload = json.loads(output)
    assert payload["title"] == "Things"
    assert payload["count"] == 1
    assert payload["has_more"] is False
    assert payload["items"] == [{"symbol": "A"}]


def test_paginated_response_empty_markdown() -> None:
    output = paginated_response(
        items=[],
        limit=20,
        offset=0,
        fmt=ResponseFormat.MARKDOWN,
        item_formatter=_fmt,
        title="Nothing",
    )
    assert "_No items._" in output
