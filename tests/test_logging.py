"""Regression tests: httpx/httpcore request logs must not leak signed query strings.

FastMCP's constructor configures root logging at INFO to stderr, so without a fix the
httpx logger emits one "HTTP Request: ..." line per call — full query string included,
i.e. timestamp + signature (and, for signed GETs, every other param). MCP clients
persist stderr to log files, so that line quietly writes account activity to disk.
"""

from __future__ import annotations

import logging
import subprocess
import sys

import httpx
import pytest

from binance_mcp.client import BinanceClient
from binance_mcp.config import Settings


def test_httpx_logger_effective_level_is_at_least_warning() -> None:
    """Checked in a fresh subprocess, not in-process.

    pytest's own logging plugin pre-attaches a handler to the root logger before any
    test runs, which makes the SDK's `logging.basicConfig` a silent no-op (it does
    nothing once the root logger already has handlers) — masking the bug if checked
    in-process. A bare subprocess has no such handler, so it reproduces the real,
    fresh-process behavior a running MCP server actually has.
    """
    script = (
        "import logging, binance_mcp.server\n"
        "level = logging.getLogger('httpx').getEffectiveLevel()\n"
        "raise SystemExit(0 if level >= logging.WARNING else 1)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, f"httpx effective level too low (stderr: {result.stderr})"


async def test_signed_request_emits_no_httpx_log_record(caplog: pytest.LogCaptureFixture) -> None:
    import binance_mcp.server  # noqa: F401  (import triggers the fix under test)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = BinanceClient(
        settings=Settings(binance_api_key="fake-key", binance_api_secret="fake-secret"),
        transport=httpx.MockTransport(handler),
    )

    with caplog.at_level(logging.INFO):
        await client.request("GET", "/api/v3/account", auth="signed")

    assert not any(record.name.startswith("httpx") for record in caplog.records)
    assert not any("signature=" in record.getMessage() for record in caplog.records)
