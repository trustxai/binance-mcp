"""Shared pytest configuration: marker gating + settings isolation."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from binance_mcp.config import get_settings

# Load .env so live-marked tests can pick up real credentials; non-live tests
# are isolated from it by the autouse fixture below.
load_dotenv()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    has_creds = bool(os.environ.get("BINANCE_API_KEY")) and bool(
        os.environ.get("BINANCE_API_SECRET") or os.environ.get("BINANCE_PRIVATE_KEY_PATH")
    )
    on_testnet = os.environ.get("BINANCE_TESTNET", "").lower() in ("1", "true", "yes")
    allow_trading = os.environ.get("BINANCE_TEST_ALLOW_TRADING") == "1"
    skip_live = pytest.mark.skip(reason="live tests require BINANCE_API_KEY + secret/PEM in the environment / .env")
    skip_trading = pytest.mark.skip(
        reason="trading tests run ONLY on the spot testnet: require BINANCE_TESTNET=1 and BINANCE_TEST_ALLOW_TRADING=1"
    )
    for item in items:
        if "live" in item.keywords and not has_creds:
            item.add_marker(skip_live)
        if "trading" in item.keywords and not (has_creds and on_testnet and allow_trading):
            item.add_marker(skip_trading)


@pytest.fixture(autouse=True)
def _isolate_settings_env(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Keep the developer's real BINANCE_* env / .env out of non-live tests.

    Strips ambient BINANCE_* vars, chdirs away from the repo (so
    `Settings(env_file=".env")` finds nothing), and clears the settings cache
    before and after each non-live test. Live/trading tests are left untouched.
    """
    if "live" in request.keywords or "trading" in request.keywords:
        return
    for key in list(os.environ):
        if key.startswith("BINANCE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path_factory.mktemp("isolated-cwd"))
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)
