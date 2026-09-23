"""Unit tests for Settings and the cached singleton."""

from __future__ import annotations

import pytest

from binance_mcp.config import DEFAULT_API_URL, TESTNET_API_URL, Settings, get_settings


def test_defaults() -> None:
    settings = Settings()
    assert settings.binance_api_key == ""
    assert settings.binance_api_secret == ""
    assert settings.binance_private_key_path == ""
    assert settings.binance_api_url == DEFAULT_API_URL
    assert settings.binance_testnet is False
    assert settings.binance_recv_window_ms == 5000
    assert settings.binance_request_timeout_seconds == 30.0
    assert settings.binance_allow_trading is False
    assert settings.has_api_key is False
    assert settings.has_credentials is False
    assert settings.base_url == DEFAULT_API_URL


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_API_KEY", "key123")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret456")
    monkeypatch.setenv("BINANCE_RECV_WINDOW_MS", "10000")
    monkeypatch.setenv("BINANCE_REQUEST_TIMEOUT_SECONDS", "5.5")
    monkeypatch.setenv("BINANCE_ALLOW_TRADING", "1")
    settings = Settings()
    assert settings.binance_api_key == "key123"
    assert settings.binance_api_secret == "secret456"
    assert settings.binance_recv_window_ms == 10000
    assert settings.binance_request_timeout_seconds == 5.5
    assert settings.binance_allow_trading is True
    assert settings.has_credentials is True


def test_key_without_secret_is_not_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_API_KEY", "key123")
    settings = Settings()
    assert settings.has_api_key is True
    assert settings.has_credentials is False


def test_pem_path_counts_as_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_API_KEY", "key123")
    monkeypatch.setenv("BINANCE_PRIVATE_KEY_PATH", "/tmp/ed25519.pem")
    assert Settings().has_credentials is True


def test_testnet_switches_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_TESTNET", "true")
    assert Settings().base_url == TESTNET_API_URL


def test_custom_url_wins_over_testnet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_TESTNET", "true")
    monkeypatch.setenv("BINANCE_API_URL", "https://api-gcp.binance.com")
    assert Settings().base_url == "https://api-gcp.binance.com"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
