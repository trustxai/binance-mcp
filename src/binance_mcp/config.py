"""Settings for the Binance MCP server, loaded from env vars / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_API_URL = "https://api.binance.com"
TESTNET_API_URL = "https://testnet.binance.vision"


class Settings(BaseSettings):
    """All configuration, derived from `BINANCE_*` environment variables.

    Every field has a default so importing the package never fails; missing
    credentials surface as a descriptive error on the first signed request.
    Public market-data endpoints work with no credentials at all.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API key (sent in the X-MBX-APIKEY header). Create it in Binance → API Management
    # with "Enable Reading" + "Enable Spot & Margin Trading", withdrawals OFF, and an IP
    # allowlist. This server never exposes a withdrawal endpoint regardless of the key.
    binance_api_key: str = ""

    # HMAC-SHA256 secret (for "System generated" HMAC keys). Ignored when a PEM
    # private key path is configured.
    binance_api_secret: str = ""

    # Path to a PEM private key for "Self-generated" Ed25519 (recommended by Binance)
    # or RSA keys. When set, it takes precedence over the HMAC secret.
    binance_private_key_path: str = ""
    binance_private_key_passphrase: str = ""

    # REST base URL. Override only for proxies/testing; see `base_url` for the testnet switch.
    binance_api_url: str = DEFAULT_API_URL

    # Route /api/v3 calls to the Spot testnet (https://testnet.binance.vision). The
    # testnet has NO /sapi endpoints (wallet, fiat, convert, pay, algo) — those 404 there.
    binance_testnet: bool = False

    # Signed-request validity window in ms (Binance default 5000, max 60000).
    binance_recv_window_ms: int = 5000

    binance_request_timeout_seconds: float = 30.0

    # Kill-switch for anything that moves funds or changes account state (placing or
    # cancelling orders, transfers, convert acceptQuote, algo orders, dust conversion).
    # Off by default: those tools return an error until BINANCE_ALLOW_TRADING=1.
    # `POST /api/v3/order/test` (dry-run validation) is always allowed.
    binance_allow_trading: bool = False

    @property
    def has_api_key(self) -> bool:
        return bool(self.binance_api_key)

    @property
    def has_credentials(self) -> bool:
        """True when a signed request can be built (key + a signing secret/key)."""
        return self.has_api_key and bool(self.binance_api_secret or self.binance_private_key_path)

    @property
    def base_url(self) -> str:
        """Effective REST base: the testnet when `binance_testnet` is on and no custom URL is set."""
        if self.binance_testnet and self.binance_api_url == DEFAULT_API_URL:
            return TESTNET_API_URL
        return self.binance_api_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton (tests call `get_settings.cache_clear()`)."""
    return Settings()
