# Security Policy

This server holds API keys for a live exchange account. This page covers how to report
a vulnerability, what counts as one, and how to run the server safely.

## Reporting a vulnerability

**Do not open a public issue, pull request or discussion.** Report privately through
GitHub:

1. Open the repository's **Security** tab → **Report a vulnerability**
   (direct link: <https://github.com/trustxai/binance-mcp/security/advisories/new>).
2. Include:
   - the version (`pip show amazing-binance-mcp`, or the commit SHA);
   - the settings involved — variable **names** only, e.g. whether
     `BINANCE_ALLOW_TRADING` or `BINANCE_TESTNET` was set;
   - the tool calls and steps to reproduce, and what an attacker gains.

**Never include real credentials** — no API keys, secrets, PEM files, passphrases or
account identifiers. Build proofs of concept against the
[spot testnet](https://testnet.binance.vision) or with mocked responses (the test suite
uses `httpx.MockTransport`). If a live key ends up in a report anyway, delete it in
Binance → API Management right away: a key that has been shared is compromised.

### What to expect

The project has a single maintainer, so these are targets, not guarantees:

| Step | Target |
|---|---|
| Acknowledge the report | 3 business days |
| Confirm or rule out the issue, with a severity assessment | 10 business days |
| Release a fix or mitigation | 90 days from the report — sooner for anything that can move funds |

Once the fix is on PyPI, a GitHub Security Advisory is published (with a CVE when
warranted) and the CHANGELOG entry links to it. Reporters are credited in the advisory
unless they ask not to be. Please keep the details private until the advisory is out.

## Supported versions

The project is pre-1.0. Only the **latest release** on PyPI receives security fixes;
fixes ship as a new release, not as backports.

| Version | Supported |
|---|---|
| Latest `0.x` release | ✅ |
| Anything older | ❌ — upgrade with `uvx amazing-binance-mcp@latest` |

## Scope

The server makes three promises (see the [safety model](README.md#safety-model)).
Anything that breaks one of them is a vulnerability. The withdrawal block and the
trading kill-switch are enforced in `src/binance_mcp/client.py` on every request,
before anything is signed, so no tool can skip them.

**In scope**

- **Reaching a withdrawal or fiat-rail endpoint** through any tool, under any
  configuration — the client refuses these paths even when the key could call them.
- **Moving funds or changing account state with trading off** — placing, cancelling or
  replacing orders, transfers, convert, TWAP or dust conversion while
  `BINANCE_ALLOW_TRADING` is unset. The order dry-runs and a short allowlist of
  read-only POST endpoints are the only exceptions.
- **Credential exposure** — the API key, HMAC secret, private key or its passphrase
  appearing in tool output or error messages, or being sent anywhere other than the
  configured `BINANCE_API_URL`.
- **Signing or parameter smuggling** — a request whose transmitted parameters differ
  from what was signed, or a tool input that adds parameters its schema does not declare.
- **False confirmations** — a tool reporting an order as live, a TWAP as executed or a
  convert as done when Binance's response says otherwise, or reporting success when the
  outcome is unknown. These lead to duplicate orders, so they are treated as security bugs.
- **Supply chain** — the `amazing-binance-mcp` package on PyPI, the workflows in
  `.github/workflows/`, and the `Dockerfile`.

**Out of scope**

- Vulnerabilities in Binance itself → [Binance's bug bounty on Bugcrowd](https://bugcrowd.com/engagements/binance).
- Vulnerabilities in the MCP SDK or an MCP client (Claude Desktop, Cursor, Claude
  Code…) → the respective project.
- Attacks that need control of the machine or the server's environment: editing `.env`
  or the client config, pointing `BINANCE_API_URL` at a malicious host, reading the PEM
  file. Whoever controls the environment already holds the key.
- Actions a model takes **with trading enabled** that stay within what a tool
  documents, including prompt injection (see [Known limitations](#known-limitations)).
  Injection that crosses a rail — a withdrawal, or a trade with trading off — is in scope.
- Rate limits or IP bans caused by your own request volume, and hardening ideas with no
  demonstrated impact — open a regular issue for those.
- Similarly named packages on PyPI (`binance-mcp`, `binance-mcp-server`) are unrelated
  projects. Report a malicious one to PyPI with **Report project as malware** on its page.

## What the server does with your credentials

- The API key is sent **only** in the `X-MBX-APIKEY` header, and only to
  `BINANCE_API_URL` (default `https://api.binance.com`).
- The HMAC secret and the private key **never leave the process** — only signatures do,
  and a signed request is valid only within `BINANCE_RECV_WINDOW_MS` of its timestamp.
- Nothing is logged, and no credential is echoed in tool output.
- The server speaks MCP over **stdio only**: it opens no port and has no HTTP/SSE
  transport. Its only outbound connections go to the REST base URL.

## Known limitations

- **Prompt injection through Binance data.** Some tool output contains text chosen by
  other people — for example the counterparty names in `binance_get_pay_history` and
  `binance_get_pay_transactions`. Anyone can send you a small Pay transfer under a name
  that reads like an instruction to the model. With trading off, the worst outcome is a
  misleading answer. With trading on, the model could be talked into placing or
  cancelling orders, transfers or converts inside your account — never a withdrawal.
- **The kill-switch is all or nothing.** `BINANCE_ALLOW_TRADING=1` unlocks every 🔒 tool;
  the server has no per-tool, per-symbol or amount limits. What remains are the key's
  own permissions and your MCP client's approval prompts.

## Running it safely

- **Scope the key.** Enable Reading; add Spot & Margin Trading only if you will trade;
  keep **Enable Withdrawals off** and **restrict the key to your IP**.
  `binance_health_check` warns when withdrawals are on or the IP allowlist is missing.
- **Prefer an Ed25519 key.** Keep the PEM outside any repository or synced folder,
  `chmod 600` it, and encrypt it (`BINANCE_PRIVATE_KEY_PASSPHRASE`) on a shared machine.
- **Leave trading off** unless a client needs it, and then set `BINANCE_ALLOW_TRADING=1`
  only in that client's `env`. Try new flows on the testnet first (`BINANCE_TESTNET=1`).
- **Keep approval prompts on** for the 🔒 tools in your MCP client, above all with
  trading enabled — they are the human check against prompt injection.
- **Guard your client config.** MCP client config files hold the `env` block in plain
  text; keep them out of dotfile repositories and shared backups. (`.env` and `*.pem`
  are git-ignored in this repository.)
- **Point `BINANCE_API_URL` only at Binance hosts** — the key header goes wherever it
  points.
- **If a key leaks, delete it first** in Binance → API Management, then create a new one.
- **Install the right package:** `amazing-binance-mcp`, from PyPI or this repository.

## How releases are protected

- Releases are built and uploaded by GitHub Actions through **PyPI Trusted Publishing**
  (OIDC): the workflows use no stored PyPI token, and the publish job holds only
  `id-token: write` and `contents: read`.
- The CI workflow runs with a read-only `GITHUB_TOKEN`.
- **Secret scanning with push protection** is enabled, so a push containing a
  recognizable credential is blocked.
- The Docker image runs as an unprivileged user (UID 10001).
