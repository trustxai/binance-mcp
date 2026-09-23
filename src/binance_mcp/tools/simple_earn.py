"""Simple Earn read-only tools: Flexible/Locked positions and the account summary.

Scope per `.memory/research/02-endpoint-inventory.md` §L: three SIGNED, read-only,
USER_DATA endpoints under `/sapi/v1/simple-earn/`. There is no subscribe/redeem tool
here — this module only reports state.

Inventory caveat (S1-only): the endpoint shapes and the IP weight (150 for all three)
come from the `binance-api-swagger` spec (S1, last pushed 2024-10). Binance's current
docs site renders the `/docs/simple_earn/...` pages EMPTY (client-side JS landing
page), so none of this could be cross-checked against S2/S3. Treat the weight and the
exact response field set as approximate until verified live (see `binance_health_check`
for confirming credentials work at all, and t15-live-smoke for the real cross-check).

Pagination note: Binance paginates `flexible/position` and `locked/position` by a
1-based page number (`current`) + page size (`size`, default 10, max 100) — not a row
offset. To keep the tool surface consistent with the rest of this server (house style:
`limit`/`offset`), this module exposes `limit` (mapped to `size`) and `offset`
(house-style, converted to `current = offset // limit + 1`). Pass `offset` as an exact
multiple of `limit` for precise paging; a non-aligned offset is rounded down to the
start of the page that contains it.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, field_validator

from binance_mcp.client import get_client
from binance_mcp.errors import handle_api_error
from binance_mcp.formatters import (
    ResponseFormat,
    clip_response,
    epoch_to_human,
    fmt_num,
    paginated_response,
    to_json,
)
from binance_mcp.server import mcp

_ASSET_PATTERN = r"^[A-Z0-9]{2,20}$"


def _fmt_pct(value: Any) -> str:
    """Render a Binance decimal fraction ("0.05") as a percentage string ("5%")."""
    if value in (None, ""):
        return "N/A"
    try:
        pct = Decimal(str(value)) * 100
    except InvalidOperation:
        return str(value)
    return f"{fmt_num(pct)}%"


def _fmt_tier_apr(value: Any) -> str:
    """Render the tiered-APR map some Flexible products return (e.g. {"0-5BTC": 0.05})."""
    if not isinstance(value, dict) or not value:
        return "N/A"
    return "; ".join(f"{tier} {_fmt_pct(rate)}" for tier, rate in value.items())


def _format_flexible_position(row: dict[str, Any]) -> str:
    asset = row.get("asset", "?")
    lines = [
        f"- **{asset}** — product `{row.get('productId', '?')}`: amount {fmt_num(row.get('totalAmount'))}, "
        f"latest APR {_fmt_pct(row.get('latestAnnualPercentageRate'))}",
        f"  tiered APR: {_fmt_tier_apr(row.get('tierAnnualPercentageRate'))}",
        f"  yesterday airdrop rate: {_fmt_pct(row.get('yesterdayAirdropPercentageRate'))}, "
        f"redeemable: {bool(row.get('canRedeem'))}, auto-subscribe: {bool(row.get('autoSubscribe'))}",
    ]
    if row.get("airDropAsset"):
        lines.append(f"  airdrop asset: {row['airDropAsset']}, collateral: {fmt_num(row.get('collateralAmount'))}")
    lines.append(
        f"  rewards — yesterday {fmt_num(row.get('yesterdayRealTimeRewards'))}, "
        f"cumulative real-time {fmt_num(row.get('cumulativeRealTimeRewards'))}, "
        f"cumulative bonus {fmt_num(row.get('cumulativeBonusRewards'))}, "
        f"cumulative total {fmt_num(row.get('cumulativeTotalRewards'))}"
    )
    return "\n".join(lines)


def _format_locked_position(row: dict[str, Any]) -> str:
    asset = row.get("asset", "?")
    lines = [
        f"- **{asset}** — position `{row.get('positionId', '?')}` (project `{row.get('projectId', '?')}`): "
        f"amount {fmt_num(row.get('amount'))}, APY {_fmt_pct(row.get('APY'))}",
        f"  duration {row.get('duration', '?')}d, accrued {row.get('accrualDays', '?')}d, "
        f"reward asset {row.get('rewardAsset', '?')}",
        f"  purchased {epoch_to_human(row.get('purchaseTime'))}, redeem date {epoch_to_human(row.get('redeemDate'))}",
        f"  renewable: {bool(row.get('isRenewable'))}, auto-renew: {bool(row.get('isAutoRenew'))}",
    ]
    return "\n".join(lines)


class _EarnFlexiblePositionsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    asset: str | None = Field(
        default=None,
        pattern=_ASSET_PATTERN,
        description="Filter by asset, e.g. USDT. Case-insensitive; normalized to uppercase.",
    )
    product_id: str | None = Field(default=None, description="Filter by a specific Flexible product id (e.g. USDT001).")
    limit: int = Field(default=10, ge=1, le=100, description="Rows per page (Binance `size`); max 100.")
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Zero-based row offset. Binance paginates by 1-based page number (`current`), not a "
            "true offset — this tool converts as `current = offset // limit + 1`. Pass a multiple "
            "of `limit` for exact paging; other values round down to the containing page."
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: markdown (default) or json."
    )

    @field_validator("asset", mode="before")
    @classmethod
    def _upper_asset(cls, value: str | None) -> str | None:
        return value.upper() if isinstance(value, str) else value


class _EarnLockedPositionsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    asset: str | None = Field(
        default=None,
        pattern=_ASSET_PATTERN,
        description="Filter by asset, e.g. AXS. Case-insensitive; normalized to uppercase.",
    )
    position_id: str | None = Field(default=None, description="Filter by a specific Locked position id.")
    project_id: str | None = Field(default=None, description="Filter by a specific Locked project id (e.g. Axs*90).")
    limit: int = Field(default=10, ge=1, le=100, description="Rows per page (Binance `size`); max 100.")
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Zero-based row offset. Binance paginates by 1-based page number (`current`), not a "
            "true offset — this tool converts as `current = offset // limit + 1`. Pass a multiple "
            "of `limit` for exact paging; other values round down to the containing page."
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: markdown (default) or json."
    )

    @field_validator("asset", mode="before")
    @classmethod
    def _upper_asset(cls, value: str | None) -> str | None:
        return value.upper() if isinstance(value, str) else value


class _EarnAccountInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: markdown (default) or json."
    )


@mcp.tool(
    name="binance_get_earn_flexible_positions",
    annotations=ToolAnnotations(
        title="Binance Simple Earn — Flexible Positions",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_earn_flexible_positions(params: _EarnFlexiblePositionsInput) -> str:
    """List the caller's Simple Earn Flexible subscriptions and their live APR.

    Calls `GET /sapi/v1/simple-earn/flexible/position` (SIGNED, USER_DATA). IP weight
    150 per call (S1 spec, 2024-10 — Binance's docs site renders the Simple Earn pages
    empty, so this weight was not re-verified; treat it as approximate).

    When to Use:
    - To see which Flexible products you are subscribed to, how much is deposited,
      and the current (and tiered, when Binance returns tiers) annual percentage rate.
    - Before deciding whether to redeem or top up a Flexible position.

    When NOT to Use:
    - For time-locked Simple Earn products — use `binance_get_earn_locked_positions`.
    - For a single aggregate balance across all Earn products — use `binance_get_earn_account`.

    Returns:
    A markdown list (or JSON with `response_format="json"`) of positions: asset,
    product id, deposited amount, latest APR, tiered APR breakdown (when present),
    yesterday's airdrop rate, redeemability, auto-subscribe flag, and cumulative
    rewards (yesterday, real-time, bonus, total).

    Pagination:
    `limit` (Binance `size`, max 100, default 10) and `offset` (house-style; see the
    module docstring for the `offset` -> `current` page-number mapping). `total` from
    Binance drives `has_more`.

    Examples:
    params = {"asset": "USDT", "limit": 20}
    params = {"product_id": "BTC001"}

    Error Handling:
    -2015 means the key lacks Simple Earn / USER_DATA permission, or this machine's
    IP is not on the key's allowlist. An empty `rows` list means no Flexible
    subscriptions exist (or the asset/product_id filter matched nothing).
    """
    try:
        client = get_client()
        current_page = params.offset // params.limit + 1
        query: dict[str, Any] = {
            "asset": params.asset,
            "productId": params.product_id,
            "current": current_page,
            "size": params.limit,
        }
        resp = await client.request("GET", "/sapi/v1/simple-earn/flexible/position", params=query, auth="signed")
        data = resp.json()
        rows = data.get("rows", [])
        return paginated_response(
            items=rows,
            limit=params.limit,
            offset=params.offset,
            fmt=params.response_format,
            item_formatter=_format_flexible_position,
            title="Simple Earn — Flexible Positions",
            total=data.get("total"),
        )
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_earn_locked_positions",
    annotations=ToolAnnotations(
        title="Binance Simple Earn — Locked Positions",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_earn_locked_positions(params: _EarnLockedPositionsInput) -> str:
    """List the caller's Simple Earn Locked subscriptions and their APY/redeem dates.

    Calls `GET /sapi/v1/simple-earn/locked/position` (SIGNED, USER_DATA). IP weight
    150 per call (S1 spec, 2024-10 — same unverified caveat as
    `binance_get_earn_flexible_positions`; treat it as approximate).

    When to Use:
    - To see time-locked Earn positions: duration, accrued days, APY, and redeem date.
    - Before deciding whether to let a position auto-renew or opt out.

    When NOT to Use:
    - For on-demand redeemable positions — use `binance_get_earn_flexible_positions`.
    - For a single aggregate balance — use `binance_get_earn_account`.

    Returns:
    A markdown list (or JSON) of positions: asset, position id, project id, amount,
    APY, duration/accrued days, reward asset, purchase and redeem dates, and the
    renewable/auto-renew flags.

    Pagination:
    Same `limit`/`offset` -> Binance `size`/`current` mapping as
    `binance_get_earn_flexible_positions` (see the module docstring).

    Examples:
    params = {"asset": "AXS"}
    params = {"position_id": "123123"}

    Error Handling:
    -2015 as above (permission or IP allowlist). An empty `rows` list means no
    Locked subscriptions match the filters.
    """
    try:
        client = get_client()
        current_page = params.offset // params.limit + 1
        query: dict[str, Any] = {
            "asset": params.asset,
            "positionId": params.position_id,
            "projectId": params.project_id,
            "current": current_page,
            "size": params.limit,
        }
        resp = await client.request("GET", "/sapi/v1/simple-earn/locked/position", params=query, auth="signed")
        data = resp.json()
        rows = data.get("rows", [])
        return paginated_response(
            items=rows,
            limit=params.limit,
            offset=params.offset,
            fmt=params.response_format,
            item_formatter=_format_locked_position,
            title="Simple Earn — Locked Positions",
            total=data.get("total"),
        )
    except Exception as exc:
        return handle_api_error(exc)


@mcp.tool(
    name="binance_get_earn_account",
    annotations=ToolAnnotations(
        title="Binance Simple Earn — Account Summary",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def binance_get_earn_account(params: _EarnAccountInput) -> str:
    """Summarize total Simple Earn holdings (Flexible + Locked) in BTC and USDT.

    Calls `GET /sapi/v1/simple-earn/account` (SIGNED, USER_DATA). IP weight 150 per
    call (S1 spec, 2024-10 — same unverified caveat as the position tools above).

    When to Use:
    - For a one-call snapshot of total Earn value without listing every position.

    When NOT to Use:
    - To see individual products or positions — use
      `binance_get_earn_flexible_positions` / `binance_get_earn_locked_positions`.

    Returns:
    A markdown block (or JSON with `response_format="json"`) with total, flexible-only,
    and locked-only amounts, each in BTC and USDT.

    Examples:
    params = {}
    params = {"response_format": "json"}

    Error Handling:
    -2015 means the key lacks Simple Earn / USER_DATA permission or this machine's IP
    is not on the key's allowlist. All-zero amounts mean no funds are in Simple Earn.
    """
    try:
        client = get_client()
        resp = await client.request("GET", "/sapi/v1/simple-earn/account", auth="signed")
        data = resp.json()
        if params.response_format is ResponseFormat.JSON:
            return clip_response(to_json(data))
        lines = [
            "# Simple Earn — Account Summary",
            "",
            f"- **total**: {fmt_num(data.get('totalAmountInBTC'))} BTC / {fmt_num(data.get('totalAmountInUSDT'))} USDT",
            f"- **flexible**: {fmt_num(data.get('totalFlexibleAmountInBTC'))} BTC / "
            f"{fmt_num(data.get('totalFlexibleAmountInUSDT'))} USDT",
            f"- **locked**: {fmt_num(data.get('totalLockedInBTC'))} BTC / "
            f"{fmt_num(data.get('totalLockedInUSDT'))} USDT",
        ]
        return clip_response("\n".join(lines))
    except Exception as exc:
        return handle_api_error(exc)
