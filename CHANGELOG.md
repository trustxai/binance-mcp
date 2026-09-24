# Changelog

## 0.1.0 (2026-09-23)


### Features

* **client:** `Repeat` marker for parameters Binance wants as repeated keys ([5758a3c](https://github.com/trustxai/binance-mcp/commit/5758a3c1b87a941fe4c0f427923bced6655a8719))
* **convert:** pairs, quotes, accept, history walk and limit orders ([87e626c](https://github.com/trustxai/binance-mcp/commit/87e626c9e9df860d2b1d95a1ee677846ddd982f3))
* **fiat:** fiat orders, payments and the since-walk history ([36b7a84](https://github.com/trustxai/binance-mcp/commit/36b7a84cf1f5468cd351e5bae0b822035db9fc36))
* **market-data:** exchange info, order book, trades, klines and tickers ([4bcbaa9](https://github.com/trustxai/binance-mcp/commit/4bcbaa94d2ceab31cb2f5089217c331e2f5f3d5f))
* **order-lists:** OCO/OTO/OTOCO placement, query, cancel and listings ([e30199f](https://github.com/trustxai/binance-mcp/commit/e30199fa0634a30e6094738bac275933f5137cb2))
* **pay:** pay transactions and the 18-month history walk ([f413660](https://github.com/trustxai/binance-mcp/commit/f413660e28f3257745cae3d41b8644ef263c4835))
* Phase 0 foundation (spine, signing client with kill-switch, stub registry, oracle, CI, release tooling) ([4aa06e3](https://github.com/trustxai/binance-mcp/commit/4aa06e3b9cf553e741f93259e43472913e88b0fd))
* **simple-earn:** flexible/locked positions and earn account ([a5f784f](https://github.com/trustxai/binance-mcp/commit/a5f784f862292a6831aff1050d1a5d21272d6bc4))
* **spot-account:** account, commission, order rate limits, prevented matches, allocations ([f695046](https://github.com/trustxai/binance-mcp/commit/f6950469694f9b6a126ed0b8612db7df4824a030))
* **spot-algo:** TWAP placement, cancel, open/historical orders and sub-orders ([40a0d17](https://github.com/trustxai/binance-mcp/commit/40a0d17922d0b3209426117e6571d027ed53b302))
* **spot-orders:** place/test/query/cancel/cancel-replace and order listings ([fff11df](https://github.com/trustxai/binance-mcp/commit/fff11dfa934476b740578c9556f7a575916c7331))
* **trade-history:** my trades, traded-symbol discovery and the all-trades walk ([b817b20](https://github.com/trustxai/binance-mcp/commit/b817b2094c20e4f17de1ab16b6bb536a8acf6591))
* **wallet-account:** account status, api restrictions, snapshot and system status ([b8e63fd](https://github.com/trustxai/binance-mcp/commit/b8e63fd24b42cd54f97392666ba17415f48a37cd))
* **wallet-asset:** funding/user assets, wallet balances, transfers, dust and asset info ([16b72a9](https://github.com/trustxai/binance-mcp/commit/16b72a97e182d11ea1a626a1f82be7194f74d633))
* **wallet-capital:** deposit/withdraw history, the since-walks, addresses and coin config ([92c05b9](https://github.com/trustxai/binance-mcp/commit/92c05b9c361f93ca4a3783c3dde219ea7395e4a7))


### Bug Fixes

* **convert:** apply review findings ([0f7fa9b](https://github.com/trustxai/binance-mcp/commit/0f7fa9b5da199ebbc64de44240a698acd5f763b5))
* **fiat:** apply review findings ([f86783e](https://github.com/trustxai/binance-mcp/commit/f86783ea44016f23c8197e6953b2268715a3da8c))
* **fiat:** report walk stops truthfully ([201e835](https://github.com/trustxai/binance-mcp/commit/201e83546490eb2e6790ccf9eed4ce127be8becc))
* **market-data:** apply review findings ([5a4dc4e](https://github.com/trustxai/binance-mcp/commit/5a4dc4ee9eaa8cf39b7634a26090d311fce20799))
* **order-lists:** apply review findings ([d4ad2a6](https://github.com/trustxai/binance-mcp/commit/d4ad2a6e873189529972394bba75a8d43e2a2835))
* **pay:** apply review findings ([6e4405e](https://github.com/trustxai/binance-mcp/commit/6e4405eebeb648736cd6f70f7f4d54243bdbe994))
* **pay:** clamp `since` inside the 18-month lookback; keep rows + cursor on a mid-walk error ([6d56995](https://github.com/trustxai/binance-mcp/commit/6d56995af34779e926d55f66259bcd5d46654455))
* **pay:** document the partial-result contract; refuse ranges older than the lookback locally ([17f6c5b](https://github.com/trustxai/binance-mcp/commit/17f6c5b2788d89145d123b5c226f498631d5678a))
* **pay:** one shape for the two empty-walk early returns; test the JSON arm ([7f08d9e](https://github.com/trustxai/binance-mcp/commit/7f08d9ee65a9ee39f3292cee01e4d0b76429462d))
* **simple-earn:** apply review findings ([19a708d](https://github.com/trustxai/binance-mcp/commit/19a708d745447e1c6933ba06ea9cc06551707a96))
* **spot-account:** apply review findings ([0e1d73b](https://github.com/trustxai/binance-mcp/commit/0e1d73be52a27c4ae3b77e5f3c0b2865ea32f97e))
* **spot-algo:** apply review findings ([5247b02](https://github.com/trustxai/binance-mcp/commit/5247b02cd389be7d8c35410aac66e884eeb0fbb2))
* **spot-orders:** apply security review findings ([e14a411](https://github.com/trustxai/binance-mcp/commit/e14a41148662e160fa1cf8e5b0910dd72ccd410f))
* **trade-history:** apply review findings ([52cdd40](https://github.com/trustxai/binance-mcp/commit/52cdd40330abc092719659cb739c69709c7ef233))
* **wallet-account:** apply review findings ([5ede577](https://github.com/trustxai/binance-mcp/commit/5ede5777b5ee465d9d4b31d5533d5e943d1e127b))
* **wallet-asset:** apply security review findings ([3e934b8](https://github.com/trustxai/binance-mcp/commit/3e934b81f8e5f3cb8aeb6b13e0fffcedd037fdcb))
* **wallet-capital:** apply review findings ([9440dd1](https://github.com/trustxai/binance-mcp/commit/9440dd1b2bf2ddd6c5239e8bf476712f32e02dc8))


### Documentation

* full README — safety model, key setup, client configs, history walks, generated tool table ([16e0e3d](https://github.com/trustxai/binance-mcp/commit/16e0e3d11dd6cbc7d1057a51c406dfe936a7eccf))
* **spot-account:** one-line summary for binance_get_allocations ([7895698](https://github.com/trustxai/binance-mcp/commit/78956982ccf62229058cce1f314421524df7c762))

## Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and releases are cut by
release-please from Conventional Commits.
