# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""WebArena `func:` site-API helpers.

Classic WebArena program_html targets reference helper functions like
`func:shopping_get_latest_order_url()` (URL resolution) or
`func:gitlab_get_project_memeber_role(__page__, "byteblaze")` (content
extraction). This module ports them from the standalone harness
(osworld_internal webarena/common/classic_evaluation.py) with two changes:

- Site URLs / credentials come from the server config, not env vars.
- HTTP goes through NeMo-Gym's global aiohttp client (never httpx/requests).

`resolve_helper_expression()` evaluates the restricted `func:` call syntax via
`ast` (no `eval`): a single call to a known helper with literal arguments,
plus the `__page__` / `__last_url__` placeholders.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import urllib.parse
from typing import Any, Dict, Optional

from nemo_gym.server_utils import request


logger = logging.getLogger(__name__)

HELPER_NAMES = frozenset(
    {
        "shopping_get_latest_order_url",
        "shopping_admin_get_cart_price_rule",
        "shopping_get_sku_latest_review_author",
        "shopping_get_sku_latest_review_rating",
        "reddit_get_post_url",
        "gitlab_get_project_memeber_role",
    }
)


def expression_uses_helpers(expr: Any) -> bool:
    text = str(expr or "")
    return text.startswith("func:") or any(name in text for name in HELPER_NAMES)


def normalize_number_string(value: Any) -> str:
    text = str(value or "")
    if "." in text:
        return text.rstrip("0").rstrip(".")
    return text


def reddit_get_post_url(url: str) -> str:
    """Strip a Postmill comment path down to the post URL."""
    parsed = urllib.parse.urlparse(str(url))
    tok_url = parsed.path.split("/")
    if len(tok_url) < 4 or tok_url[1] != "f":
        return str(url)
    subreddit = tok_url[2]
    post_id = tok_url[3]
    return f"{parsed.scheme}://{parsed.netloc}/f/{subreddit}/{post_id}/"


class WebArenaSiteAPI:
    """Async Magento/GitLab helper calls with cached admin tokens."""

    def __init__(
        self,
        site_urls: Dict[str, str],
        site_credentials: Dict[str, Dict[str, str]],
        request_timeout_seconds: float = 60.0,
    ) -> None:
        self._site_urls = dict(site_urls)
        self._site_credentials = {k: dict(v) for k, v in site_credentials.items()}
        self._timeout = request_timeout_seconds
        self._tokens: Dict[str, str] = {}
        self._token_locks: Dict[str, asyncio.Lock] = {}

    def _site_url(self, site: str) -> str:
        value = self._site_urls.get(site)
        if not value:
            raise RuntimeError(f"site_urls[{site!r}] is required for WebArena helper evaluation")
        return value.rstrip("/")

    def _shopping_admin_api_base_url(self) -> str:
        return self._site_url("shopping_admin").removesuffix("/admin")

    async def _get_json(self, url: str, headers: Optional[dict] = None, params: Optional[dict] = None) -> Any:
        response = await request("GET", url, headers=headers, params=params)
        response.raise_for_status()
        return await response.json()

    async def _post_json(self, url: str, payload: dict) -> Any:
        response = await request("POST", url, json=payload)
        response.raise_for_status()
        return await response.json()

    async def _auth_token(self, base_url: str) -> str:
        """Magento admin token for a site base URL, cached per base URL."""
        if base_url in self._tokens:
            return self._tokens[base_url]
        lock = self._token_locks.setdefault(base_url, asyncio.Lock())
        async with lock:
            if base_url not in self._tokens:
                creds = self._site_credentials.get("shopping_admin") or {}
                token = await self._post_json(
                    f"{base_url}/rest/default/V1/integration/admin/token",
                    {"username": creds.get("username", ""), "password": creds.get("password", "")},
                )
                self._tokens[base_url] = str(token)
        return self._tokens[base_url]

    async def _shopping_headers(self) -> dict:
        token = await self._auth_token(self._site_url("shopping"))
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def _shopping_admin_headers(self) -> dict:
        token = await self._auth_token(self._shopping_admin_api_base_url())
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def shopping_get_latest_order_url(self) -> str:
        """Get the latest order URL from the shopping website."""
        params = {
            "searchCriteria[sortOrders][0][field]": "created_at",
            "searchCriteria[sortOrders][0][direction]": "DESC",
            "searchCriteria[pageSize]": "1",
        }
        data = await self._get_json(
            f"{self._site_url('shopping')}/rest/V1/orders",
            headers=await self._shopping_headers(),
            params=params,
        )
        order_id = int(data["items"][0]["increment_id"])
        return f"{self._site_url('shopping')}/sales/order/view/order_id/{order_id}/"

    async def shopping_admin_get_cart_price_rule(self, rule_name: str) -> str:
        """Return normalized cart price rule fields for a saved Magento sales rule."""
        headers = await self._shopping_admin_headers()
        base = self._shopping_admin_api_base_url()
        params = {
            "searchCriteria[filter_groups][0][filters][0][field]": "name",
            "searchCriteria[filter_groups][0][filters][0][value]": str(rule_name),
            "searchCriteria[filter_groups][0][filters][0][condition_type]": "eq",
            "searchCriteria[pageSize]": "10",
        }
        data = await self._get_json(f"{base}/rest/V1/salesRules/search", headers=headers, params=params)
        items = data.get("items", [])
        if not items:
            data = await self._get_json(
                f"{base}/rest/V1/salesRules/search",
                headers=headers,
                params={"searchCriteria[pageSize]": "100"},
            )
            items = data.get("items", [])
        if not items:
            logger.info("shopping_admin_get_cart_price_rule: no cart price rules found")
            return ""
        rule = next((item for item in items if str(item.get("name", "")).lower() == str(rule_name).lower()), None)
        if rule is None:
            logger.info(
                "shopping_admin_get_cart_price_rule: no rule named %r; available rules=%s",
                rule_name,
                [item.get("name") for item in items],
            )
            return ""
        normalized = {
            "name": rule.get("name"),
            "customer_group_ids": rule.get("customer_group_ids"),
            "simple_action": rule.get("simple_action"),
            "discount_amount": normalize_number_string(rule.get("discount_amount")),
        }
        return json.dumps(normalized, ensure_ascii=True, sort_keys=True)

    async def _shopping_sku_reviews(self, sku: str) -> list:
        return await self._get_json(
            f"{self._site_url('shopping')}/rest/V1/products/{sku}/reviews",
            headers=await self._shopping_headers(),
        )

    async def shopping_get_sku_latest_review_author(self, sku: str) -> str:
        reviews = await self._shopping_sku_reviews(sku)
        if not reviews:
            return ""
        return str(reviews[-1]["nickname"])

    async def shopping_get_sku_latest_review_rating(self, sku: str) -> str:
        reviews = await self._shopping_sku_reviews(sku)
        if not reviews:
            return ""
        return str(reviews[-1]["ratings"][0]["percent"])

    @staticmethod
    def reddit_get_post_url(url: str) -> str:
        return reddit_get_post_url(url)

    @staticmethod
    async def gitlab_get_project_memeber_role(page, account_name: str) -> str:
        # [sic] "memeber" matches the benchmark data's helper name.
        try:
            account_idx = await page.evaluate(
                f"""(() => {{
                    const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                    let index = -1;

                    for(let i = 0; i < elements.length; i++) {{
                        if(elements[i].outerText === '@{account_name}') {{
                            index = i;
                            break;
                        }}
                    }}

                    return index;
                }})()"""
            )
            return str(
                await page.evaluate(
                    f"""(() => {{
                        return document.querySelectorAll("td.col-max-role span")[{account_idx}].outerText;
                    }})()"""
                )
            )
        except Exception:
            return ""


def _literal(node: ast.AST, page: Any) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "__page__":
            return page
        raise ValueError(f"Unsupported name in helper expression: {node.id}")
    raise ValueError(f"Unsupported argument node in helper expression: {ast.dump(node)}")


async def resolve_helper_expression(expr: str, api: WebArenaSiteAPI, page: Any, last_url: str) -> Any:
    """Evaluate a `func:helper(args...)` expression from classic task data.

    Mirrors the harness semantics: strip the `func:` prefix, substitute
    `__last_url__` with the relevant page URL, then call the named helper.
    Only a single call to a known helper with literal / placeholder arguments
    is accepted (parsed with `ast`, never `eval`).
    """
    helper_expr = expr.split("func:", 1)[1] if expr.startswith("func:") else expr
    # In the benchmark data __last_url__ always appears inside a quoted string
    # (e.g. func:reddit_get_post_url('__last_url__')), so a textual replacement
    # keeps the expression parseable — same semantics as the harness.
    helper_expr = helper_expr.replace("__last_url__", last_url)

    tree = ast.parse(helper_expr.strip(), mode="eval")
    node = tree.body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"Unsupported helper expression: {expr}")
    name = node.func.id
    if name not in HELPER_NAMES:
        raise ValueError(f"Unknown helper: {name}")
    if node.keywords:
        raise ValueError(f"Keyword arguments not supported in helper expression: {expr}")

    args = [_literal(arg, page) for arg in node.args]
    result = getattr(api, name)(*args)
    if asyncio.iscoroutine(result):
        return await result
    return result
