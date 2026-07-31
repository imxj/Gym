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
"""Schemas for the WebArena resources server.

Reuses the browser_gym CUA schemas for step/verify/close so the existing
browser_agent works against this server unchanged; extends only the config
(site URLs, credentials, judge model) and the seed request.
"""

from typing import Dict, Optional

from pydantic import Field

from nemo_gym.config_types import ModelServerRef
from resources_servers.browser_gym.schemas import (
    BrowserGymResourcesServerConfig,
    CUASeedSessionRequest,
)


# Canonical WebArena benchmark demo accounts (public in the upstream benchmark).
DEFAULT_SITE_CREDENTIALS: Dict[str, Dict[str, str]] = {
    "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "classifieds": {"username": "blake.sullivan@gmail.com", "password": "Password.123"},
}


class WebArenaResourcesServerConfig(BrowserGymResourcesServerConfig):
    # Site name -> base URL, e.g. {"shopping_admin": "http://<host>:7780/admin"}.
    # Referenced by __SHOPPING_ADMIN__-style placeholders in task data.
    site_urls: Dict[str, str] = Field(default_factory=dict)
    site_credentials: Dict[str, Dict[str, str]] = Field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_SITE_CREDENTIALS.items()}
    )
    # Judge model used for fuzzy_match / exact-match fallback / ua_match.
    # Optional: without it, judge-dependent checks conservatively score 0.0.
    judge_model_server: Optional[ModelServerRef] = None
    judge_max_output_tokens: int = 1024
    # Log into all configured sites (cookie-cached after the first session).
    login_sites_on_seed: bool = True
    login_settle_seconds: float = 2.0
    # program_html evaluation
    program_html_wait_seconds: float = 3.0
    verify_navigation_timeout_ms: int = 60000


class WebArenaSeedSessionRequest(CUASeedSessionRequest):
    """start_url may contain site placeholders (e.g. "__SHOPPING_ADMIN__")."""
