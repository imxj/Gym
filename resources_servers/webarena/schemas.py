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

from typing import Dict, List, Optional

from pydantic import ConfigDict, Field

from nemo_gym.config_types import ModelServerRef
from resources_servers.browser_gym.schemas import (
    BrowserGymResourcesServerConfig,
    CUASeedSessionRequest,
    CUAVerifyResponse,
)


# Canonical WebArena benchmark demo accounts (public in the upstream benchmark).
DEFAULT_SITE_CREDENTIALS: Dict[str, Dict[str, str]] = {
    "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "classifieds": {"username": "blake.sullivan@gmail.com", "password": "Password.123"},
}


# Termination reasons that make the reward unreliable: the failure is the
# environment's (or the eval infrastructure's), not the policy's, so the RL
# trainer should drop the sample instead of learning from a bogus 0.
# Deliberately NOT masked: "max_steps" (running out of steps without finishing
# is a genuine failure, matching the standalone harness's "fail" status).
DEFAULT_MASKED_TERMINATION_REASONS: List[str] = [
    # agent/browser infrastructure
    "browser_stuck",
    "adapter_error",
    "adapter_init_error",
    "run_timeout",
    "empty_trajectory",
    # model protocol exhaustion (harness parity: status=error, eval skipped)
    "no_tool_calls",
    "unparseable_action",
    # verify-side infrastructure
    "judge_unavailable",
    "judge_call_failed",
    "program_html_infra_error",
    "site_api_error",
    "verification_error",
]


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
    # Termination reasons whose rewards are masked for the RL trainer.
    masked_termination_reasons: List[str] = Field(default_factory=lambda: list(DEFAULT_MASKED_TERMINATION_REASONS))


class WebArenaSeedSessionRequest(CUASeedSessionRequest):
    """start_url may contain site placeholders (e.g. "__SHOPPING_ADMIN__")."""


class WebArenaVerifyResponse(CUAVerifyResponse):
    """CUA verify response plus the unreliable-reward contract.

    mask_sample=True means "do not train on this reward" — the episode or its
    verification failed for an infrastructure reason listed in
    ``masked_termination_reasons`` (the formalization of the standalone
    harness's "status=error, evaluation skipped").
    """

    model_config = ConfigDict(extra="allow")

    mask_sample: bool = False
    termination_reason: Optional[str] = None
