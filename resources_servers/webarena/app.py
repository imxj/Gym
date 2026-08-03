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
"""WebArena resources server.

Extends the generic browser_gym CUA server (browser pool, /step, /close) with:

- seed_session: substitutes __SHOPPING__-style placeholders in start_url and
  authenticates the fresh browser context against the configured WebArena
  sites (UI login once per site, cookie-cached for subsequent sessions).
- verify: classic WebArena evaluation — string_match (with LLM-judge
  fuzzy/fallback via an optional judge model server), url_match against the
  trajectory's visited URLs, and program_html evaluated live against the
  sites in a fresh authenticated context.

The reward is the product of all eval_types (0.0 or 1.0 for classic tasks).
Ported from osworld_internal webarena/common/classic_evaluation.py, including
the `func:` site-API helpers (see site_api_helpers.py).
"""

from __future__ import annotations

import asyncio
import base64
import html as html_lib
import logging
import uuid
from typing import Any, Dict, List, Optional

from pydantic import PrivateAttr

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json
from resources_servers.browser_gym.app import BrowserGymResourcesServer
from resources_servers.browser_gym.schemas import (
    CUASeedSessionResponse,
    CUAVerifyRequest,
)
from resources_servers.webarena.evaluators import (
    JUDGE_SYSTEM_MESSAGE,
    judge_passed,
    score_program_html_required,
    score_url_match_candidates,
    string_match_local,
    substitute_site_placeholders,
    trajectory_candidate_urls,
)
from resources_servers.webarena.schemas import (
    WebArenaResourcesServerConfig,
    WebArenaSeedSessionRequest,
    WebArenaVerifyResponse,
)
from resources_servers.webarena.site_api_helpers import (
    WebArenaSiteAPI,
    resolve_helper_expression,
)


logger = logging.getLogger(__name__)


class WebArenaResourcesServer(BrowserGymResourcesServer):
    config: WebArenaResourcesServerConfig

    # site -> cookie list captured after a successful UI login
    _auth_cookies: Dict[str, List[dict]] = PrivateAttr(default_factory=dict)
    _login_locks: Dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _site_api: Optional[WebArenaSiteAPI] = PrivateAttr(default=None)

    def _get_site_api(self) -> WebArenaSiteAPI:
        if self._site_api is None:
            self._site_api = WebArenaSiteAPI(self.config.site_urls, self.config.site_credentials)
        return self._site_api

    ########################################
    # Seeding: placeholder substitution + site login
    ########################################

    async def seed_session(self, body: WebArenaSeedSessionRequest) -> CUASeedSessionResponse:
        env_id = str(uuid.uuid4())
        start_url = substitute_site_placeholders(body.start_url or "about:blank", self.config.site_urls)

        await self.browser_pool.create_session(
            env_id=env_id,
            start_url="about:blank",
            viewport_width=body.viewport_width,
            viewport_height=body.viewport_height,
        )
        try:
            session = self.browser_pool.get_session(env_id)
            if self.config.login_sites_on_seed:
                await self._authenticate_context(session.context)
            await session.page.goto(start_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(1)
            screenshot_bytes = await session.page.screenshot(type="png", full_page=False)
            screenshot = base64.b64encode(screenshot_bytes).decode("utf-8")
            return CUASeedSessionResponse(env_id=env_id, screenshot=screenshot)
        except Exception:
            await self.browser_pool.close_session(env_id)
            raise

    def _login_lock(self, site: str) -> asyncio.Lock:
        if site not in self._login_locks:
            self._login_locks[site] = asyncio.Lock()
        return self._login_locks[site]

    async def _authenticate_context(self, context) -> None:
        """Authenticate a fresh browser context against every configured site.

        First login per site is a scripted UI login whose cookies are cached;
        later contexts just replay the cached cookies. Login failures are
        logged and skipped so one broken site does not take down seeding.
        """
        for site, url in self.config.site_urls.items():
            if site not in self.config.site_credentials:
                continue
            try:
                cookies = self._auth_cookies.get(site)
                if cookies is None:
                    async with self._login_lock(site):
                        cookies = self._auth_cookies.get(site)
                        if cookies is None:
                            await self._login_site(context, site, url)
                            cookies = await context.cookies(url)
                            self._auth_cookies[site] = cookies
                            continue  # this context is already logged in
                if cookies:
                    await context.add_cookies(cookies)
            except Exception as e:
                logger.warning("WebArena login failed for site=%s: %s", site, e)

    async def _login_site(self, context, site: str, url: str) -> None:
        """Scripted UI login flows, ported from the standalone harness."""
        creds = self.config.site_credentials.get(site)
        if not creds:
            return
        url = url.rstrip("/")
        page = await context.new_page()
        try:
            if site == "reddit":
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.get_by_role("link", name="Log in").click()
                await page.get_by_label("Username").fill(creds["username"])
                await page.get_by_label("Password").fill(creds["password"])
                await page.get_by_role("button", name="Log in").click()
            elif site == "gitlab":
                await page.goto(f"{url}/users/sign_in", wait_until="domcontentloaded", timeout=30000)
                await page.get_by_label("Username or email").fill(creds["username"])
                await page.get_by_label("Password").fill(creds["password"])
                await page.get_by_role("button", name="Sign in").click()
            elif site == "shopping":
                await page.goto(f"{url}/customer/account/login/", wait_until="domcontentloaded", timeout=30000)
                await page.get_by_label("Email", exact=True).fill(creds["username"])
                await page.get_by_label("Password", exact=True).fill(creds["password"])
                await page.get_by_role("button", name="Sign In").click()
            elif site == "shopping_admin":
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.get_by_label("Username").fill(creds["username"])
                await page.get_by_label("Password").fill(creds["password"])
                await page.get_by_role("button", name="Sign in").click()
            elif site == "classifieds":
                await page.goto(f"{url}/index.php?page=login", wait_until="domcontentloaded", timeout=30000)
                await page.locator("#email").fill(creds["username"])
                await page.locator("#password").fill(creds["password"])
                await page.get_by_role("button", name="Log in").click()
            else:
                return  # wikipedia / map need no login
            await asyncio.sleep(self.config.login_settle_seconds)
        finally:
            await page.close()

    ########################################
    # Verification
    ########################################

    async def verify(self, body: CUAVerifyRequest) -> WebArenaVerifyResponse:
        vm = body.verifier_metadata or {}
        eval_cfg: Dict[str, Any] = vm.get("eval") or {}
        intent = str(vm.get("intent") or "")
        detail: Dict[str, Any] = {"messages": []}
        masked = set(self.config.masked_termination_reasons)
        # Verify-side infra failures collected while scoring (judge down,
        # site unreachable, ...). Any masked entry marks the reward unreliable.
        flags: List[str] = []

        def respond(reward: float, termination_reason: Optional[str]) -> WebArenaVerifyResponse:
            mask_sample = termination_reason in masked
            if mask_sample:
                detail["messages"].append(
                    f"mask_sample=True ({termination_reason}): reward is unreliable, trainer should drop it"
                )
            payload = body.model_dump()
            # Replayed rollout rows may already carry these as extras.
            for key in ("reward", "mask_sample", "termination_reason", "verification_result"):
                payload.pop(key, None)
            return WebArenaVerifyResponse(
                **payload,
                reward=reward,
                mask_sample=mask_sample,
                termination_reason=termination_reason,
                verification_result=detail,
            )

        try:
            agent_reason = self._agent_termination_reason(body)
            if agent_reason:
                detail["messages"].append(f"agent termination_reason: {agent_reason}")
            if agent_reason in masked:
                # The episode itself is unreliable (browser stuck, adapter
                # error, parse exhaustion, ...). Skip scoring entirely — the
                # standalone harness's "status=error, evaluation skipped".
                detail["messages"].append("scoring skipped for masked termination reason")
                return respond(0.0, agent_reason)

            eval_types = eval_cfg.get("eval_types") or []
            if not eval_types:
                # A row that cannot be scored at all is a data problem, not a
                # policy failure — mask it.
                detail["messages"].append("no eval_types in verifier_metadata.eval")
                return respond(0.0, "verification_error")

            answer = self._final_answer(body)
            candidate_urls = self._candidate_urls(body)
            detail["answer"] = answer
            detail["candidate_urls"] = candidate_urls

            score = 1.0
            # (eval_type, score, had_infra_flags) — needed for causal masking.
            components: List[tuple] = []
            for eval_type in eval_types:
                flags_before = len(flags)
                if eval_type == "string_match":
                    cur_score = await self._string_match(eval_cfg, intent, answer, detail, flags)
                elif eval_type == "url_match":
                    cur_score, matched_url, unique_urls = score_url_match_candidates(
                        eval_cfg, candidate_urls, self.config.site_urls
                    )
                    detail["messages"].append(
                        f"url_match: score={cur_score}, pred={matched_url!r}, candidates={unique_urls!r}"
                    )
                elif eval_type == "program_html":
                    cur_score = await self._program_html(eval_cfg, candidate_urls, detail, flags)
                else:
                    detail["messages"].append(f"unknown eval_type: {eval_type}")
                    cur_score = 0.0
                components.append((eval_type, cur_score, len(flags) > flags_before))
                score *= cur_score

            # Causal masking: an infra flag only makes the reward unreliable if
            # it could have determined the outcome. A full-score episode was
            # scored despite the hiccup; an episode with a clean (unflagged)
            # zero component failed genuinely regardless of the infra issue.
            verify_reason = None
            if score < 1.0 and not any(s < 1.0 and not flagged for _, s, flagged in components):
                verify_reason = next((flag for flag in flags if flag in masked), None)
            elif flags:
                detail["messages"].append(
                    f"infra flags {flags!r} did not determine the outcome — reward kept unmasked"
                )
            return respond(float(score), verify_reason or agent_reason)
        except Exception as e:
            logger.error("WebArena verification failed: %s: %s", type(e).__name__, e, exc_info=True)
            detail["messages"].append(f"verification error: {type(e).__name__}: {e}")
            return respond(0.0, "verification_error")

    @staticmethod
    def _final_answer(body: CUAVerifyRequest) -> str:
        response = body.response
        trajectory = getattr(response, "trajectory", None)
        final_message = getattr(trajectory, "final_message", None)
        return "" if final_message is None else str(final_message)

    @staticmethod
    def _agent_termination_reason(body: CUAVerifyRequest) -> Optional[str]:
        """The agent-reported reason the episode ended abnormally, if any."""
        trajectory = getattr(body.response, "trajectory", None)
        reason = getattr(trajectory, "termination_reason", None)
        if reason:
            return str(reason)
        # An episode with no steps and no answer never really started. An
        # empty-string answer (terminate with no answer) still counts as an
        # answer — only None means the model never responded.
        steps = getattr(trajectory, "steps", None) or []
        if not steps and getattr(trajectory, "final_message", None) is None:
            return "empty_trajectory"
        return None

    @staticmethod
    def _candidate_urls(body: CUAVerifyRequest) -> List[str]:
        trajectory = getattr(body.response, "trajectory", None)
        steps = list(getattr(trajectory, "steps", None) or [])
        return trajectory_candidate_urls([getattr(step, "current_url", "") or "" for step in steps])

    ########################################
    # string_match (+ LLM judge)
    ########################################

    async def _string_match(
        self, eval_cfg: Dict[str, Any], intent: str, answer: str, detail: Dict[str, Any], flags: List[str]
    ) -> float:
        score, pending = string_match_local(eval_cfg, intent, answer)
        for judge_request in pending:
            score *= await self._resolve_judge_request(judge_request, detail, flags)
        detail["messages"].append(f"string_match: score={score}, pending_judge_requests={len(pending)}")
        return score

    async def _resolve_judge_request(
        self, judge_request: Dict[str, str], detail: Dict[str, Any], flags: List[str]
    ) -> float:
        if self.config.judge_model_server is None:
            detail["messages"].append(
                f"judge unavailable — {judge_request['judge_type']} scored 0.0 "
                "(configure judge_model_server to enable fuzzy matching)"
            )
            flags.append("judge_unavailable")
            return 0.0

        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(role="system", content=JUDGE_SYSTEM_MESSAGE),
                NeMoGymEasyInputMessage(role="user", content=judge_request["user_message"]),
            ],
            max_output_tokens=self.config.judge_max_output_tokens,
        )
        try:
            response = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=responses_create_params,
            )
            judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
            output_text = self._response_output_text(judge_response)
        except Exception as e:
            logger.warning("WebArena judge call failed: %s", e)
            detail["messages"].append(f"judge call failed ({judge_request['judge_type']}): {e}")
            flags.append("judge_call_failed")
            return 0.0

        passed = judge_passed(judge_request["judge_type"], output_text)
        detail.setdefault("judge_evaluations", []).append(
            {
                "judge_type": judge_request["judge_type"],
                "reference": judge_request["reference"],
                "prediction": judge_request["prediction"],
                "response": output_text,
                "passed": bool(passed),
            }
        )
        return passed

    @staticmethod
    def _response_output_text(response: NeMoGymResponse) -> str:
        texts: List[str] = []
        for output in response.output or []:
            if getattr(output, "type", None) != "message":
                continue
            for content in getattr(output, "content", None) or []:
                if getattr(content, "type", None) == "output_text":
                    texts.append(getattr(content, "text", "") or "")
        return "\n".join(texts)

    ########################################
    # program_html — live-site DOM checks
    ########################################

    async def _program_html(
        self, eval_cfg: Dict[str, Any], candidate_urls: List[str], detail: Dict[str, Any], flags: List[str]
    ) -> float:
        targets = eval_cfg.get("program_html") or []
        if not targets:
            return 1.0

        env_id = f"webarena-verify-{uuid.uuid4()}"
        await self.browser_pool.create_session(env_id=env_id, start_url="about:blank")
        try:
            session = self.browser_pool.get_session(env_id)
            await self._authenticate_context(session.context)
            page = session.page

            score = 1.0
            for target in targets:
                target_url = target.get("url", "")

                if str(target_url).startswith("func:"):
                    last_url = candidate_urls[0] if candidate_urls else "about:blank"
                    try:
                        target_url = str(
                            await resolve_helper_expression(str(target_url), self._get_site_api(), page, last_url)
                        )
                        detail["messages"].append(f"program_html: resolved func url -> {target_url!r}")
                    except Exception as e:
                        # Site-API infra failure (Magento REST down, auth token
                        # rejected, ...) — the check could not run at all.
                        detail["messages"].append(f"program_html: func url resolution failed: {e} — scored 0.0")
                        flags.append("site_api_error")
                        score *= 0.0
                        continue

                if target_url == "last":
                    urls = candidate_urls or []
                    target_score = 0.0
                    candidate_errors = 0
                    for candidate in urls:
                        try:
                            target_score = max(
                                target_score, await self._program_html_target(page, candidate, target, detail)
                            )
                            if target_score == 1.0:
                                break
                        except Exception as e:
                            detail["messages"].append(f"program_html candidate {candidate!r} failed: {e}")
                            candidate_errors += 1
                    if not urls:
                        detail["messages"].append("program_html: url='last' but trajectory has no URLs")
                    # A candidate hiccup only matters if the target never passed
                    # afterwards — otherwise the infra failure was inconsequential.
                    if candidate_errors and target_score < 1.0:
                        flags.append("program_html_infra_error")
                    score *= target_score
                else:
                    resolved = substitute_site_placeholders(target_url, self.config.site_urls)
                    try:
                        score *= await self._program_html_target(page, resolved, target, detail)
                    except Exception as e:
                        # Navigation/extraction infrastructure failure — a
                        # content mismatch never raises, so this is not a
                        # genuine task failure.
                        detail["messages"].append(f"program_html target {resolved!r} failed: {e} — scored 0.0")
                        flags.append("program_html_infra_error")
                        score *= 0.0
            return score
        finally:
            await self.browser_pool.close_session(env_id)

    async def _program_html_target(self, page, url: str, target: Dict[str, Any], detail: Dict[str, Any]) -> float:
        await page.goto(url, wait_until="domcontentloaded", timeout=self.config.verify_navigation_timeout_ms)
        await asyncio.sleep(self.config.program_html_wait_seconds)

        locator = str(target.get("locator") or "")
        if not locator.strip():
            selected_element = await page.content()
        elif locator.startswith("document.") or locator.startswith("[...document."):
            for prep_action in target.get("prep_actions") or []:
                try:
                    await page.evaluate(f"() => {prep_action}")
                except Exception:
                    logger.debug("program_html prep_action failed: %s", prep_action, exc_info=True)
            try:
                selected_element = str(await page.evaluate(f"() => {locator}") or "")
            except Exception:
                selected_element = ""
        elif locator.startswith("func:"):
            selected_element = str(await resolve_helper_expression(locator, self._get_site_api(), page, page.url))
        else:
            raise ValueError(f"Unknown program_html locator: {locator}")

        selected_element = html_lib.unescape(selected_element)
        target_score = score_program_html_required(target["required_contents"], selected_element)
        detail["messages"].append(
            f"program_html: url={url!r} locator={locator!r} score={target_score} extracted={selected_element[:200]!r}"
        )
        return target_score


if __name__ == "__main__":
    WebArenaResourcesServer.run_webserver()
