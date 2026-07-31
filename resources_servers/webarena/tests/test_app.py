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
"""Tests for the WebArena resources server.

Tier 0: pure evaluator functions (no server, no browser, no model).
Tier 1: verify() on a constructed server with a mocked ServerClient —
string_match and url_match paths need no Playwright browser.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import resources_servers.browser_gym.app as browser_gym_app
import resources_servers.webarena.app as webarena_app
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.browser_gym.schemas import (
    BrowserAction,
    CUANeMoGymResponse,
    CUAStep,
    CUATrajectory,
    CUAVerifyRequest,
)
from resources_servers.webarena import evaluators as ev
from resources_servers.webarena import site_api_helpers as sah
from resources_servers.webarena.app import WebArenaResourcesServer
from resources_servers.webarena.schemas import WebArenaResourcesServerConfig


SITE_URLS = {
    "gitlab": "http://wa-host:8023",
    "shopping_admin": "http://wa-host:7780/admin",
}


########################################
# Tier 0 — pure functions
########################################


def test_clean_answer_strips_quotes_and_lowercases():
    assert ev.clean_answer('  "Quest Lumaflex Band"  ') == "quest lumaflex band"
    assert ev.clean_answer(None) == ""
    assert ev.clean_answer(42) == "42"


def test_exact_match_and_alternatives():
    assert ev.exact_match(ref="Sprite", pred="sprite") == 1.0
    assert ev.exact_match(ref="Sprite", pred="sprite zero") == 0.0
    assert ev.reference_alternatives("a") == ["a"]
    assert ev.reference_alternatives(["a", "b"]) == ["a", "b"]


def test_must_include_single_char_tokenize_guard():
    # Substring semantics by default...
    assert ev.must_include(ref="a", pred="grape") == 1.0
    # ...but single-character refs are token-matched when tokenize=True.
    assert ev.must_include(ref="a", pred="grape", tokenize=True) == 0.0
    assert ev.must_include(ref="a", pred="a grape", tokenize=True) == 1.0


def test_string_match_local_exact_pass_no_judge_needed():
    eval_cfg = {"reference_answers": {"exact_match": "Quest Lumaflex Band"}}
    score, pending = ev.string_match_local(eval_cfg, "intent", "Quest Lumaflex Band")
    assert score == 1.0
    assert pending == []


def test_string_match_local_exact_fail_requests_judge_fallback():
    eval_cfg = {"reference_answers": {"exact_match": ["Sprite", "Fanta"]}}
    score, pending = ev.string_match_local(eval_cfg, "intent", "coke")
    assert score == 1.0
    assert [p["judge_type"] for p in pending] == ["exact_match_fallback"]
    assert "one of: Sprite or Fanta" in pending[0]["reference"]


def test_string_match_local_must_include_and_fuzzy():
    eval_cfg = {"reference_answers": {"must_include": ["alpha", "beta"]}}
    score, pending = ev.string_match_local(eval_cfg, "intent", "alpha and beta are here")
    assert (score, pending) == (1.0, [])

    score, pending = ev.string_match_local(eval_cfg, "intent", "only alpha")
    assert [p["judge_type"] for p in pending] == ["must_include_fallback"]

    eval_cfg = {"reference_answers": {"fuzzy_match": ["around $30"]}}
    _, pending = ev.string_match_local(eval_cfg, "intent", "roughly thirty dollars")
    assert [p["judge_type"] for p in pending] == ["fuzzy_match"]


def test_string_match_local_not_achievable():
    eval_cfg = {"reference_answers": {"fuzzy_match": "N/A"}, "string_note": "no such product"}
    score, pending = ev.string_match_local(eval_cfg, "intent", "N/A")
    assert (score, pending) == (1.0, [])

    _, pending = ev.string_match_local(eval_cfg, "intent", "cannot be done")
    assert [p["judge_type"] for p in pending] == ["ua_match"]
    assert pending[0]["reference"] == "no such product"


def test_string_match_local_unknown_approach_raises():
    with pytest.raises(ValueError):
        ev.string_match_local({"reference_answers": {"regex_match": "x"}}, "intent", "y")


def test_substitute_site_placeholders():
    assert (
        ev.substitute_site_placeholders("__GITLAB__/dashboard/todos", SITE_URLS)
        == "http://wa-host:8023/dashboard/todos"
    )
    # Unknown placeholders are left alone.
    assert ev.substitute_site_placeholders("__MAP__/x", SITE_URLS) == "__MAP__/x"


def test_url_match_gold_in_pred():
    eval_cfg = {"reference_url": "__GITLAB__/dashboard/todos"}
    assert ev.url_match(eval_cfg, "http://wa-host:8023/dashboard/todos", SITE_URLS) == 1.0
    # Extra query params in the prediction are fine (GOLD in PRED).
    assert ev.url_match(eval_cfg, "http://wa-host:8023/dashboard/todos?page=2", SITE_URLS) == 1.0
    assert ev.url_match(eval_cfg, "http://wa-host:8023/dashboard", SITE_URLS) == 0.0


def test_url_match_reference_query_must_be_contained():
    eval_cfg = {"reference_url": "__GITLAB__/search?q=dotfiles"}
    assert ev.url_match(eval_cfg, "http://wa-host:8023/search?q=dotfiles&sort=stars", SITE_URLS) == 1.0
    assert ev.url_match(eval_cfg, "http://wa-host:8023/search?q=other", SITE_URLS) == 0.0


def test_url_match_alternatives_and_unknown_rule():
    eval_cfg = {"reference_url": "__GITLAB__/a |OR| __GITLAB__/b"}
    assert ev.url_match(eval_cfg, "http://wa-host:8023/b", SITE_URLS) == 1.0
    with pytest.raises(ValueError):
        ev.url_match({"reference_url": "__GITLAB__/a", "url_note": "EXACT"}, "http://x", SITE_URLS)


def test_score_url_match_candidates_scans_all_tabs():
    eval_cfg = {"reference_url": "__GITLAB__/dashboard/todos"}
    score, matched, unique = ev.score_url_match_candidates(
        eval_cfg,
        ["http://wa-host:8023/other", "http://wa-host:8023/dashboard/todos"],
        SITE_URLS,
    )
    assert score == 1.0
    assert matched == "http://wa-host:8023/dashboard/todos"
    assert len(unique) == 2
    assert ev.score_url_match_candidates(eval_cfg, [], SITE_URLS)[0] == 0.0


def test_score_program_html_required():
    assert ev.score_program_html_required({"exact_match": "42"}, "42") == 1.0
    assert ev.score_program_html_required({"must_include": ["foo", "bar |OR| baz"]}, "foo and baz") == 1.0
    assert ev.score_program_html_required({"must_include": ["foo", "bar |OR| baz"]}, "foo only") == 0.0
    # HTML entities are unescaped before matching.
    assert ev.score_program_html_required({"exact_match": "a&b"}, "a&amp;b") == 1.0
    with pytest.raises(ValueError):
        ev.score_program_html_required({"regex": "x"}, "y")


def test_parse_judge_json_label_tolerates_wrappers():
    allowed = {"correct", "incorrect", "partially correct"}
    assert ev.parse_judge_json_label('{"judgement": "correct", "reasoning": "ok"}', allowed) == "correct"
    fenced = 'Sure!\n```json\n{"judgement": "incorrect", "reasoning": "no"}\n```'
    assert ev.parse_judge_json_label(fenced, allowed) == "incorrect"
    thinking = '<think>{"judgement": "correct"}</think>{"judgement": "incorrect", "reasoning": "r"}'
    assert ev.parse_judge_json_label(thinking, allowed) == "incorrect"
    assert ev.parse_judge_json_label("no json here", allowed) is None
    assert ev.judge_passed("ua_match", '{"judgement": "same", "reasoning": "r"}') == 1.0
    assert ev.judge_passed("fuzzy_match", '{"judgement": "partially correct", "reasoning": "r"}') == 0.0


def test_trajectory_candidate_urls_most_recent_first():
    urls = ["http://a", "http://b", "", "error:browser_stuck", "http://a", "http://c"]
    assert ev.trajectory_candidate_urls(urls) == ["http://c", "http://a", "http://b"]


########################################
# Tier 0 — func: site-API helper resolution
########################################


def test_reddit_get_post_url_strips_comment_path():
    assert sah.reddit_get_post_url("http://h:9999/f/pics/129416/comment/1") == "http://h:9999/f/pics/129416/"
    # Non-post URLs pass through unchanged.
    assert sah.reddit_get_post_url("http://h:9999/user/x/comments") == "http://h:9999/user/x/comments"


def test_normalize_number_string():
    assert sah.normalize_number_string("10.0000") == "10"
    assert sah.normalize_number_string("12.50") == "12.5"
    assert sah.normalize_number_string("7") == "7"
    assert sah.normalize_number_string(None) == ""


def test_expression_uses_helpers():
    assert sah.expression_uses_helpers("func:reddit_get_post_url('__last_url__')")
    assert not sah.expression_uses_helpers("document.querySelector('#x').outerText")


class _FakeSiteAPI:
    async def shopping_admin_get_cart_price_rule(self, rule_name):
        return f"rule:{rule_name}"

    def reddit_get_post_url(self, url):
        return sah.reddit_get_post_url(url)

    async def gitlab_get_project_memeber_role(self, page, account_name):
        return f"{page.tag}:{account_name}"


class _FakePage:
    tag = "PAGE"


@pytest.mark.asyncio
async def test_resolve_helper_expression_dispatch():
    api, page = _FakeSiteAPI(), _FakePage()
    assert (
        await sah.resolve_helper_expression(
            "func:shopping_admin_get_cart_price_rule('fall discount')", api, page, "http://h/"
        )
        == "rule:fall discount"
    )
    # __last_url__ is substituted textually inside the quoted argument.
    assert (
        await sah.resolve_helper_expression(
            "func:reddit_get_post_url('__last_url__')", api, page, "http://h:9999/f/pics/12/comment/9"
        )
        == "http://h:9999/f/pics/12/"
    )
    # __page__ resolves to the live page object.
    assert (
        await sah.resolve_helper_expression(
            "func:gitlab_get_project_memeber_role(__page__, 'vinta')", api, page, "x"
        )
        == "PAGE:vinta"
    )


@pytest.mark.asyncio
async def test_resolve_helper_expression_rejects_unknown():
    api, page = _FakeSiteAPI(), _FakePage()
    with pytest.raises(ValueError):
        await sah.resolve_helper_expression("func:os_system('rm -rf /')", api, page, "x")
    with pytest.raises(ValueError):
        await sah.resolve_helper_expression("func:reddit_get_post_url(url='x')", api, page, "x")
    with pytest.raises(ValueError):
        await sah.resolve_helper_expression("func:reddit_get_post_url('a') + 'b'", api, page, "x")


########################################
# Tier 1 — verify() with a mocked ServerClient
########################################


def _make_server(monkeypatch, **config_overrides) -> WebArenaResourcesServer:
    monkeypatch.setattr(browser_gym_app, "ensure_playwright", lambda: None)
    config = WebArenaResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        site_urls=dict(SITE_URLS),
        **config_overrides,
    )
    return WebArenaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_verify_request(final_message, step_urls, verifier_metadata) -> CUAVerifyRequest:
    steps = [
        CUAStep(action=BrowserAction(action_type="goto", url=url), screenshot_after="", current_url=url)
        for url in step_urls
    ]
    trajectory = CUATrajectory(steps=steps, task_prompt="task", final_message=final_message)
    response = CUANeMoGymResponse(
        id="cua_test",
        created_at=0,
        model="test-model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg_test",
                content=[NeMoGymResponseOutputText(annotations=[], text=final_message or "")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        env_id="env-test",
        trajectory=trajectory,
        local_storage_dump=None,
    )
    return CUAVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "task"}]},
        response=response,
        verifier_metadata=verifier_metadata,
    )


def _judge_response_payload(judgement: str) -> dict:
    return NeMoGymResponse(
        id="judge_test",
        created_at=0,
        model="judge-model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg_judge",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[], text=f'{{"judgement": "{judgement}", "reasoning": "r"}}'
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    ).model_dump()


@pytest.mark.asyncio
async def test_verify_string_match_exact_pass(monkeypatch):
    server = _make_server(monkeypatch)
    body = _make_verify_request(
        final_message="Quest Lumaflex Band",
        step_urls=["http://wa-host:7780/admin/dashboard"],
        verifier_metadata={
            "intent": "Get the top-1 best-selling product name in 2022",
            "eval": {"eval_types": ["string_match"], "reference_answers": {"exact_match": "Quest Lumaflex Band"}},
        },
    )
    result = await server.verify(body)
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_verify_string_match_fail_without_judge(monkeypatch):
    server = _make_server(monkeypatch)
    body = _make_verify_request(
        final_message="wrong answer",
        step_urls=[],
        verifier_metadata={
            "intent": "intent",
            "eval": {"eval_types": ["string_match"], "reference_answers": {"exact_match": "Sprite"}},
        },
    )
    result = await server.verify(body)
    assert result.reward == 0.0
    assert any("judge unavailable" in message for message in result.verification_result["messages"])


@pytest.mark.asyncio
async def test_verify_string_match_fuzzy_with_judge(monkeypatch):
    server = _make_server(
        monkeypatch,
        judge_model_server=ModelServerRef(type="responses_api_models", name="webarena_judge_model"),
    )
    server.server_client.post = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(
        webarena_app, "get_response_json", AsyncMock(return_value=_judge_response_payload("correct"))
    )
    body = _make_verify_request(
        final_message="roughly thirty dollars",
        step_urls=[],
        verifier_metadata={
            "intent": "How much is shipping?",
            "eval": {"eval_types": ["string_match"], "reference_answers": {"fuzzy_match": ["around $30"]}},
        },
    )
    result = await server.verify(body)
    assert result.reward == 1.0
    assert result.verification_result["judge_evaluations"][0]["passed"] is True

    monkeypatch.setattr(
        webarena_app, "get_response_json", AsyncMock(return_value=_judge_response_payload("incorrect"))
    )
    result = await server.verify(body)
    assert result.reward == 0.0


@pytest.mark.asyncio
async def test_verify_url_match_from_trajectory(monkeypatch):
    server = _make_server(monkeypatch)
    verifier_metadata = {
        "intent": "Open your todos",
        "eval": {"eval_types": ["url_match"], "reference_url": "__GITLAB__/dashboard/todos"},
    }
    body = _make_verify_request(
        final_message=None,
        step_urls=["http://wa-host:8023/", "http://wa-host:8023/dashboard/todos"],
        verifier_metadata=verifier_metadata,
    )
    result = await server.verify(body)
    assert result.reward == 1.0

    body = _make_verify_request(
        final_message=None,
        step_urls=["http://wa-host:8023/"],
        verifier_metadata=verifier_metadata,
    )
    result = await server.verify(body)
    assert result.reward == 0.0


@pytest.mark.asyncio
async def test_verify_unknown_eval_type_and_missing_eval(monkeypatch):
    server = _make_server(monkeypatch)
    body = _make_verify_request(
        final_message="x",
        step_urls=[],
        verifier_metadata={"intent": "i", "eval": {"eval_types": ["html_match"]}},
    )
    result = await server.verify(body)
    assert result.reward == 0.0

    body = _make_verify_request(final_message="x", step_urls=[], verifier_metadata={})
    result = await server.verify(body)
    assert result.reward == 0.0


@pytest.mark.asyncio
async def test_verify_multiplies_eval_types(monkeypatch):
    server = _make_server(monkeypatch)
    body = _make_verify_request(
        final_message="Sprite",
        step_urls=["http://wa-host:8023/dashboard/todos"],
        verifier_metadata={
            "intent": "i",
            "eval": {
                "eval_types": ["string_match", "url_match"],
                "reference_answers": {"exact_match": "Sprite"},
                "reference_url": "__GITLAB__/dashboard/todos",
            },
        },
    )
    result = await server.verify(body)
    assert result.reward == 1.0

    body.verifier_metadata["eval"]["reference_url"] = "__GITLAB__/elsewhere"
    result = await server.verify(body)
    assert result.reward == 0.0
