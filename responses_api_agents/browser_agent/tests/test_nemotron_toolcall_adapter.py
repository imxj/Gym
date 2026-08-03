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
"""Tests for the Nemotron tool-call CUA adapter (no model, no browser)."""

import json
from typing import Any, Dict, List

import pytest

from responses_api_agents.browser_agent.adapters import AdapterFactory
from responses_api_agents.browser_agent.adapters.nemotron_toolcall_adapter import (
    IMAGE_TOKEN_BUDGET,
    RESPONSE_TOKEN_HEADROOM,
    SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
    NemotronToolCallAdapter,
    decode_arguments,
)


VIEWPORT = {"viewport_width": 1000, "viewport_height": 500}


def _adapter(api_caller=None, **overrides) -> NemotronToolCallAdapter:
    kwargs = {"model": "test-model", **VIEWPORT, **overrides}
    return NemotronToolCallAdapter(api_caller=api_caller, **kwargs)


def _tool_call_response(name: str, arguments: str, call_id: str = "call_1", text: str = "") -> Dict[str, Any]:
    output: List[Dict[str, Any]] = []
    if text:
        output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
    output.append({"type": "function_call", "name": name, "arguments": arguments, "call_id": call_id})
    return {"output": output, "usage": {"input_tokens": 10, "output_tokens": 5}}


class _RecordingCaller:
    """Captures payloads and replays a scripted list of responses."""

    def __init__(self, responses: List[Dict[str, Any]]):
        self._responses = list(responses)
        self.payloads: List[Dict[str, Any]] = []

    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.payloads.append(payload)
        return self._responses.pop(0)


########################################
# Registration and payload shape
########################################


def test_adapter_is_registered():
    assert "nemotron_toolcall" in AdapterFactory.available_adapters()
    adapter = AdapterFactory.create("nemotron_toolcall", model="m", **VIEWPORT)
    assert isinstance(adapter, NemotronToolCallAdapter)


def test_tool_definitions_match_harness_contract():
    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert names == ["navigate", "computer", "tabs_create", "tabs_focus", "terminate"]
    # Responses-API tools are flat (type/name/description/parameters), not nested.
    assert all("function" not in tool for tool in TOOL_DEFINITIONS)
    # Coordinates must be advertised as normalized [0, 1].
    computer = next(t for t in TOOL_DEFINITIONS if t["name"] == "computer")
    coord = computer["parameters"]["properties"]["actions"]["items"]["properties"]["coordinate"]
    assert coord["anyOf"][0]["prefixItems"][0] == {"type": "number", "minimum": 0, "maximum": 1}
    assert "[0, 1]" in SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_initialize_sends_tools_and_screenshot_first():
    caller = _RecordingCaller([_tool_call_response("computer", '{"actions": []}')])
    adapter = _adapter(caller)
    await adapter.initialize("Find the cheapest laptop", "SCREENSHOT")

    payload = caller.payloads[0]
    assert payload["instructions"] == SYSTEM_PROMPT
    assert [t["name"] for t in payload["tools"]] == [t["name"] for t in TOOL_DEFINITIONS]
    assert payload["metadata"]["chat_template_kwargs"] == '{"truncate_history_thinking": false}'

    content = payload["input"][0]["content"]
    assert content[0]["type"] == "input_image"
    assert content[0]["image_url"].endswith("SCREENSHOT")
    assert "Find the cheapest laptop" in content[1]["text"]
    assert "Step 1" in content[1]["text"]


########################################
# Tool call -> BrowserAction mapping
########################################


@pytest.mark.asyncio
async def test_computer_click_scales_normalized_coordinates():
    caller = _RecordingCaller(
        [_tool_call_response("computer", '{"actions": [{"action": "left_click", "coordinate": [0.5, 0.4]}]}')]
    )
    result = await _adapter(caller).initialize("t", "S")
    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.action_type == "click"
    assert action.coordinate == [500, 200]  # 0.5*1000, 0.4*500
    assert not result.done


@pytest.mark.asyncio
async def test_computer_executes_action_sequence_in_order():
    args = (
        '{"actions": ['
        '{"action": "left_click", "coordinate": [0.1, 0.1]},'
        '{"action": "type", "text": "hello"},'
        '{"action": "key_press", "keys": ["ctrl", "a"]}'
        "]}"
    )
    result = await _adapter(_RecordingCaller([_tool_call_response("computer", args)])).initialize("t", "S")
    assert [a.action_type for a in result.actions] == ["click", "type", "keypress"]
    assert result.actions[1].text == "hello"
    assert result.actions[2].keys == ["ctrl", "a"]


@pytest.mark.asyncio
async def test_scroll_direction_and_drag_and_wait():
    args = (
        '{"actions": ['
        '{"action": "scroll", "scroll_parameters": {"scroll_direction": "up", "scroll_amount": 3}},'
        '{"action": "left_click_drag", "start_coordinate": [0.0, 0.0], "coordinate": [1.0, 1.0]},'
        '{"action": "wait", "duration": 2}'
        "]}"
    )
    result = await _adapter(_RecordingCaller([_tool_call_response("computer", args)])).initialize("t", "S")
    scroll, drag, wait = result.actions
    assert (scroll.scroll_x, scroll.scroll_y) == (0, -300)
    assert scroll.coordinate == [500, 250]  # viewport center default
    assert drag.start_coordinate == [0, 0] and drag.end_coordinate == [999, 499]
    assert wait.duration == 2000


@pytest.mark.asyncio
async def test_navigate_back_forward_and_url():
    for url, expected in [("back", "go_back"), ("forward", "go_forward"), ("http://x/", "goto")]:
        caller = _RecordingCaller([_tool_call_response("navigate", '{"url": "%s"}' % url)])
        result = await _adapter(caller).initialize("t", "S")
        assert result.actions[0].action_type == expected
        if expected == "goto":
            assert result.actions[0].url == "http://x/"


@pytest.mark.asyncio
async def test_tabs_create_and_focus():
    caller = _RecordingCaller([_tool_call_response("tabs_create", '{"url": "http://a/"}')])
    result = await _adapter(caller).initialize("t", "S")
    assert result.actions[0].action_type == "new_tab" and result.actions[0].url == "http://a/"

    caller = _RecordingCaller([_tool_call_response("tabs_focus", '{"tab_id": 2}')])
    result = await _adapter(caller).initialize("t", "S")
    assert result.actions[0].action_type == "switch_tab" and result.actions[0].tab_index == 2


@pytest.mark.asyncio
async def test_terminate_ends_episode_with_answer_as_message():
    caller = _RecordingCaller(
        [_tool_call_response("terminate", '{"status": "success", "answer": "Quest Lumaflex Band"}')]
    )
    result = await _adapter(caller).initialize("t", "S")
    assert result.done
    # The answer becomes final_message, which is what string_match scores.
    assert result.message == "Quest Lumaflex Band"
    assert result.actions == []


def _no_tool_call_response(text: str = "hm") -> Dict[str, Any]:
    return {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]}


@pytest.mark.asyncio
async def test_no_tool_call_ends_episode_when_retries_disabled():
    caller = _RecordingCaller([_no_tool_call_response()])
    result = await _adapter(caller, parse_retries=1, retry_sleep_seconds=0).initialize("t", "S")
    assert result.done and result.message == "hm"
    assert result.termination_reason == "no_tool_calls"


########################################
# Parse retries (harness parity: 3 blind resamples)
########################################


@pytest.mark.asyncio
async def test_parse_retry_recovers_and_keeps_history_clean():
    caller = _RecordingCaller(
        [
            _no_tool_call_response("oops 1"),
            _no_tool_call_response("oops 2"),
            _tool_call_response("computer", '{"actions": [{"action": "left_click", "coordinate": [0.5, 0.5]}]}'),
        ]
    )
    adapter = _adapter(caller, retry_sleep_seconds=0)
    result = await adapter.initialize("t", "S")

    assert not result.done
    assert result.termination_reason is None
    assert [a.action_type for a in result.actions] == ["click"]
    assert len(caller.payloads) == 3
    # Blind resample: retry payloads are identical to the first attempt's.
    assert caller.payloads[1]["input"] == caller.payloads[0]["input"]
    # Only the ACCEPTED assistant turn reaches history — no failed attempts.
    assistant_items = [i for i in adapter._input_items if i.get("type") in ("message", "function_call")]
    assistant_items = [i for i in assistant_items if i.get("role") != "user"]
    assert len(assistant_items) == 1
    assert assistant_items[0]["type"] == "function_call"


@pytest.mark.asyncio
async def test_parse_retry_exhaustion_masks_episode():
    caller = _RecordingCaller([_no_tool_call_response(f"try {i}") for i in range(3)])
    result = await _adapter(caller, retry_sleep_seconds=0).initialize("t", "S")

    assert result.done
    assert result.termination_reason == "no_tool_calls"
    assert result.message == "try 2"  # last attempt's prose
    assert len(caller.payloads) == 3


@pytest.mark.asyncio
async def test_terminate_with_preceding_actions_drops_them():
    # The agent loop never executes actions once done=True; returning them
    # would misrepresent the episode (and the harness would have executed
    # them, so this is logged loudly).
    response = {
        "output": [
            {
                "type": "function_call",
                "name": "computer",
                "arguments": '{"actions": [{"action": "left_click", "coordinate": [0.5, 0.5]}]}',
                "call_id": "call_a",
            },
            {
                "type": "function_call",
                "name": "terminate",
                "arguments": '{"status": "success", "answer": "done"}',
                "call_id": "call_b",
            },
        ]
    }
    result = await _adapter(_RecordingCaller([response]), retry_sleep_seconds=0).initialize("t", "S")
    assert result.done and result.message == "done"
    assert result.actions == []


@pytest.mark.asyncio
async def test_feedback_recovery_strips_token_ids():
    valid = _tool_call_response("terminate", '{"status": "success", "answer": "42"}')
    valid["output"][-1]["prompt_token_ids"] = [1, 2, 3]
    valid["output"][-1]["generation_token_ids"] = [4, 5]
    valid["output"][-1]["generation_log_probs"] = [-0.1, -0.2]

    # Blind-resample recovery keeps token IDs (prompt unchanged)...
    caller = _RecordingCaller([_no_tool_call_response(), json.loads(json.dumps(valid))])
    result = await _adapter(caller, retry_sleep_seconds=0).initialize("t", "S")
    assert result.prompt_token_ids == [1, 2, 3]

    # ...feedback-mode recovery must not: its prompt included transient
    # messages that are absent from persisted history.
    caller = _RecordingCaller([_no_tool_call_response(), json.loads(json.dumps(valid))])
    result = await _adapter(caller, parse_error_feedback=True, retry_sleep_seconds=0).initialize("t", "S")
    assert result.prompt_token_ids == []
    assert result.generation_token_ids == []
    assert result.message == "42"  # the recovery itself still works


@pytest.mark.asyncio
async def test_parse_retry_feedback_mode_sends_transient_correction():
    caller = _RecordingCaller(
        [
            _no_tool_call_response("bad output"),
            _tool_call_response("terminate", '{"status": "success", "answer": "42"}'),
        ]
    )
    adapter = _adapter(caller, parse_retries=2, parse_error_feedback=True, retry_sleep_seconds=0)
    result = await adapter.initialize("t", "S")

    assert result.done and result.message == "42"
    # Retry payload carries the failed output plus the corrective user message...
    retry_texts = [
        part.get("text", "")
        for item in caller.payloads[1]["input"]
        if isinstance(item.get("content"), list)
        for part in item["content"]
        if isinstance(part, dict)
    ]
    assert any("did not contain a valid tool call" in t for t in retry_texts)
    # ...but neither is persisted: history has no trace of the failed attempt.
    history_json = str(adapter._input_items)
    assert "bad output" not in history_json
    assert "did not contain a valid tool call" not in history_json


@pytest.mark.asyncio
async def test_malformed_and_unknown_actions_do_not_crash():
    caller = _RecordingCaller([_tool_call_response("computer", "not json at all")])
    result = await _adapter(caller).initialize("t", "S")
    assert result.actions == [] and not result.done

    args = '{"actions": [{"action": "teleport", "coordinate": [0.5, 0.5]}]}'
    result = await _adapter(_RecordingCaller([_tool_call_response("computer", args)])).initialize("t", "S")
    assert result.actions == []

    caller = _RecordingCaller([_tool_call_response("tabs_focus", "{}")])
    result = await _adapter(caller).initialize("t", "S")
    assert result.actions == []


def test_decode_arguments():
    assert decode_arguments('{"a": 1}') == {"a": 1}
    assert decode_arguments({"a": 1}) == {"a": 1}
    assert decode_arguments("") == {}
    assert decode_arguments("[1, 2]") == {}
    assert decode_arguments("garbage") == {}


########################################
# Multi-turn history / compaction
########################################


@pytest.mark.asyncio
async def test_step_emits_function_call_output_for_each_pending_call():
    caller = _RecordingCaller(
        [
            _tool_call_response("computer", '{"actions": [{"action": "left_click", "coordinate": [0.5, 0.5]}]}'),
            _tool_call_response("terminate", '{"status": "success", "answer": "done"}', call_id="call_2"),
        ]
    )
    adapter = _adapter(caller)
    await adapter.initialize("task", "S1")
    await adapter.step("S2", action_result="Computer actions executed.")

    second_payload_input = caller.payloads[1]["input"]
    outputs = [i for i in second_payload_input if i.get("type") == "function_call_output"]
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "call_1"
    assert outputs[0]["output"] == "Computer actions executed."
    # The assistant's function_call is echoed back into history before its output.
    assert [i.get("type") for i in second_payload_input] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]


@pytest.mark.asyncio
async def test_step_reports_action_errors_to_the_model():
    caller = _RecordingCaller(
        [
            _tool_call_response("computer", '{"actions": [{"action": "left_click", "coordinate": [0.5, 0.5]}]}'),
            _tool_call_response("terminate", '{"status": "failure"}', call_id="call_2"),
        ]
    )
    adapter = _adapter(caller)
    await adapter.initialize("task", "S1")
    await adapter.step("S2", action_error="element not found")

    outputs = [i for i in caller.payloads[1]["input"] if i.get("type") == "function_call_output"]
    assert "element not found" in outputs[0]["output"]


@pytest.mark.asyncio
async def test_only_recent_screenshots_are_kept_as_images():
    responses = [_tool_call_response("computer", '{"actions": []}', call_id=f"call_{i}") for i in range(4)]
    caller = _RecordingCaller(responses)
    adapter = _adapter(caller, max_image_history=2)
    await adapter.initialize("task", "S1")
    await adapter.step("S2")
    await adapter.step("S3")
    await adapter.step("S4")

    def image_count(payload):
        return sum(
            1
            for item in payload["input"]
            if isinstance(item.get("content"), list)
            for part in item["content"]
            if part.get("type") == "input_image"
        )

    assert image_count(caller.payloads[0]) == 1
    assert image_count(caller.payloads[-1]) == 2  # older screenshots stripped to text


@pytest.mark.asyncio
async def test_text_budget_drops_oldest_turns_with_redaction_notice():
    responses = [
        _tool_call_response("computer", '{"actions": []}', call_id=f"call_{i}", text="x" * 4000) for i in range(4)
    ]
    caller = _RecordingCaller(responses)
    # Tiny budget: max_model_len barely above the image + response headroom,
    # leaving ~400 tokens of text for the whole conversation.
    tight_max_model_len = IMAGE_TOKEN_BUDGET + RESPONSE_TOKEN_HEADROOM + 400
    adapter = _adapter(caller, max_image_history=1, max_model_len=tight_max_model_len)
    await adapter.initialize("task", "S1")
    await adapter.step("S2")
    await adapter.step("S3")
    await adapter.step("S4")

    last_input = caller.payloads[-1]["input"]
    texts = [
        part["text"]
        for item in last_input
        if isinstance(item.get("content"), list)
        for part in item["content"]
        if part.get("type") == "input_text"
    ]
    assert any("redacted to fit the context window" in t for t in texts)
    # The most recent turn always survives compaction.
    assert any("Step 4" in t for t in texts)


@pytest.mark.asyncio
async def test_reset_clears_state():
    caller = _RecordingCaller([_tool_call_response("computer", '{"actions": []}')])
    adapter = _adapter(caller)
    await adapter.initialize("task", "S1")
    adapter.reset()
    assert adapter._input_items == []
    assert adapter._step_num == 0
    assert adapter._pending_call_ids == []
