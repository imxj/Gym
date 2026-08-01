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
"""Nemotron tool-call CUA adapter.

Replicates the policy interface of the internal WebArena/OSWorld harness
(osworld_internal webarena/nvidia/nemotron_toolcall_agent.py) so that models
trained on those trajectories see the same protocol under NeMo-Gym:

- **native tool calls** (not JSON-in-text): the five harness tools
  ``navigate`` / ``computer`` / ``tabs_create`` / ``tabs_focus`` / ``terminate``
  are sent as Responses-API function tools; the model replies with
  ``function_call`` output items.
- **normalized [0, 1] coordinates** relative to the viewport, scaled to pixels
  only when mapping onto ``BrowserAction``.
- ``computer`` carries a **sequence** of actions executed in order.
- ``terminate(status, answer)`` ends the episode and its ``answer`` becomes the
  trajectory's ``final_message`` — which is what the WebArena resources server
  scores with ``string_match``.
- **history compaction**: keep only the last ``max_image_history`` screenshots
  as images, and drop whole oldest turns (with a redaction notice) to stay
  inside a text-token budget derived from ``max_model_len``.

Differences from the harness that are inherent to Gym (documented in the
webarena README): observations are headless Playwright viewport screenshots
rather than full X-display grabs, actions execute through the browser pool
rather than pyautogui, and there is no captcha handler.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Optional

from resources_servers.browser_gym.schemas import BrowserAction
from responses_api_agents.browser_agent.adapters.base import (
    BaseCUAAdapter,
    CUAAdapterResponse,
    CUAAdapterUsage,
    extract_token_ids_from_response,
)


logger = logging.getLogger(__name__)

# Harness constants (nemotron_toolcall_agent.py:37-39).
IMAGE_TOKEN_BUDGET = 2040
RESPONSE_TOKEN_HEADROOM = 4096
FALLBACK_CHARS_PER_TOKEN = 4

# Verbatim from the harness so trained models see the exact prompt they expect.
SYSTEM_PROMPT = """
You are a GUI agent controlling a web browser. You are given a task instruction, a screenshot of the browser, and your previous interactions. You need to perform a series of actions to complete the task. The browser is already open and logged into the required websites.

<tool_guidelines>
- Operate via x,y coordinates from the latest screenshot using the `computer` tool.
- Coordinates are relative to the viewport in [0, 1], with (0, 0) at the top-left.
- Use `tabs_create` and `tabs_focus` to manage tabs.
- Use `navigate` to go to URLs or use "back"/"forward" for browser history.
- When the task is complete, call `terminate` with status and answer.
</tool_guidelines>
""".strip()

_COORDINATE_SCHEMA = {
    "anyOf": [
        {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "prefixItems": [
                {"type": "number", "minimum": 0, "maximum": 1},
                {"type": "number", "minimum": 0, "maximum": 1},
            ],
        },
        {"type": "null"},
    ],
    "default": None,
}

# Responses-API tool format: flat {type, name, description, parameters}.
# (The harness uses the Chat-Completions nesting; vllm_model's converter
# re-nests these on the way out, so the model sees the same tool schema.)
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "navigate",
        "description": "Navigate to a URL, or go forward/back in browser history.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        'The URL to navigate to. Use "forward" to go forward in history '
                        'or "back" to go back in history.'
                    ),
                },
                "tab_id": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "description": "Tab ID to navigate.",
                    "default": None,
                },
            },
            "required": ["url"],
        },
    },
    {
        "type": "function",
        "name": "computer",
        "description": "Interact with the web browser with a sequence of computer actions.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": "List of actions to perform sequentially.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "The action to perform: `left_click`, `middle_click`, `right_click`, "
                                    "`double_click`, `triple_click`, `mouse_move` (coordinate), `type` (text), "
                                    "`key_press` (list of keys to press), `scroll` (direction + amount, optional "
                                    "coordinate), `left_click_drag` (start_coordinate to coordinate), or `wait` "
                                    "(duration in seconds)."
                                ),
                                "enum": [
                                    "left_click",
                                    "middle_click",
                                    "right_click",
                                    "double_click",
                                    "triple_click",
                                    "mouse_move",
                                    "type",
                                    "key_press",
                                    "wait",
                                    "scroll",
                                    "left_click_drag",
                                ],
                            },
                            "coordinate": {
                                **_COORDINATE_SCHEMA,
                                "description": (
                                    "(x, y) relative coordinates in the [0, 1] range, where (0, 0) is the top-left "
                                    "of the viewport and (1, 1) is the bottom-right. Required for click actions and "
                                    "`mouse_move`. For `scroll`, defaults to the screen center when omitted. For "
                                    "`left_click_drag`, this is the end position."
                                ),
                            },
                            "duration": {
                                "anyOf": [{"type": "integer", "minimum": 0, "maximum": 30}, {"type": "null"}],
                                "default": None,
                                "description": "The number of seconds to wait. Required for `wait`. Maximum 30 seconds.",
                            },
                            "keys": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "List of keys to press for the `key_press` action. Use platform modifier keys "
                                    'such as "cmd" on Mac or "ctrl" on Windows/Linux, e.g., ["ctrl", "a"] for '
                                    "select all."
                                ),
                            },
                            "scroll_parameters": {
                                "anyOf": [
                                    {
                                        "type": "object",
                                        "properties": {
                                            "scroll_amount": {
                                                "type": "integer",
                                                "minimum": 0,
                                                "default": 1,
                                                "description": (
                                                    "Number of mouse wheel clicks to scroll in the requested "
                                                    "direction. This value is uncapped."
                                                ),
                                            },
                                            "scroll_direction": {
                                                "type": "string",
                                                "enum": ["up", "down", "left", "right"],
                                                "default": "down",
                                                "description": "The direction to scroll in.",
                                            },
                                        },
                                        "required": ["scroll_direction", "scroll_amount"],
                                    },
                                    {"type": "null"},
                                ],
                                "default": None,
                                "description": "The parameters to scroll with. Required for `scroll`.",
                            },
                            "start_coordinate": {
                                **_COORDINATE_SCHEMA,
                                "description": (
                                    "(x, y) relative starting coordinates in the [0, 1] range for `left_click_drag`."
                                ),
                            },
                            "text": {
                                "type": "string",
                                "description": "The text to type. Only used for the `type` action.",
                            },
                        },
                        "required": ["action"],
                    },
                },
            },
            "required": ["actions"],
        },
    },
    {
        "type": "function",
        "name": "tabs_create",
        "description": "Creates a new empty tab in the current tab group",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Start URL for new tab. Default about:blank.",
                    "default": "about:blank",
                },
            },
        },
    },
    {
        "type": "function",
        "name": "tabs_focus",
        "description": "Focus an existing tab in the current tab group.",
        "parameters": {
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "integer",
                    "description": "Tab ID to focus.",
                },
            },
            "required": ["tab_id"],
        },
    },
    {
        "type": "function",
        "name": "terminate",
        "description": "Terminate the current task and report its completion status.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "failure"],
                    "description": "The status of the task.",
                },
                "answer": {
                    "type": "string",
                    "description": "The answer of the task.",
                },
            },
            "required": ["status"],
        },
    },
]

# harness `computer` action -> browser_pool action_type
_CLICK_ACTIONS = {
    "left_click": "click",
    "middle_click": "middle_click",
    "right_click": "right_click",
    "double_click": "double_click",
    "triple_click": "triple_click",
    "mouse_move": "hover",
}

# Scroll wheel clicks -> pixels, matching the harness's pyautogui scroll scale.
SCROLL_PIXELS_PER_CLICK = 100


def decode_arguments(raw: Any) -> Dict[str, Any]:
    """Tool-call arguments arrive as a JSON string (or already-parsed dict)."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Failed to decode tool call arguments: %r", raw)
        return {}
    return parsed if isinstance(parsed, dict) else {}


class NemotronToolCallAdapter(BaseCUAAdapter):
    """Native tool-call adapter matching the internal Nemotron browser harness."""

    def __init__(
        self,
        model: str,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        max_image_history: int = 3,
        max_model_len: int = 131072,
        max_output_tokens: int = 16384,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        thinking: bool = True,
        api_caller=None,
    ):
        self._model = model
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._max_image_history = max_image_history
        self._max_model_len = max_model_len
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._thinking = thinking
        self._api_caller = api_caller

        self._input_items: List[Dict[str, Any]] = []
        self._task_prompt = ""
        self._step_num = 0
        # call_id of the tool call awaiting a function_call_output
        self._pending_call_ids: List[str] = []

    ########################################
    # Message construction
    ########################################

    def _user_turn(self, screenshot_b64: str, extra_text: Optional[str] = None) -> Dict[str, Any]:
        """A user turn: screenshot first, then the step text (harness order)."""
        text_parts = []
        if self._step_num == 1:
            text_parts.append(f"# Task Instruction:\n\n{self._task_prompt}")
        text_parts.append(f"You are currently on Step {self._step_num}.")
        if extra_text:
            text_parts.append(extra_text)
        return {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                    "detail": "auto",
                },
                {"type": "input_text", "text": "\n\n".join(text_parts)},
            ],
        }

    def _build_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._model,
            "instructions": SYSTEM_PROMPT,
            "input": self._compact_input_items(),
            "tools": copy.deepcopy(TOOL_DEFINITIONS),
            "max_output_tokens": self._max_output_tokens,
            # Harness parity: keep prior thinking blocks in the rendered prompt.
            "metadata": {"chat_template_kwargs": json.dumps({"truncate_history_thinking": False})},
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._top_p is not None:
            payload["top_p"] = self._top_p
        return payload

    ########################################
    # History compaction (ported from the harness)
    ########################################

    def _available_text_budget(self) -> int:
        image_budget = max(0, self._max_image_history) * IMAGE_TOKEN_BUDGET
        return max(0, self._max_model_len - image_budget - RESPONSE_TOKEN_HEADROOM)

    @staticmethod
    def _is_image_part(part: Any) -> bool:
        return isinstance(part, dict) and part.get("type") == "input_image"

    @classmethod
    def _without_images(cls, item: Dict[str, Any]) -> Dict[str, Any]:
        stripped = copy.deepcopy(item)
        content = stripped.get("content")
        if isinstance(content, list):
            stripped["content"] = [part for part in content if not cls._is_image_part(part)]
        return stripped

    @classmethod
    def _count_text_tokens(cls, items: List[Dict[str, Any]]) -> int:
        """Character-based estimate (the harness's tokenizer-free fallback).

        Exact token accounting would need the policy's tokenizer, which the
        adapter does not have; the budget is deliberately conservative.
        """
        total_chars = 0
        for item in items:
            content = item.get("content")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and not cls._is_image_part(part):
                        total_chars += len(str(part.get("text", "")))
            for key in ("arguments", "output", "name"):
                value = item.get(key)
                if isinstance(value, str):
                    total_chars += len(value)
        return total_chars // FALLBACK_CHARS_PER_TOKEN

    @staticmethod
    def _turn_groups(items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group items into turns starting at each user message.

        A turn is [user message, assistant function_call(s), function_call_output(s)],
        so compaction never splits a tool call from its result (which would make
        the conversation unparseable for the model).
        """
        groups: List[List[Dict[str, Any]]] = []
        for item in items:
            starts_turn = item.get("type") == "message" and item.get("role") == "user"
            if starts_turn or not groups:
                groups.append([item])
            else:
                groups[-1].append(item)
        return groups

    @staticmethod
    def _prepend_redaction_notice(items: List[Dict[str, Any]], dropped_turns: int) -> None:
        if dropped_turns == 1:
            notice = "Earlier interaction step 1 was redacted to fit the context window."
        else:
            notice = f"Earlier interaction steps 1-{dropped_turns} were redacted to fit the context window."
        for item in items:
            if item.get("type") == "message" and item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "input_text":
                            part["text"] = f"{notice}\n\n{part.get('text', '')}"
                            return
                    content.append({"type": "input_text", "text": notice})
                return

    def _keep_recent_images(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        image_indices = [
            idx
            for idx, item in enumerate(items)
            if isinstance(item.get("content"), list) and any(self._is_image_part(p) for p in item["content"])
        ]
        keep_count = max(0, self._max_image_history)
        keep = set(image_indices[-keep_count:]) if keep_count else set()
        return [
            self._without_images(item) if idx in image_indices and idx not in keep else item
            for idx, item in enumerate(items)
        ]

    def _compact_input_items(self) -> List[Dict[str, Any]]:
        items = self._keep_recent_images(copy.deepcopy(self._input_items))
        if self._max_model_len <= 0 or not items:
            return items

        # Always keep the most recent turn intact.
        groups = self._turn_groups(items)
        if len(groups) <= 1:
            return items

        prior, final_turn = groups[:-1], groups[-1]
        budget = self._available_text_budget()
        dropped = 0

        def flatten(prior_groups):
            return [item for group in prior_groups for item in group] + final_turn

        compacted = flatten(prior)
        while prior and self._count_text_tokens(compacted) > budget:
            prior.pop(0)
            dropped += 1
            compacted = flatten(prior)

        if dropped:
            compacted = copy.deepcopy(compacted)
            self._prepend_redaction_notice(compacted, dropped)
            logger.info("Redacted %s prior turn(s) to fit the context text budget", dropped)
        return compacted

    ########################################
    # Tool call -> BrowserAction mapping
    ########################################

    def _to_pixels(self, coord: Any) -> Optional[List[int]]:
        """Normalized [0, 1] viewport coordinates -> pixel ints."""
        if not isinstance(coord, (list, tuple)) or len(coord) < 2:
            return None
        try:
            x, y = float(coord[0]), float(coord[1])
        except (TypeError, ValueError):
            return None
        return [
            max(0, min(self._viewport_width - 1, int(round(x * self._viewport_width)))),
            max(0, min(self._viewport_height - 1, int(round(y * self._viewport_height)))),
        ]

    def _map_computer_action(self, action: Dict[str, Any]) -> Optional[BrowserAction]:
        name = str(action.get("action") or "").strip()
        coord = self._to_pixels(action.get("coordinate"))

        if name in _CLICK_ACTIONS:
            mapped = _CLICK_ACTIONS[name]
            if mapped == "click":
                return BrowserAction(action_type="click", coordinate=coord, button="left")
            return BrowserAction(action_type=mapped, coordinate=coord)
        if name == "type":
            return BrowserAction(action_type="type", coordinate=coord, text=str(action.get("text") or ""))
        if name == "key_press":
            keys = action.get("keys") or []
            if isinstance(keys, str):
                keys = [keys]
            return BrowserAction(action_type="keypress", keys=[str(k) for k in keys])
        if name == "scroll":
            params = action.get("scroll_parameters") or {}
            direction = str(params.get("scroll_direction") or "down").lower()
            amount = params.get("scroll_amount", 1)
            try:
                magnitude = int(amount) * SCROLL_PIXELS_PER_CLICK
            except (TypeError, ValueError):
                magnitude = SCROLL_PIXELS_PER_CLICK
            deltas = {
                "down": (0, magnitude),
                "up": (0, -magnitude),
                "right": (magnitude, 0),
                "left": (-magnitude, 0),
            }
            scroll_x, scroll_y = deltas.get(direction, (0, magnitude))
            center = [self._viewport_width // 2, self._viewport_height // 2]
            return BrowserAction(
                action_type="scroll", coordinate=coord or center, scroll_x=scroll_x, scroll_y=scroll_y
            )
        if name == "left_click_drag":
            return BrowserAction(
                action_type="drag",
                start_coordinate=self._to_pixels(action.get("start_coordinate")),
                end_coordinate=coord,
            )
        if name == "wait":
            duration = action.get("duration")
            try:
                seconds = float(duration) if duration is not None else 1.0
            except (TypeError, ValueError):
                seconds = 1.0
            return BrowserAction(action_type="wait", duration=int(seconds * 1000))

        logger.warning("Unknown computer action %r; skipping", name)
        return None

    def _map_tool_call(self, name: str, args: Dict[str, Any]) -> List[BrowserAction]:
        if name == "computer":
            actions = args.get("actions") or []
            if not isinstance(actions, list):
                logger.warning("computer.actions is not a list: %r", actions)
                return []
            mapped = [self._map_computer_action(a) for a in actions if isinstance(a, dict)]
            return [a for a in mapped if a is not None]
        if name == "navigate":
            url = str(args.get("url") or "").strip()
            if url.lower() == "back":
                return [BrowserAction(action_type="go_back")]
            if url.lower() == "forward":
                return [BrowserAction(action_type="go_forward")]
            return [BrowserAction(action_type="goto", url=url)]
        if name == "tabs_create":
            return [BrowserAction(action_type="new_tab", url=str(args.get("url") or "about:blank"))]
        if name == "tabs_focus":
            try:
                tab_index = int(args.get("tab_id"))
            except (TypeError, ValueError):
                logger.warning("tabs_focus called without a valid tab_id: %r", args)
                return []
            return [BrowserAction(action_type="switch_tab", tab_index=tab_index)]

        logger.warning("Unsupported tool call %r", name)
        return []

    ########################################
    # Response parsing
    ########################################

    def _parse_response(self, response: Dict[str, Any]) -> CUAAdapterResponse:
        output = response.get("output", []) or []
        text_chunks: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                tool_calls.append(item)
            elif item.get("type") == "message" and item.get("role") == "assistant":
                for block in item.get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text_chunks.append(block.get("text", ""))

        assistant_text = "".join(text_chunks)

        # Echo the assistant turn back into history so the next request carries it.
        for item in output:
            if isinstance(item, dict) and item.get("type") in ("message", "function_call", "reasoning"):
                self._input_items.append(copy.deepcopy(item))

        actions: List[BrowserAction] = []
        done = False
        message: Optional[str] = None
        self._pending_call_ids = []

        for call in tool_calls:
            name = str(call.get("name") or "")
            args = decode_arguments(call.get("arguments"))
            call_id = call.get("call_id") or call.get("id") or ""
            self._pending_call_ids.append(str(call_id))

            if name == "terminate":
                done = True
                # The answer is what string_match scores; fall back to any prose.
                message = args.get("answer") or assistant_text or ""
                break
            actions.extend(self._map_tool_call(name, args))

        if not tool_calls:
            # No tool call: the harness treats this as a failed step. End the
            # episode with whatever prose the model produced.
            done = True
            message = assistant_text
            logger.warning("Model returned no tool calls; ending episode")

        usage = None
        resp_usage = response.get("usage")
        if isinstance(resp_usage, dict):
            in_tok = resp_usage.get("input_tokens", 0) or 0
            out_tok = resp_usage.get("output_tokens", 0) or 0
            usage = CUAAdapterUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok)

        token_ids = extract_token_ids_from_response(response)
        return CUAAdapterResponse(
            actions=actions,
            message=message,
            raw_response=response,
            done=done,
            usage=usage,
            prompt_token_ids=token_ids["prompt_token_ids"],
            generation_token_ids=token_ids["generation_token_ids"],
            generation_log_probs=token_ids["generation_log_probs"],
        )

    ########################################
    # BaseCUAAdapter interface
    ########################################

    async def _call_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._api_caller:
            raise RuntimeError("NemotronToolCallAdapter requires an api_caller.")
        return await self._api_caller(payload)

    async def initialize(self, task_prompt: str, screenshot_b64: str) -> CUAAdapterResponse:
        self.reset()
        self._task_prompt = task_prompt
        self._step_num = 1
        self._input_items.append(self._user_turn(screenshot_b64))
        return self._parse_response(await self._call_api(self._build_payload()))

    async def step(
        self, screenshot_b64: str, action_result: Optional[str] = None, action_error: Optional[str] = None
    ) -> CUAAdapterResponse:
        # Every pending tool call needs a matching function_call_output, or the
        # next request is malformed.
        if action_error:
            result_text = f"Action failed: {action_error}"
        else:
            result_text = action_result or "Actions executed."
        for call_id in self._pending_call_ids:
            self._input_items.append({"type": "function_call_output", "call_id": call_id, "output": result_text})
        self._pending_call_ids = []

        self._step_num += 1
        self._input_items.append(self._user_turn(screenshot_b64))
        return self._parse_response(await self._call_api(self._build_payload()))

    def reset(self):
        self._input_items = []
        self._task_prompt = ""
        self._step_num = 0
        self._pending_call_ids = []
