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
"""
Vision-based CUA adapter.

Uses any vision-capable LLM via the OpenAI Responses API (no computer_use tool).
Sends screenshots as images, instructs the model to return JSON browser actions.
Manages conversation history client-side with turn-based trimming.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from resources_servers.browser_gym.schemas import BrowserAction
from responses_api_agents.browser_agent.adapters.base import (
    BaseCUAAdapter,
    CUAAdapterResponse,
    CUAAdapterUsage,
    extract_token_ids_from_response,
)


logger = logging.getLogger(__name__)

VISION_CUA_SYSTEM_PROMPT = """\
You are a browser automation agent. You see screenshots and return JSON actions.

Reply with exactly ONE JSON object per turn. All coordinates are [x, y] pixel values for a 1280x720 viewport.

## Actions

Mouse:
{"action_type": "click", "coordinate": [x, y]}
{"action_type": "double_click", "coordinate": [x, y]}
{"action_type": "triple_click", "coordinate": [x, y]}
{"action_type": "right_click", "coordinate": [x, y]}
{"action_type": "middle_click", "coordinate": [x, y]}
{"action_type": "hover", "coordinate": [x, y]}
{"action_type": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}

Keyboard:
{"action_type": "type", "coordinate": [x, y], "text": "hello"}
{"action_type": "type", "coordinate": [x, y], "text": "hello", "press_enter": true}
{"action_type": "type", "coordinate": [x, y], "text": "hello", "clear_before_typing": true}
{"action_type": "keypress", "keys": ["Enter"]}
{"action_type": "keypress", "keys": ["Control", "a"]}

Scroll:
{"action_type": "scroll", "coordinate": [x, y], "scroll_x": 0, "scroll_y": -300}

Navigation:
{"action_type": "goto", "url": "https://..."}
{"action_type": "go_back"}
{"action_type": "go_forward"}

Tabs:
{"action_type": "new_tab", "url": "https://..."}
{"action_type": "close_tab"}
{"action_type": "switch_tab", "tab_index": 0}

Utility:
{"action_type": "screenshot"}
{"action_type": "wait", "duration": 1000}

Completion:
{"action_type": "done", "message": "Task completed"}

Return ONLY the JSON object. No markdown, no explanation. Act autonomously — do not ask the user for help.
"""

BROWSER_ACTION_SCHEMA = {
    "type": "object",
    # OpenAI strict structured outputs require EVERY property key in `required`,
    # with optionality expressed as a null type union instead. vLLM accepts both
    # forms, so this shape works for all providers.
    "required": [
        "action_type",
        "coordinate",
        "start_coordinate",
        "end_coordinate",
        "text",
        "keys",
        "url",
        "scroll_x",
        "scroll_y",
        "tab_index",
        "duration",
        "message",
        "press_enter",
        "clear_before_typing",
        "button",
    ],
    "properties": {
        "action_type": {
            "type": "string",
            "enum": [
                "click", "double_click", "triple_click", "right_click", "middle_click",
                "hover", "drag", "type", "keypress", "scroll",
                "goto", "go_back", "go_forward",
                "new_tab", "close_tab", "switch_tab",
                "screenshot", "wait", "done",
            ],
        },
        "coordinate": {"type": ["array", "null"], "items": {"type": "number"}},
        "start_coordinate": {"type": ["array", "null"], "items": {"type": "number"}},
        "end_coordinate": {"type": ["array", "null"], "items": {"type": "number"}},
        "text": {"type": ["string", "null"]},
        "keys": {"type": ["array", "null"], "items": {"type": "string"}},
        "url": {"type": ["string", "null"]},
        "scroll_x": {"type": ["number", "null"]},
        "scroll_y": {"type": ["number", "null"]},
        "tab_index": {"type": ["integer", "null"]},
        "duration": {"type": ["integer", "null"]},
        "message": {"type": ["string", "null"]},
        "press_enter": {"type": ["boolean", "null"]},
        "clear_before_typing": {"type": ["boolean", "null"]},
        "button": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}


class VisionCUAAdapter(BaseCUAAdapter):
    def __init__(
        self,
        model: str = "aws/anthropic/bedrock-claude-opus-4-6",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        max_turns_to_keep: int = 8,
        api_caller=None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        self._model = model
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._max_turns_to_keep = max_turns_to_keep
        self._api_caller = api_caller
        self._temperature = temperature
        self._top_p = top_p
        self._messages: List[Dict[str, Any]] = []
        self._system_prompt = ""

    async def _call_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._api_caller:
            raise RuntimeError("VisionCUAAdapter requires an api_caller.")
        return await self._api_caller(payload)

    def _trim_history(self):
        """Keep only the last N turns to avoid context overflow."""
        if len(self._messages) > self._max_turns_to_keep * 2:
            # Keep system context (first user message) and recent turns
            self._messages = self._messages[:1] + self._messages[-(self._max_turns_to_keep * 2 - 1):]

    def _build_user_content(self, text: str, screenshot_b64: str) -> List[Dict[str, Any]]:
        return [
            {"type": "input_text", "text": text},
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{screenshot_b64}",
                "detail": "auto",
            },
        ]

    def _normalize_coordinates(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert normalized [0,1] float coordinates to pixel integers."""
        for key in ("coordinate", "start_coordinate", "end_coordinate"):
            coord = action_data.get(key)
            if not coord or not isinstance(coord, (list, tuple)) or len(coord) < 2:
                continue
            # Detect normalized floats: all values are between 0 and 1
            if all(isinstance(v, (int, float)) and 0 <= v <= 1.0 for v in coord):
                # Only scale if they look normalized (at least one is a true float)
                if any(isinstance(v, float) and v != int(v) for v in coord):
                    action_data[key] = [
                        int(round(coord[0] * self._viewport_width)),
                        int(round(coord[1] * self._viewport_height)),
                    ]
                    continue
            # Round any remaining floats to int
            action_data[key] = [int(round(v)) for v in coord]
        return action_data

    @staticmethod
    def _sanitize_json_text(text: str) -> str:
        """Normalize common non-standard JSON from LLMs into parseable JSON."""
        # Replace literal two-char \n sequences with actual newlines
        text = text.replace("\\n", "\n")
        # Replace single-quoted strings with double-quoted (simple heuristic)
        text = re.sub(r"(?<=[\[{,:\s])\s*'([^']*?)'\s*(?=[\]},:])", r'"\1"', text)
        return text

    def _try_json_loads(self, text: str) -> Optional[Dict[str, Any]]:
        """Try json.loads, falling back to sanitized version."""
        for candidate in (text, self._sanitize_json_text(text)):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                continue
        return None

    def _parse_action_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract action from model response: JSON, pyautogui code, or markdown fences."""
        text = text.strip()

        # Strip <think>...</think> blocks (thinking models emit these)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Try to extract from markdown code fence (json or python)
        fence_match = re.search(r"```(?:json|python)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            fenced = fence_match.group(1).strip()
            # Check if fenced content is pyautogui code
            pyauto = self._parse_pyautogui(fenced)
            if pyauto:
                return pyauto
            # Otherwise try as JSON
            text = fenced

        # Try to find a JSON object
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            parsed = self._try_json_loads(json_match.group())
            if parsed:
                return self._normalize_coordinates(parsed)

        # Try pyautogui parsing on the raw text
        pyauto = self._parse_pyautogui(text)
        if pyauto:
            return pyauto

        # Try the whole thing as JSON (with sanitization fallback)
        parsed = self._try_json_loads(text)
        if parsed:
            return self._normalize_coordinates(parsed)

        logger.warning("Failed to parse action JSON from: %s", text[:200])
        return None

    def _parse_pyautogui(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse pyautogui-style code into action dict."""
        text = text.strip()
        # pyautogui.click(x, y)
        m = re.search(r"pyautogui\.click\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", text)
        if m:
            return self._normalize_coordinates(
                {"action_type": "click", "coordinate": [float(m.group(1)), float(m.group(2))]}
            )
        # pyautogui.moveTo(x, y) then pyautogui.click()
        m = re.search(r"pyautogui\.moveTo\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", text)
        if m and "click" in text:
            return self._normalize_coordinates(
                {"action_type": "click", "coordinate": [float(m.group(1)), float(m.group(2))]}
            )
        # pyautogui.typewrite("text") or pyautogui.write("text")
        m = re.search(r"pyautogui\.(?:typewrite|write)\(\s*['\"](.+?)['\"]\s*\)", text)
        if m:
            return {"action_type": "type", "text": m.group(1)}
        # pyautogui.hotkey("ctrl", "a") etc.
        m = re.search(r"pyautogui\.hotkey\(\s*(.+?)\s*\)", text)
        if m:
            keys = [k.strip().strip("'\"") for k in m.group(1).split(",")]
            return {"action_type": "keypress", "keys": ["+".join(keys)]}
        # pyautogui.press("enter")
        m = re.search(r"pyautogui\.press\(\s*['\"](.+?)['\"]\s*\)", text)
        if m:
            return {"action_type": "keypress", "keys": [m.group(1).capitalize()]}
        # pyautogui.scroll(y) or pyautogui.scroll(y, x=x, y=y)
        m = re.search(r"pyautogui\.scroll\(\s*(-?[0-9.]+)", text)
        if m:
            return {"action_type": "scroll", "coordinate": [640, 360], "scroll_x": 0, "scroll_y": int(float(m.group(1)))}
        return None

    # Normalize model-generated action types to canonical forms
    _ACTION_ALIASES: Dict[str, str] = {
        "left_click": "click", ".click": "click", " click": "click",
        "stock_click": "click", "green_click": "click",
        "doubleclick": "double_click", "dclick": "double_click",
        "rightclick": "right_click",
        "write": "type", "typeify": "type",
        "press": "keypress", "press_enter": "keypress",
        "machine_keypress": "keypress",
        "move": "hover", "moveto": "hover", "mouseup": "hover",
        "back": "go_back", "browser_back": "go_back", "go": "goto",
        "refresh": "screenshot",
        "think": "screenshot", "analyze": "screenshot", "pause": "screenshot",
        "ping": "screenshot", "clipboard": "screenshot",
        "scrolledown": "scroll", "scrolled": "scroll", "scrolldown": "scroll",
        "scrollup": "scroll",
        "delay": "wait", "time": "wait",
        "select_all": "keypress",
        "clear": "keypress",
        "quit": "done", "terminate": "done", "stop": "done",
    }

    def _map_action(self, action_data: Dict[str, Any]) -> Optional[BrowserAction]:
        """Map parsed JSON to a BrowserAction."""
        raw_type = action_data.get("action_type", "").strip()
        action_type = raw_type.lower()

        # Apply aliases
        action_type = self._ACTION_ALIASES.get(action_type, action_type)

        if action_type == "done":
            return None  # Signals completion

        coord = action_data.get("coordinate")

        if action_type == "click":
            return BrowserAction(action_type="click", coordinate=coord, button="left")
        elif action_type == "double_click":
            return BrowserAction(action_type="double_click", coordinate=coord)
        elif action_type == "triple_click":
            return BrowserAction(action_type="triple_click", coordinate=coord)
        elif action_type == "right_click":
            return BrowserAction(action_type="right_click", coordinate=coord)
        elif action_type == "type":
            return BrowserAction(
                action_type="type",
                coordinate=coord,
                text=action_data.get("text", ""),
                press_enter=action_data.get("press_enter"),
                clear_before_typing=action_data.get("clear_before_typing"),
            )
        elif action_type == "keypress":
            keys = action_data.get("keys", [])
            if isinstance(keys, str):
                keys = [keys]
            # Handle special aliases
            if raw_type.lower() in ("select_all", "clear") and not keys:
                keys = ["Control", "a"]
            elif raw_type.lower() in ("press_enter",) and not keys:
                keys = ["Enter"]
            return BrowserAction(action_type="keypress", keys=keys)
        elif action_type == "scroll":
            return BrowserAction(
                action_type="scroll",
                coordinate=coord,
                scroll_x=action_data.get("scroll_x"),
                scroll_y=action_data.get("scroll_y"),
            )
        elif action_type == "hover":
            return BrowserAction(action_type="hover", coordinate=coord)
        elif action_type == "drag":
            return BrowserAction(
                action_type="drag",
                start_coordinate=action_data.get("start_coordinate"),
                end_coordinate=action_data.get("end_coordinate"),
            )
        elif action_type == "goto":
            return BrowserAction(action_type="goto", url=action_data.get("url"))
        elif action_type == "wait":
            # `or` (not a .get default): strict-schema models emit explicit nulls
            return BrowserAction(action_type="wait", duration=action_data.get("duration") or 1000)
        elif action_type == "screenshot":
            return BrowserAction(action_type="screenshot")
        elif action_type == "new_tab":
            return BrowserAction(action_type="new_tab", url=action_data.get("url"))
        elif action_type == "close_tab":
            return BrowserAction(action_type="close_tab")
        elif action_type == "switch_tab":
            return BrowserAction(action_type="switch_tab", tab_index=action_data.get("tab_index"))
        elif action_type == "go_back":
            return BrowserAction(action_type="go_back")
        elif action_type == "go_forward":
            return BrowserAction(action_type="go_forward")
        else:
            # Unknown type: take a screenshot instead of terminating the episode
            logger.warning("Unknown action type '%s', falling back to screenshot", raw_type)
            return BrowserAction(action_type="screenshot")

    def _parse_response(self, response: Dict[str, Any]) -> CUAAdapterResponse:
        """Parse the responses API output into CUAAdapterResponse."""
        output = response.get("output", [])
        text = ""
        for item in output:
            if item.get("type") == "message" and item.get("role") == "assistant":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        text += block.get("text", "")

        # Store assistant response in history
        self._messages.append({"role": "assistant", "content": text})

        action_data = self._parse_action_json(text)

        actions = []
        message = None
        done = False

        if action_data:
            if action_data.get("action_type") == "done":
                done = True
                message = action_data.get("message") or "Task completed."
            else:
                browser_action = self._map_action(action_data)
                if browser_action:
                    actions.append(browser_action)
                else:
                    done = True
                    message = text
        else:
            # Couldn't parse JSON — treat as done with message
            done = True
            message = text

        usage = None
        resp_usage = response.get("usage")
        if resp_usage and isinstance(resp_usage, dict):
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

    async def initialize(self, task_prompt: str, screenshot_b64: str) -> CUAAdapterResponse:
        self._messages = []

        self._system_prompt = VISION_CUA_SYSTEM_PROMPT.replace(
            "1280x720", f"{self._viewport_width}x{self._viewport_height}"
        )

        user_content = self._build_user_content(
            f"TASK: {task_prompt}\n\nHere is the current screenshot. What action should I take?",
            screenshot_b64,
        )
        self._messages.append({"role": "user", "content": user_content})

        payload = self._build_payload()

        response = await self._call_api(payload)
        return self._parse_response(response)

    def _build_payload(self) -> Dict[str, Any]:
        """Build the API payload with sampling params from the training config."""
        payload = {
            "model": self._model,
            "instructions": self._system_prompt,
            "input": self._messages,
            "max_output_tokens": 1024,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "browser_action",
                    "schema": BROWSER_ACTION_SCHEMA,
                }
            },
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._top_p is not None:
            payload["top_p"] = self._top_p
        return payload

    async def step(
        self, screenshot_b64: str, action_result: Optional[str] = None, action_error: Optional[str] = None
    ) -> CUAAdapterResponse:
        self._trim_history()

        feedback = "Action executed. Here is the updated screenshot."
        if action_error:
            feedback = f"Action FAILED with error: {action_error}. Here is the current screenshot. Please try a different approach."

        user_content = self._build_user_content(feedback, screenshot_b64)
        self._messages.append({"role": "user", "content": user_content})

        payload = self._build_payload()

        response = await self._call_api(payload)
        return self._parse_response(response)

    def reset(self):
        self._messages = []
        self._system_prompt = ""
