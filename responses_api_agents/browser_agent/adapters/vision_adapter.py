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
You are a browser automation agent. You receive screenshots of a web page and must execute browser actions to complete the given task.

RESPONSE FORMAT:
You MUST respond with a JSON object (and nothing else) containing ONE action to execute. Valid action types and their fields:

{"action_type": "click", "coordinate": [x, y]}
{"action_type": "double_click", "coordinate": [x, y]}
{"action_type": "triple_click", "coordinate": [x, y]}
{"action_type": "right_click", "coordinate": [x, y]}
{"action_type": "type", "coordinate": [x, y], "text": "text to type"}
{"action_type": "type", "coordinate": [x, y], "text": "text to type", "press_enter": true}
{"action_type": "type", "coordinate": [x, y], "text": "text to type", "clear_before_typing": true}
{"action_type": "keypress", "keys": ["Control", "a"]}
{"action_type": "keypress", "keys": ["Enter"]}
{"action_type": "scroll", "coordinate": [x, y], "scroll_x": 0, "scroll_y": -300}
{"action_type": "hover", "coordinate": [x, y]}
{"action_type": "goto", "url": "https://..."}
{"action_type": "wait", "duration": 1000}
{"action_type": "screenshot"}
{"action_type": "done", "message": "Task completed successfully"}

Coordinates are in pixels relative to the viewport (1280x720 by default).
When the task is complete, respond with the "done" action type and a brief message.

RULES:
- Return ONLY a single JSON object per response, no markdown fences, no explanation.
- Execute actions directly and autonomously. NEVER ask for user confirmation.
- If an action failed (you'll see an error message), adjust your approach.
- Prefer short action cycles: look at the screenshot, take one action, wait for the result.
"""


class VisionCUAAdapter(BaseCUAAdapter):
    def __init__(
        self,
        model: str = "aws/anthropic/bedrock-claude-opus-4-6",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        max_turns_to_keep: int = 16,
        api_caller=None,
    ):
        self._model = model
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._max_turns_to_keep = max_turns_to_keep
        self._api_caller = api_caller
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

    def _parse_action_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from model response, handling markdown fences and extra text."""
        text = text.strip()
        # Try to extract from markdown code fence
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        # Try to find a JSON object
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # Try the whole thing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse action JSON from: %s", text[:200])
            return None

    def _map_action(self, action_data: Dict[str, Any]) -> Optional[BrowserAction]:
        """Map parsed JSON to a BrowserAction."""
        action_type = action_data.get("action_type", "")

        if action_type == "done":
            return None  # Signals completion

        coord = action_data.get("coordinate")

        if action_type in ("click", "left_click"):
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
            return BrowserAction(action_type="wait", duration=action_data.get("duration", 1000))
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
            logger.warning("Unknown action type: %s", action_type)
            return None

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
                message = action_data.get("message", "Task completed.")
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

        payload = {
            "model": self._model,
            "instructions": self._system_prompt,
            "input": self._messages,
            "max_output_tokens": 4096,
            "temperature": 0.0,
        }

        response = await self._call_api(payload)
        return self._parse_response(response)

    async def step(
        self, screenshot_b64: str, action_result: Optional[str] = None, action_error: Optional[str] = None
    ) -> CUAAdapterResponse:
        self._trim_history()

        feedback = "Action executed. Here is the updated screenshot."
        if action_error:
            feedback = f"Action FAILED with error: {action_error}. Here is the current screenshot. Please try a different approach."

        user_content = self._build_user_content(feedback, screenshot_b64)
        self._messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self._model,
            "instructions": self._system_prompt,
            "input": self._messages,
            "max_output_tokens": 4096,
            "temperature": 0.0,
        }

        response = await self._call_api(payload)
        return self._parse_response(response)

    def reset(self):
        self._messages = []
        self._system_prompt = ""
