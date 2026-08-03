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
"""Vision adapter: OpenAI-strict schema compliance and explicit-null handling.

Azure/OpenAI strict structured outputs (1) reject any schema whose `required`
does not list every property, and (2) make the model emit explicit nulls for
unused fields — which silently bypass `.get(key, default)` defaults. These
tests pin both behaviors so a schema edit can't regress them.
"""

import json

from responses_api_agents.browser_agent.adapters.vision_adapter import (
    BROWSER_ACTION_SCHEMA,
    VisionCUAAdapter,
)


def _adapter() -> VisionCUAAdapter:
    return VisionCUAAdapter(model="test-model", viewport_width=1000, viewport_height=500)


def _nulls_except(**overrides):
    """An action dict as a strict-schema model would emit: all keys, unused = null."""
    action = {key: None for key in BROWSER_ACTION_SCHEMA["properties"]}
    action.update(overrides)
    return action


########################################
# Schema shape (OpenAI strict contract)
########################################


def test_required_lists_every_property():
    assert set(BROWSER_ACTION_SCHEMA["required"]) == set(BROWSER_ACTION_SCHEMA["properties"])
    assert BROWSER_ACTION_SCHEMA["additionalProperties"] is False


def test_optional_properties_are_null_unions():
    for key, spec in BROWSER_ACTION_SCHEMA["properties"].items():
        if key == "action_type":
            assert spec["type"] == "string"  # always required for a valid action
            continue
        assert isinstance(spec["type"], list) and "null" in spec["type"], (
            f"{key} must be nullable for OpenAI strict mode"
        )


########################################
# Explicit nulls must not break action mapping
########################################


def test_wait_with_null_duration_defaults():
    action = _adapter()._map_action(_nulls_except(action_type="wait"))
    assert action.action_type == "wait"
    assert action.duration == 1000


def test_keypress_with_null_keys_defaults_to_empty():
    action = _adapter()._map_action(_nulls_except(action_type="keypress"))
    assert action.action_type == "keypress"
    assert action.keys == []


def test_click_scroll_drag_tolerate_null_fields():
    # Route through _parse_action_json like production: it normalizes [0,1]
    # coordinates to pixels before _map_action builds the BrowserAction.
    adapter = _adapter()

    click_data = adapter._parse_action_json(json.dumps(_nulls_except(action_type="click", coordinate=[0.5, 0.5])))
    click = adapter._map_action(click_data)
    assert click.action_type == "click" and click.coordinate == [500, 250]

    scroll = adapter._map_action(_nulls_except(action_type="scroll", scroll_y=-300))
    assert scroll.action_type == "scroll" and scroll.scroll_y == -300

    drag_data = adapter._parse_action_json(
        json.dumps(_nulls_except(action_type="drag", start_coordinate=[0.1, 0.1], end_coordinate=[0.9, 0.9]))
    )
    drag = adapter._map_action(drag_data)
    assert drag.action_type == "drag" and drag.start_coordinate == [100, 50]


def test_parse_action_json_normalizes_null_padded_output():
    adapter = _adapter()
    text = json.dumps(_nulls_except(action_type="click", coordinate=[0.5, 0.4]))
    parsed = adapter._parse_action_json(text)
    assert parsed is not None
    assert parsed["action_type"] == "click"
    assert parsed["coordinate"] == [500, 200]  # [0,1] floats scaled to pixels


########################################
# done with null message ends cleanly
########################################


def _response_with_text(text: str) -> dict:
    return {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ]
    }


def test_done_with_null_message_uses_fallback():
    adapter = _adapter()
    text = json.dumps(_nulls_except(action_type="done"))
    result = adapter._parse_response(_response_with_text(text))
    assert result.done is True
    assert result.message == "Task completed."
    assert result.actions == []
