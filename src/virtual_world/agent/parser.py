"""Parse structured JSON decisions from raw LLM text."""

from __future__ import annotations

import json
import re
from typing import Any

from virtual_world.agent.schema import AgentDecision
from virtual_world.harness.action_space import ActionSpace


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output."""
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)

    block_match = _JSON_BLOCK_RE.search(stripped)
    if block_match is not None:
        return json.loads(block_match.group(1))

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(stripped[start : end + 1])

    raise ValueError("No JSON object found in LLM response")


def parse_decision(text: str, action_space: ActionSpace) -> AgentDecision:
    """Parse and validate an :class:`AgentDecision` from LLM output."""
    payload = extract_json_object(text)
    decision = AgentDecision.model_validate(payload)
    action_space.get(decision.action_id)  # validate membership
    return decision
