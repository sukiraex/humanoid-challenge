"""Prompt templates for the grid-world LLM agent."""

from __future__ import annotations

from virtual_world.agent.schema import AgentDecision
from virtual_world.harness.action_space import ActionSpace
from virtual_world.harness.observation import Observation
from virtual_world.harness.serialization import to_json


SYSTEM_PROMPT = """You are an autonomous agent in a text-based grid world.

Each turn you receive an observation (task, local view, inventory, feedback).
Choose exactly ONE action from the catalog.

Respond with ONLY a JSON object (no markdown fences) matching this schema:
{
  "reasoning": "<brief chain-of-thought>",
  "action_id": "<one valid action_id>"
}

Rules:
- Use the local view: @ is you, G is goal, k is key, D is locked door, d is open door.
- pick_up only when standing on a key.
- use only when a locked door is directly in front of you and you hold the matching key.
- Do not repeat failed moves; read the Feedback line.
"""


def build_messages(
    observation: Observation,
    action_space: ActionSpace,
    *,
    history: list[tuple[Observation, AgentDecision]] | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for the LLM from the current state and recent history."""
    user_parts = [
        observation.to_prompt(),
        "",
        action_space.to_prompt_block(),
        "",
        "Respond with JSON only.",
    ]

    if history:
        user_parts.extend(["", "## Recent history"])
        for past_obs, decision in history[-3:]:
            user_parts.append(
                f"- step {past_obs.step}: thought={decision.reasoning!r} "
                f"action={decision.action_id!r} feedback={past_obs.feedback!r}"
            )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def response_format_hint() -> dict[str, object]:
    """JSON schema hint for APIs that support structured output."""
    return AgentDecision.model_json_schema()
