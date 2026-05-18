"""Pure render helpers for the terminal dashboard (testable without Rich)."""

from __future__ import annotations

from virtual_world.agent.schema import AgentDecision
from virtual_world.harness.observation import Observation


def status_line(observation: Observation, *, total_reward: float, phase: str) -> str:
    keys = ", ".join(observation.inventory) if observation.inventory else "(none)"
    return (
        f"Step {observation.step}/{observation.max_steps}  |  "
        f"Pos {observation.position}  facing {observation.facing}  |  "
        f"Keys: {keys}  |  Reward: {total_reward:.3f}  |  {phase}"
    )


def thought_block(decision: AgentDecision | None, *, phase: str) -> str:
    if phase == "thinking":
        return "Consulting the model..."
    if decision is None:
        return "—"
    return decision.reasoning


def action_block(decision: AgentDecision | None) -> str:
    if decision is None:
        return "—"
    return decision.action_id


def feedback_block(observation: Observation) -> str:
    return observation.feedback or "—"


def map_block(ascii_map: str) -> str:
    return ascii_map


def episode_banner(result_success: bool, steps: int, total_reward: float) -> str:
    label = "SUCCESS" if result_success else "FAILED"
    return f"Episode {label} — steps={steps}, total_reward={total_reward:.3f}"
