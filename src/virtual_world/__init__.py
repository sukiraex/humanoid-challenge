"""Text-based grid world for LLM agent challenges."""

from virtual_world.actions import Action, ActionKind
from virtual_world.agent import AgentDecision, AgentLoop, MockLLMClient, create_llm_client
from virtual_world.harness import (
    ActionSpace,
    EnvConfig,
    GridWorldEnv,
    Observation,
    to_json,
)
from virtual_world.types import CellKind, Direction, Position, StepResult, WorldState
from virtual_world.world import VirtualWorld

__all__ = [
    "Action",
    "ActionKind",
    "ActionSpace",
    "AgentDecision",
    "AgentLoop",
    "CellKind",
    "Direction",
    "EnvConfig",
    "GridWorldEnv",
    "MockLLMClient",
    "Observation",
    "Position",
    "StepResult",
    "VirtualWorld",
    "WorldState",
    "create_llm_client",
    "to_json",
]

__version__ = "0.4.0"
