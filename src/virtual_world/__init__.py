"""Text-based grid world for LLM agent challenges."""

from virtual_world.actions import Action, ActionKind
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
    "CellKind",
    "Direction",
    "EnvConfig",
    "GridWorldEnv",
    "Observation",
    "Position",
    "StepResult",
    "VirtualWorld",
    "WorldState",
    "to_json",
]

__version__ = "0.2.0"
