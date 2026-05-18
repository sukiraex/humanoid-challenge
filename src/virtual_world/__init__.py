"""Text-based grid world for LLM agent challenges."""

from virtual_world.actions import Action, ActionKind
from virtual_world.types import CellKind, Direction, Position, StepResult, WorldState
from virtual_world.world import VirtualWorld

__all__ = [
    "Action",
    "ActionKind",
    "CellKind",
    "Direction",
    "Position",
    "StepResult",
    "VirtualWorld",
    "WorldState",
]

__version__ = "0.1.0"
