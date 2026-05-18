"""Core value types for the grid world."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Direction(Enum):
    """Cardinal directions for movement."""

    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)

    @property
    def delta(self) -> tuple[int, int]:
        return self.value

    def label(self) -> str:
        return self.name.lower()


class CellKind(Enum):
    """Static terrain occupying a grid cell."""

    FLOOR = auto()
    WALL = auto()
    GOAL = auto()


@dataclass(frozen=True, slots=True)
class Position:
    """Integer grid coordinates with origin at top-left."""

    x: int
    y: int

    def moved(self, direction: Direction) -> Position:
        dx, dy = direction.delta
        return Position(self.x + dx, self.y + dy)

    def manhattan_distance(self, other: Position) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass(frozen=True, slots=True)
class Inventory:
    """Items the agent is carrying."""

    keys: frozenset[str] = frozenset()

    def with_key(self, key_id: str) -> Inventory:
        return Inventory(keys=self.keys | {key_id})

    def has_key(self, key_id: str) -> bool:
        return key_id in self.keys


@dataclass(frozen=True, slots=True)
class WorldState:
    """Serializable snapshot of the world for observations and logging."""

    width: int
    height: int
    agent: Position
    facing: Direction
    inventory: Inventory
    steps: int
    done: bool
    success: bool
    message: str
    ascii_map: str
    visible_cells: tuple[tuple[str, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StepResult:
    """Outcome of applying one action in the environment."""

    state: WorldState
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
