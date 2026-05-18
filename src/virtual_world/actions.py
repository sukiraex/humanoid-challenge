"""Action definitions the agent (or harness) can issue."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from virtual_world.types import Direction


class ActionKind(Enum):
    """Discrete action types supported by the world."""

    MOVE = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    PICK_UP = auto()
    USE = auto()
    WAIT = auto()


@dataclass(frozen=True, slots=True)
class Action:
    """A single environment command."""

    kind: ActionKind
    direction: Direction | None = None

    @classmethod
    def move(cls, direction: Direction) -> Action:
        return cls(kind=ActionKind.MOVE, direction=direction)

    @classmethod
    def turn_left(cls) -> Action:
        return cls(kind=ActionKind.TURN_LEFT)

    @classmethod
    def turn_right(cls) -> Action:
        return cls(kind=ActionKind.TURN_RIGHT)

    @classmethod
    def pick_up(cls) -> Action:
        return cls(kind=ActionKind.PICK_UP)

    @classmethod
    def use(cls) -> Action:
        return cls(kind=ActionKind.USE)

    @classmethod
    def wait(cls) -> Action:
        return cls(kind=ActionKind.WAIT)
