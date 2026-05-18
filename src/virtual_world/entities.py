"""Dynamic entities placed on top of terrain cells."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from virtual_world.types import Position


class EntityKind(Enum):
    """Kinds of interactive objects in the world."""

    KEY = auto()
    DOOR = auto()
    OBSTACLE = auto()


@dataclass(slots=True)
class Entity:
    """Interactive object with a grid position and optional key association."""

    kind: EntityKind
    position: Position
    symbol: str
    blocking: bool = False
    removable: bool = False
    key_id: str | None = None
    locked: bool | None = None

    @classmethod
    def key(cls, position: Position, key_id: str = "main") -> Entity:
        return cls(
            kind=EntityKind.KEY,
            position=position,
            symbol="k",
            blocking=False,
            removable=True,
            key_id=key_id,
        )

    @classmethod
    def door(cls, position: Position, key_id: str = "main", *, locked: bool = True) -> Entity:
        return cls(
            kind=EntityKind.DOOR,
            position=position,
            symbol="D" if locked else "d",
            blocking=locked,
            removable=False,
            key_id=key_id,
            locked=locked,
        )

    @classmethod
    def obstacle(cls, position: Position, symbol: str = "O") -> Entity:
        return cls(
            kind=EntityKind.OBSTACLE,
            position=position,
            symbol=symbol,
            blocking=True,
            removable=False,
        )

    def unlock(self) -> None:
        if self.kind != EntityKind.DOOR:
            raise ValueError("Only doors can be unlocked")
        self.locked = False
        self.blocking = False
        self.symbol = "d"
