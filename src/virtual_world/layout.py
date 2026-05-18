"""ASCII layout parsing and built-in scenario maps."""

from __future__ import annotations

from dataclasses import dataclass

from virtual_world.entities import Entity
from virtual_world.types import CellKind, Direction, Position


@dataclass(frozen=True, slots=True)
class ParsedLayout:
    """Result of parsing an ASCII map string."""

    terrain: tuple[tuple[CellKind, ...], ...]
    entities: tuple[Entity, ...]
    agent_start: Position
    agent_facing: Direction
    width: int
    height: int


_LEGEND: dict[str, CellKind | None] = {
    ".": CellKind.FLOOR,
    " ": CellKind.FLOOR,
    "#": CellKind.WALL,
    "G": CellKind.GOAL,
    "@": None,  # agent marker — terrain underneath is floor
    "k": None,
    "K": None,
    "D": None,
    "d": None,
    "O": None,
}


def _parse_entity(char: str, position: Position) -> Entity | None:
    if char == "k":
        return Entity.key(position, "main")
    if char == "K":
        return Entity.key(position, "red")
    if char in {"D", "d"}:
        return Entity.door(position, "main", locked=char == "D")
    if char == "R":
        return Entity.door(position, "red", locked=True)
    if char == "O":
        return Entity.obstacle(position)
    return None


def parse_layout(
    layout: str,
    *,
    agent_facing: Direction = Direction.NORTH,
) -> ParsedLayout:
    """Parse a multi-line ASCII map into terrain and entities.

    Legend:
        . or space — floor
        # — wall
        G — goal tile
        @ — agent start (floor underneath)
        k / K — key (main / red id)
        D / d — locked / unlocked door (main key)
        R — locked red door
        O — obstacle (blocking crate)
    """
    rows = [line.rstrip("\n") for line in layout.strip().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Layout must contain at least one row")

    width = max(len(row) for row in rows)
    height = len(rows)

    terrain: list[list[CellKind]] = []
    entities: list[Entity] = []
    agent_start: Position | None = None

    for y, row in enumerate(rows):
        padded = row.ljust(width)
        terrain_row: list[CellKind] = []
        for x, char in enumerate(padded):
            position = Position(x, y)
            if char == "@":
                agent_start = position
                terrain_row.append(CellKind.FLOOR)
                continue

            entity = _parse_entity(char, position)
            if entity is not None:
                entities.append(entity)
                terrain_row.append(CellKind.FLOOR)
                continue

            cell = _LEGEND.get(char)
            if cell is None:
                raise ValueError(f"Unknown layout character '{char}' at ({x}, {y})")
            terrain_row.append(cell)

        terrain.append(terrain_row)

    if agent_start is None:
        raise ValueError("Layout must include an agent marker '@'")

    terrain_tuple = tuple(tuple(row) for row in terrain)
    return ParsedLayout(
        terrain=terrain_tuple,
        entities=tuple(entities),
        agent_start=agent_start,
        agent_facing=agent_facing,
        width=width,
        height=height,
    )


KEY_AND_DOOR_SCENARIO = """
##############
#@...........#
#.###.#####.##
#...#...#...G#
###.#.#.#.####
#...#k#.#....#
#.###.#.###.##
#.....#.....R#
##############
"""

SIMPLE_GOAL_SCENARIO = """
#######
#@...G#
#######
"""

MINI_KEY_DOOR_SCENARIO = """
#####
#@k.#
#D.G#
#####
"""
