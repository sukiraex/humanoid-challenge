"""Grid world simulation: physics, interactions, and win conditions."""

from __future__ import annotations

from dataclasses import dataclass, field

from virtual_world.actions import Action, ActionKind
from virtual_world.entities import Entity, EntityKind
from virtual_world.layout import ParsedLayout, parse_layout
from virtual_world.types import (
    CellKind,
    Direction,
    Inventory,
    Position,
    StepResult,
    WorldState,
)


_TURN_LEFT: dict[Direction, Direction] = {
    Direction.NORTH: Direction.WEST,
    Direction.WEST: Direction.SOUTH,
    Direction.SOUTH: Direction.EAST,
    Direction.EAST: Direction.NORTH,
}

_TURN_RIGHT: dict[Direction, Direction] = {
    Direction.NORTH: Direction.EAST,
    Direction.EAST: Direction.SOUTH,
    Direction.SOUTH: Direction.WEST,
    Direction.WEST: Direction.NORTH,
}


@dataclass
class VirtualWorld:
    """Text-based grid world with obstacles, keys, doors, and a goal.

    The world is the single source of truth for simulation state. Higher-level
    harness code should call :meth:`reset` and :meth:`step` rather than mutating
    internals directly.
    """

    terrain: tuple[tuple[CellKind, ...], ...]
    entities: list[Entity]
    agent: Position
    facing: Direction
    inventory: Inventory = field(default_factory=Inventory)
    steps: int = 0
    max_steps: int = 200
    message: str = ""
    _done: bool = False
    _success: bool = False

    @property
    def width(self) -> int:
        return len(self.terrain[0])

    @property
    def height(self) -> int:
        return len(self.terrain)

    @classmethod
    def from_layout(cls, layout: str, *, max_steps: int = 200) -> VirtualWorld:
        """Construct a world from an ASCII map string."""
        parsed = parse_layout(layout)
        return cls.from_parsed(parsed, max_steps=max_steps)

    @classmethod
    def from_parsed(cls, parsed: ParsedLayout, *, max_steps: int = 200) -> VirtualWorld:
        """Construct a world from a parsed layout."""
        return cls(
            terrain=parsed.terrain,
            entities=list(parsed.entities),
            agent=parsed.agent_start,
            facing=parsed.agent_facing,
            max_steps=max_steps,
        )

    def reset(self) -> WorldState:
        """Reset step counters and return the initial observation."""
        self.steps = 0
        self.message = "Episode started."
        self._done = False
        self._success = False
        return self.observe()

    def step(self, action: Action) -> StepResult:
        """Apply an action and return the transition result."""
        if self._done:
            state = self.observe()
            return StepResult(
                state=state,
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"reason": "episode_already_finished"},
            )

        reward = -0.01
        self.steps += 1
        self.message = self._apply_action(action)

        if self._on_goal():
            self._done = True
            self._success = True
            self.message = "Goal reached!"
            reward = 1.0

        truncated = self.steps >= self.max_steps and not self._done
        if truncated:
            self._done = True
            self.message = "Step limit reached."

        state = self.observe()
        return StepResult(
            state=state,
            reward=reward,
            terminated=self._success,
            truncated=truncated,
            info={"last_action": action.kind.name},
        )

    def observe(self) -> WorldState:
        """Build a rich observation snapshot for agents and loggers."""
        visible = self._visible_grid()
        return WorldState(
            width=self.width,
            height=self.height,
            agent=self.agent,
            facing=self.facing,
            inventory=self.inventory,
            steps=self.steps,
            done=self._done,
            success=self._success,
            message=self.message,
            ascii_map=self.render(),
            visible_cells=visible,
            metadata={
                "keys_held": sorted(self.inventory.keys),
                "entity_count": len(self.entities),
            },
        )

    def render(self) -> str:
        """Return an ASCII representation of the full map."""
        grid = [[self._terrain_symbol(x, y) for x in range(self.width)] for y in range(self.height)]

        for entity in self.entities:
            x, y = entity.position.x, entity.position.y
            grid[y][x] = entity.symbol

        ax, ay = self.agent.x, self.agent.y
        grid[ay][ax] = self._agent_symbol()

        return "\n".join("".join(row) for row in grid)

    def entity_at(self, position: Position) -> Entity | None:
        for entity in self.entities:
            if entity.position == position:
                return entity
        return None

    def _apply_action(self, action: Action) -> str:
        match action.kind:
            case ActionKind.MOVE:
                if action.direction is None:
                    return "Move requires a direction."
                return self._move(action.direction)
            case ActionKind.TURN_LEFT:
                self.facing = _TURN_LEFT[self.facing]
                return f"Turned left; now facing {self.facing.label()}."
            case ActionKind.TURN_RIGHT:
                self.facing = _TURN_RIGHT[self.facing]
                return f"Turned right; now facing {self.facing.label()}."
            case ActionKind.PICK_UP:
                return self._pick_up()
            case ActionKind.USE:
                return self._use_adjacent()
            case ActionKind.WAIT:
                return "Waited."
        return "Unknown action."

    def _move(self, direction: Direction) -> str:
        target = self.agent.moved(direction)
        if not self._in_bounds(target):
            return "Blocked: out of bounds."
        if self._terrain_at(target) == CellKind.WALL:
            return "Blocked: wall."
        entity = self.entity_at(target)
        if entity is not None and entity.blocking:
            label = entity.kind.name.lower()
            return f"Blocked: {label}."
        self.agent = target
        return f"Moved {direction.label()}."

    def _pick_up(self) -> str:
        entity = self.entity_at(self.agent)
        if entity is None:
            return "Nothing to pick up here."
        if entity.kind != EntityKind.KEY or entity.key_id is None:
            return "Cannot pick that up."
        self.inventory = self.inventory.with_key(entity.key_id)
        self.entities.remove(entity)
        return f"Picked up key '{entity.key_id}'."

    def _use_adjacent(self) -> str:
        """Use inventory on a door in the cell the agent faces."""
        target = self.agent.moved(self.facing)
        entity = self.entity_at(target)
        if entity is None or entity.kind != EntityKind.DOOR or entity.key_id is None:
            return "No door in front to use."
        if not entity.locked:
            return "Door is already open."
        if not self.inventory.has_key(entity.key_id):
            return f"Need key '{entity.key_id}' to unlock this door."
        entity.unlock()
        return f"Unlocked door with key '{entity.key_id}'."

    def _on_goal(self) -> bool:
        return self._terrain_at(self.agent) == CellKind.GOAL

    def _in_bounds(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def _terrain_at(self, position: Position) -> CellKind:
        return self.terrain[position.y][position.x]

    def _terrain_symbol(self, x: int, y: int) -> str:
        kind = self.terrain[y][x]
        return {
            CellKind.FLOOR: ".",
            CellKind.WALL: "#",
            CellKind.GOAL: "G",
        }[kind]

    def _agent_symbol(self) -> str:
        return {
            Direction.NORTH: "^",
            Direction.SOUTH: "v",
            Direction.EAST: ">",
            Direction.WEST: "<",
        }[self.facing]

    def _visible_grid(self, radius: int = 2) -> tuple[tuple[str, ...], ...]:
        """Local egocentric slice around the agent (for partial observability later)."""
        rows: list[tuple[str, ...]] = []
        for dy in range(-radius, radius + 1):
            row_chars: list[str] = []
            for dx in range(-radius, radius + 1):
                pos = Position(self.agent.x + dx, self.agent.y + dy)
                if pos == self.agent:
                    row_chars.append("@")
                elif not self._in_bounds(pos):
                    row_chars.append(" ")
                else:
                    entity = self.entity_at(pos)
                    if entity is not None:
                        row_chars.append(entity.symbol)
                    else:
                        row_chars.append(self._terrain_symbol(pos.x, pos.y))
            rows.append(tuple(row_chars))
        return tuple(rows)
