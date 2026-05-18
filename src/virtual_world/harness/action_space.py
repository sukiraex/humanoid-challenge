"""Discrete action space with decoding for LLM structured outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from virtual_world.actions import Action, ActionKind
from virtual_world.harness.serialization import to_json
from virtual_world.types import Direction


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """One discrete action choice exposed to the agent."""

    action_id: str
    description: str
    kind: ActionKind
    direction: Direction | None = None

    def to_action(self) -> Action:
        if self.kind == ActionKind.MOVE:
            if self.direction is None:
                raise ValueError(f"Move action '{self.action_id}' requires a direction")
            return Action.move(self.direction)
        builders = {
            ActionKind.TURN_LEFT: Action.turn_left,
            ActionKind.TURN_RIGHT: Action.turn_right,
            ActionKind.PICK_UP: Action.pick_up,
            ActionKind.USE: Action.use,
            ActionKind.WAIT: Action.wait,
        }
        return builders[self.kind]()


def _build_default_specs() -> tuple[ActionSpec, ...]:
    specs: list[ActionSpec] = []
    for direction in Direction:
        specs.append(
            ActionSpec(
                action_id=f"move_{direction.label()}",
                description=f"Move one cell {direction.label()}.",
                kind=ActionKind.MOVE,
                direction=direction,
            )
        )
    specs.extend(
        [
            ActionSpec("turn_left", "Turn 90° counter-clockwise.", ActionKind.TURN_LEFT),
            ActionSpec("turn_right", "Turn 90° clockwise.", ActionKind.TURN_RIGHT),
            ActionSpec("pick_up", "Pick up an item on your current cell.", ActionKind.PICK_UP),
            ActionSpec("use", "Use a key on the door directly in front of you.", ActionKind.USE),
            ActionSpec("wait", "Wait one timestep.", ActionKind.WAIT),
        ]
    )
    return tuple(specs)


@dataclass(frozen=True, slots=True)
class ActionSpace:
    """Discrete action space with LLM-friendly IDs and parsing."""

    specs: tuple[ActionSpec, ...] = _build_default_specs()

    @property
    def n(self) -> int:
        return len(self.specs)

    def ids(self) -> tuple[str, ...]:
        return tuple(spec.action_id for spec in self.specs)

    def get(self, action_id: str) -> ActionSpec:
        for spec in self.specs:
            if spec.action_id == action_id:
                return spec
        raise ValueError(f"Unknown action_id: {action_id!r}")

    def contains(self, action: Action) -> bool:
        for spec in self.specs:
            if spec.to_action() == action:
                return True
        return False

    def decode(self, payload: str | Mapping[str, Any]) -> Action:
        """Parse an action from an ID string or structured dict.

        Accepted dict shapes::

            {"action_id": "move_east"}
            {"action": "move", "direction": "east"}
            {"kind": "MOVE", "direction": "EAST"}
        """
        if isinstance(payload, str):
            return self.get(payload.strip().lower()).to_action()

        if "action_id" in payload:
            action_id = str(payload["action_id"]).strip().lower()
            return self.get(action_id).to_action()

        kind_raw = payload.get("kind") or payload.get("action")
        if kind_raw is None:
            raise ValueError("Action payload must include 'action_id', 'kind', or 'action'")

        kind_name = str(kind_raw).strip().upper()
        if kind_name in {"MOVE", "move"}:
            direction_raw = payload.get("direction")
            if direction_raw is None:
                raise ValueError("Move action requires 'direction'")
            direction = _parse_direction(str(direction_raw))
            return Action.move(direction)

        kind = ActionKind[kind_name.upper()]
        return ActionSpec("", "", kind).to_action()

    def describe(self) -> list[dict[str, str]]:
        """Return action catalog entries for prompts and tool schemas."""
        return [
            {"action_id": spec.action_id, "description": spec.description}
            for spec in self.specs
        ]

    def to_json(self, *, indent: int | None = 2) -> str:
        return to_json(self.describe(), indent=indent)

    def to_prompt_block(self) -> str:
        """Format available actions for inclusion in an LLM system prompt."""
        lines = ["## Available actions"]
        for entry in self.describe():
            lines.append(f"- {entry['action_id']}: {entry['description']}")
        return "\n".join(lines)


def action_to_dict(action: Action) -> dict[str, str]:
    """Serialize an :class:`Action` to a JSON-friendly dict."""
    data: dict[str, str] = {"kind": action.kind.name}
    if action.direction is not None:
        data["direction"] = action.direction.name
    return data


def _parse_direction(raw: str) -> Direction:
    normalized = raw.strip().upper()
    aliases = {
        "N": "NORTH",
        "S": "SOUTH",
        "E": "EAST",
        "W": "WEST",
        "NORTH": "NORTH",
        "SOUTH": "SOUTH",
        "EAST": "EAST",
        "WEST": "WEST",
    }
    key = aliases.get(normalized, normalized)
    return Direction[key]
