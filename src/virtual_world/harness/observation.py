"""Agent-facing observation format with serialization and prompt rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from virtual_world.harness.serialization import from_json, to_json
from virtual_world.types import Direction, WorldState


@dataclass(frozen=True, slots=True)
class Observation:
    """Structured observation passed to the LLM agent and loggers.

    Designed to be JSON-serializable and convertible to a compact natural-language
    prompt via :meth:`to_prompt`.
    """

    task: str
    step: int
    max_steps: int
    position: tuple[int, int]
    facing: str
    inventory: tuple[str, ...]
    local_view: tuple[tuple[str, ...], ...]
    feedback: str
    done: bool
    success: bool
    full_map: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_world_state(
        cls,
        state: WorldState,
        *,
        task: str,
        max_steps: int,
        include_full_map: bool = False,
    ) -> Observation:
        """Build an observation from the low-level world snapshot."""
        return cls(
            task=task,
            step=state.steps,
            max_steps=max_steps,
            position=(state.agent.x, state.agent.y),
            facing=state.facing.label(),
            inventory=tuple(sorted(state.inventory.keys)),
            local_view=state.visible_cells,
            feedback=state.message,
            done=state.done,
            success=state.success,
            full_map=state.ascii_map if include_full_map else None,
            metadata=dict(state.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the observation to JSON."""
        return to_json(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        """Reconstruct an observation from a dictionary."""
        local_view = tuple(tuple(row) for row in data["local_view"])
        return cls(
            task=data["task"],
            step=data["step"],
            max_steps=data["max_steps"],
            position=tuple(data["position"]),
            facing=data["facing"],
            inventory=tuple(data["inventory"]),
            local_view=local_view,
            feedback=data["feedback"],
            done=data["done"],
            success=data["success"],
            full_map=data.get("full_map"),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> Observation:
        """Deserialize an observation from JSON."""
        return cls.from_dict(from_json(text))

    def to_prompt(self) -> str:
        """Render a concise natural-language observation for LLM prompts."""
        local = "\n".join("".join(row) for row in self.local_view)
        keys = ", ".join(self.inventory) if self.inventory else "(none)"
        lines = [
            "## Task",
            self.task.strip() or "Reach the goal tile (G).",
            "",
            "## Status",
            f"Step: {self.step} / {self.max_steps}",
            f"Position: {self.position}, facing {self.facing}",
            f"Keys held: {keys}",
            f"Feedback: {self.feedback}",
            f"Episode finished: {self.done} (success={self.success})",
            "",
            "## Local view (@ = you)",
            local,
        ]
        if self.full_map is not None:
            lines.extend(["", "## Full map", self.full_map])
        return "\n".join(lines)


def observation_schema() -> dict[str, Any]:
    """JSON-schema-style description of the observation object."""
    return {
        "type": "object",
        "required": [
            "task",
            "step",
            "max_steps",
            "position",
            "facing",
            "inventory",
            "local_view",
            "feedback",
            "done",
            "success",
        ],
        "properties": {
            "task": {"type": "string"},
            "step": {"type": "integer"},
            "max_steps": {"type": "integer"},
            "position": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            "facing": {"type": "string", "enum": [d.label() for d in Direction]},
            "inventory": {"type": "array", "items": {"type": "string"}},
            "local_view": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            "feedback": {"type": "string"},
            "done": {"type": "boolean"},
            "success": {"type": "boolean"},
            "full_map": {"type": "string"},
            "metadata": {"type": "object"},
        },
    }
