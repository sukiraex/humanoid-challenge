"""Gymnasium-style environment wrapper for :class:`VirtualWorld`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from virtual_world.actions import Action
from virtual_world.harness.action_space import ActionSpace, action_to_dict
from virtual_world.harness.observation import Observation, observation_schema
from virtual_world.harness.serialization import to_json
from virtual_world.types import WorldState
from virtual_world.world import VirtualWorld


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Configuration for a grid-world episode."""

    layout: str
    task: str = "Reach the goal tile (G)."
    max_steps: int = 200
    include_full_map: bool = False


class GridWorldEnv:
    """Gym-like facade between agents and :class:`VirtualWorld`.

    Follows the familiar ``reset`` / ``step`` contract::

        observation, info = env.reset()
        observation, reward, terminated, truncated, info = env.step(action)

  Actions may be :class:`Action` instances or payloads understood by
    :meth:`ActionSpace.decode` (action ID strings or JSON-compatible dicts).
    """

    def __init__(
        self,
        config: EnvConfig,
        *,
        action_space: ActionSpace | None = None,
    ) -> None:
        self._config = config
        self._action_space = action_space or ActionSpace()
        self._world: VirtualWorld | None = None
        self._last_result_info: dict[str, Any] = {}

    @property
    def config(self) -> EnvConfig:
        return self._config

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def world(self) -> VirtualWorld:
        if self._world is None:
            raise RuntimeError("Environment not initialized; call reset() first.")
        return self._world

    def observation_space(self) -> dict[str, Any]:
        """Return a JSON-schema-style description of observations."""
        return observation_schema()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Start a new episode and return the initial observation."""
        del seed, options  # deterministic world; seed reserved for future use

        self._world = VirtualWorld.from_layout(
            self._config.layout,
            max_steps=self._config.max_steps,
        )
        state = self._world.reset()
        observation = self._build_observation(state)
        info = {
            "action_space": self._action_space.describe(),
            "episode_message": state.message,
        }
        self._last_result_info = {}
        return observation, info

    def step(
        self,
        action: Action | str | Mapping[str, Any],
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply an action and return ``(observation, reward, terminated, truncated, info)``."""
        parsed = action if isinstance(action, Action) else self._action_space.decode(action)
        if not self._action_space.contains(parsed):
            raise ValueError(f"Action not in action space: {action_to_dict(parsed)}")

        result = self.world.step(parsed)
        observation = self._build_observation(result.state)
        info = {
            **result.info,
            "action": action_to_dict(parsed),
            "episode_message": result.state.message,
        }
        self._last_result_info = info
        return observation, result.reward, result.terminated, result.truncated, info

    def render(self) -> str:
        """Return the current ASCII map."""
        return self.world.render()

    def export_state(self) -> dict[str, Any]:
        """Serialize the full environment state for logging or checkpoints."""
        state = self.world.observe()
        observation = self._build_observation(state)
        return {
            "config": {
                "task": self._config.task,
                "max_steps": self._config.max_steps,
                "include_full_map": self._config.include_full_map,
            },
            "observation": observation.to_dict(),
            "map": self.render(),
        }

    def export_state_json(self, *, indent: int | None = 2) -> str:
        return to_json(self.export_state(), indent=indent)

    def _build_observation(self, state: WorldState) -> Observation:
        return Observation.from_world_state(
            state,
            task=self._config.task,
            max_steps=self._config.max_steps,
            include_full_map=self._config.include_full_map,
        )
