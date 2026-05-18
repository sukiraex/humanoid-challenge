"""Tests for the Gym-like harness (Step 2)."""

from __future__ import annotations

import json

import pytest

from virtual_world.harness import (
    ActionSpace,
    EnvConfig,
    GridWorldEnv,
    Observation,
    from_json,
    to_json,
)
from virtual_world.layout import MINI_KEY_DOOR_SCENARIO, SIMPLE_GOAL_SCENARIO


def test_reset_returns_observation_and_action_catalog() -> None:
    env = GridWorldEnv(EnvConfig(layout=SIMPLE_GOAL_SCENARIO, task="Walk to the goal."))
    obs, info = env.reset()

    assert isinstance(obs, Observation)
    assert obs.step == 0
    assert obs.max_steps == 200
    assert obs.task == "Walk to the goal."
    assert len(info["action_space"]) == 9


def test_step_accepts_action_id_string() -> None:
    env = GridWorldEnv(EnvConfig(layout=SIMPLE_GOAL_SCENARIO, max_steps=50))
    env.reset()

    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step("move_east")
        assert not terminated
        assert reward < 0

    obs, reward, terminated, truncated, info = env.step("move_east")
    assert terminated
    assert obs.success
    assert info["action"]["kind"] == "MOVE"


def test_step_accepts_structured_dict() -> None:
    env = GridWorldEnv(EnvConfig(layout=SIMPLE_GOAL_SCENARIO))
    env.reset()

    obs, _, _, _, _ = env.step({"action_id": "move_east"})
    assert obs.position == (2, 1)


def test_observation_json_round_trip() -> None:
    env = GridWorldEnv(EnvConfig(layout=SIMPLE_GOAL_SCENARIO, task="Goal run."))
    obs, _ = env.reset()

    restored = Observation.from_json(obs.to_json())
    assert restored == obs
    assert json.loads(obs.to_json())["task"] == "Goal run."


def test_observation_prompt_contains_task_and_view() -> None:
    env = GridWorldEnv(EnvConfig(layout=SIMPLE_GOAL_SCENARIO, task="Find G."))
    obs, _ = env.reset()
    prompt = obs.to_prompt()

    assert "Find G." in prompt
    assert "Local view" in prompt
    assert "@" in prompt


def test_key_door_task_via_harness() -> None:
    task = "Pick up the key, unlock the door ahead, and reach the goal."
    env = GridWorldEnv(
        EnvConfig(layout=MINI_KEY_DOOR_SCENARIO, task=task, max_steps=30),
    )
    obs, _ = env.reset()
    assert task in obs.to_prompt()

    script = [
        "move_east",
        "pick_up",
        "move_west",
        "turn_right",
        "turn_right",
        "use",
        "move_south",
        "move_east",
        "move_east",
        "wait",
    ]
    terminated = False
    for action_id in script:
        obs, _, terminated, _, _ = env.step(action_id)

    assert terminated
    assert obs.success


def test_export_state_json() -> None:
    env = GridWorldEnv(EnvConfig(layout=SIMPLE_GOAL_SCENARIO))
    env.reset()
    payload = from_json(env.export_state_json())

    assert payload["config"]["max_steps"] == 200
    assert "observation" in payload
    assert "map" in payload


def test_action_space_decode_move_alias() -> None:
    space = ActionSpace()
    action = space.decode({"action": "move", "direction": "north"})
    assert action.kind.name == "MOVE"
    assert action.direction is not None
    assert action.direction.name == "NORTH"


def test_unknown_action_raises() -> None:
    env = GridWorldEnv(EnvConfig(layout=SIMPLE_GOAL_SCENARIO))
    env.reset()
    with pytest.raises(ValueError, match="Unknown action_id"):
        env.step("fly_up")
