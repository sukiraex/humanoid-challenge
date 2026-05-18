"""Tests for the grid world environment (Step 1)."""

from __future__ import annotations

import pytest

from virtual_world import Action, Direction, VirtualWorld
from virtual_world.actions import ActionKind
from virtual_world.layout import MINI_KEY_DOOR_SCENARIO, SIMPLE_GOAL_SCENARIO, parse_layout
from virtual_world.types import Position


def test_parse_layout_dimensions() -> None:
    parsed = parse_layout(SIMPLE_GOAL_SCENARIO)
    assert parsed.width == 7
    assert parsed.height == 3
    assert parsed.agent_start == Position(1, 1)


def test_simple_goal_reached_by_moves() -> None:
    world = VirtualWorld.from_layout(SIMPLE_GOAL_SCENARIO, max_steps=50)
    world.reset()

    for _ in range(3):
        result = world.step(Action.move(Direction.EAST))
        assert not result.terminated

    result = world.step(Action.move(Direction.EAST))
    assert result.terminated
    assert result.state.success
    assert result.reward == pytest.approx(1.0)


def test_wall_blocks_movement() -> None:
    world = VirtualWorld.from_layout(SIMPLE_GOAL_SCENARIO)
    world.reset()
    result = world.step(Action.move(Direction.NORTH))
    assert "Blocked" in result.state.message
    assert world.agent == Position(1, 1)


def test_pick_up_key_and_unlock_door() -> None:
    world = VirtualWorld.from_layout(MINI_KEY_DOOR_SCENARIO, max_steps=50)
    world.reset()

    world.step(Action.move(Direction.EAST))
    world.step(Action.pick_up())
    assert world.inventory.has_key("main")

    world.step(Action.move(Direction.WEST))
    world.step(Action.turn_right())
    world.step(Action.turn_right())

    use_result = world.step(Action.use())
    assert "Unlocked" in use_result.state.message

    world.step(Action.move(Direction.SOUTH))
    world.step(Action.move(Direction.EAST))
    world.step(Action.move(Direction.EAST))
    result = world.step(Action.wait())
    assert result.terminated
    assert result.state.success


def test_door_blocks_without_key() -> None:
    world = VirtualWorld.from_layout(MINI_KEY_DOOR_SCENARIO)
    world.reset()

    world.step(Action.turn_right())
    world.step(Action.turn_right())
    result = world.step(Action.move(Direction.SOUTH))
    assert "Blocked" in result.state.message


def test_step_limit_truncates() -> None:
    world = VirtualWorld.from_layout(SIMPLE_GOAL_SCENARIO, max_steps=3)
    world.reset()
    for _ in range(2):
        world.step(Action.wait())
    result = world.step(Action.wait())
    assert result.truncated
    assert result.state.done


def test_render_contains_agent_facing() -> None:
    world = VirtualWorld.from_layout(SIMPLE_GOAL_SCENARIO)
    world.reset()
    assert "^" in world.render()
    world.step(Action.turn_right())
    assert ">" in world.render()


def test_observe_visible_window() -> None:
    world = VirtualWorld.from_layout(SIMPLE_GOAL_SCENARIO)
    state = world.reset()
    assert len(state.visible_cells) == 5
    assert len(state.visible_cells[0]) == 5
    assert state.visible_cells[2][2] == "@"


def test_action_kind_coverage() -> None:
    assert Action.turn_left().kind == ActionKind.TURN_LEFT
    assert Action.turn_right().kind == ActionKind.TURN_RIGHT
