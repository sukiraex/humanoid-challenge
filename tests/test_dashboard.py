"""Tests for the live terminal dashboard (Step 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from virtual_world.agent.llm import MockLLMClient
from virtual_world.agent.loop import AgentLoop
from virtual_world.agent.scenarios import get_scenario
from virtual_world.dashboard.render import (
    action_block,
    episode_banner,
    status_line,
    thought_block,
)
from virtual_world.dashboard.replay import replay_episode
from virtual_world.harness import EnvConfig, GridWorldEnv
from virtual_world.harness.observation import Observation


class RecordingCallback:
    """Collects agent loop events for assertions."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_reset(self, observation: Observation) -> None:
        self.events.append("reset")

    def on_thinking(self, observation: Observation) -> None:
        self.events.append("thinking")

    def on_decision(self, observation: Observation, decision: object) -> None:
        self.events.append("decision")

    def on_step(self, turn: object) -> None:
        self.events.append("step")

    def on_complete(self, result: object) -> None:
        self.events.append("complete")


def test_render_helpers() -> None:
    scenario = get_scenario("simple_goal")
    env = GridWorldEnv(EnvConfig(layout=scenario.layout, task=scenario.task))
    obs, _ = env.reset()

    assert "Step 0" in status_line(obs, total_reward=0.0, phase="ready")
    assert thought_block(None, phase="thinking") == "Consulting the model..."
    assert action_block(None) == "—"
    assert "SUCCESS" in episode_banner(True, 4, 0.95)


def test_agent_loop_invokes_callbacks() -> None:
    scenario = get_scenario("simple_goal")
    env = GridWorldEnv(
        EnvConfig(layout=scenario.layout, task=scenario.task, max_steps=scenario.max_steps),
    )
    recorder = RecordingCallback()
    llm = MockLLMClient(list(scenario.mock_script))

    result = AgentLoop(env, llm).run(callbacks=[recorder])

    assert result.success
    assert recorder.events[0] == "reset"
    assert recorder.events[-1] == "complete"
    assert "thinking" in recorder.events
    assert "decision" in recorder.events
    assert "step" in recorder.events


def test_episode_log_includes_initial_state(tmp_path: Path) -> None:
    scenario = get_scenario("simple_goal")
    env = GridWorldEnv(
        EnvConfig(layout=scenario.layout, task=scenario.task, max_steps=scenario.max_steps),
    )
    llm = MockLLMClient(list(scenario.mock_script))
    result = AgentLoop(env, llm).run(log_dir=tmp_path / "logs")

    assert result.log_path is not None
    payload = json.loads(result.log_path.read_text(encoding="utf-8"))
    assert "initial_observation" in payload
    assert "initial_map" in payload
    assert payload["initial_observation"]["step"] == 0


def test_replay_from_log(tmp_path: Path) -> None:
    pytest.importorskip("rich")
    scenario = get_scenario("simple_goal")
    env = GridWorldEnv(
        EnvConfig(layout=scenario.layout, task=scenario.task, max_steps=scenario.max_steps),
    )
    llm = MockLLMClient(list(scenario.mock_script))
    result = AgentLoop(env, llm).run(log_dir=tmp_path)

    assert result.log_path is not None
    success = replay_episode(result.log_path, delay_seconds=0.0)
    assert success is True


def test_live_dashboard_requires_rich() -> None:
    from virtual_world.dashboard.console import LiveDashboard

    try:
        LiveDashboard(task="test", delay_seconds=0)
    except ImportError as exc:
        pytest.skip(str(exc))
