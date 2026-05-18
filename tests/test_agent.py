"""Tests for the LLM agent core (Step 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from virtual_world.agent.llm import MockLLMClient, create_llm_client
from virtual_world.agent.loop import AgentLoop
from virtual_world.agent.parser import extract_json_object, parse_decision
from virtual_world.agent.schema import AgentDecision
from virtual_world.agent.scenarios import get_scenario
from virtual_world.harness import ActionSpace, EnvConfig, GridWorldEnv


def test_agent_decision_schema_validation() -> None:
    decision = AgentDecision(reasoning="Go east.", action_id="MOVE_EAST")
    assert decision.action_id == "move_east"

    with pytest.raises(ValidationError):
        AgentDecision(reasoning="", action_id="move_east")


def test_extract_json_from_markdown_fence() -> None:
    text = 'Here is my plan:\n```json\n{"reasoning": "hi", "action_id": "wait"}\n```'
    payload = extract_json_object(text)
    assert payload["action_id"] == "wait"


def test_parse_decision_validates_action_id() -> None:
    space = ActionSpace()
    decision = parse_decision(
        '{"reasoning": "Rest", "action_id": "wait"}',
        space,
    )
    assert decision.action_id == "wait"

    with pytest.raises(ValueError, match="Unknown action_id"):
        parse_decision('{"reasoning": "Fly", "action_id": "fly"}', space)


def test_mock_agent_completes_simple_goal() -> None:
    scenario = get_scenario("simple_goal")
    env = GridWorldEnv(
        EnvConfig(
            layout=scenario.layout,
            task=scenario.task,
            max_steps=scenario.max_steps,
        ),
    )
    llm = MockLLMClient(list(scenario.mock_script))
    result = AgentLoop(env, llm).run()

    assert result.success
    assert result.terminated
    assert not result.truncated
    assert len(result.turns) >= 4


def test_mock_agent_completes_key_door() -> None:
    scenario = get_scenario("key_door")
    env = GridWorldEnv(
        EnvConfig(layout=scenario.layout, task=scenario.task, max_steps=scenario.max_steps),
    )
    llm = MockLLMClient(list(scenario.mock_script))
    result = AgentLoop(env, llm).run()

    assert result.success
    assert result.turns[-1].observation["success"] is True


def test_episode_log_written(tmp_path: Path) -> None:
    scenario = get_scenario("simple_goal")
    env = GridWorldEnv(
        EnvConfig(layout=scenario.layout, task=scenario.task, max_steps=scenario.max_steps),
    )
    llm = MockLLMClient(list(scenario.mock_script))
    log_dir = tmp_path / "logs"
    result = AgentLoop(env, llm).run(log_dir=log_dir)

    assert result.log_path is not None
    assert result.log_path.exists()
    payload = json.loads(result.log_path.read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert len(payload["turns"]) > 0


def test_create_llm_client_mock_requires_script() -> None:
    with pytest.raises(ValueError):
        create_llm_client("mock", script=None)


def test_mock_llm_builds_messages() -> None:
    scenario = get_scenario("simple_goal")
    env = GridWorldEnv(EnvConfig(layout=scenario.layout, task=scenario.task))
    obs, _ = env.reset()
    llm = MockLLMClient(["move_east"])
    decision = llm.decide(obs, env.action_space)
    assert decision.action_id == "move_east"
    assert len(llm.calls) == 1
    assert llm.calls[0][0]["role"] == "system"
