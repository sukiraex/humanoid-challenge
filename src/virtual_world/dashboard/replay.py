"""Replay a saved episode JSON through the live dashboard."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from virtual_world.agent.schema import AgentDecision
from virtual_world.dashboard.console import LiveDashboard
from virtual_world.harness.observation import Observation
from virtual_world.harness.serialization import from_json


def replay_episode(
    log_path: Path,
    *,
    delay_seconds: float = 0.35,
    show_local_view: bool = True,
) -> bool:
    """Animate a previously saved episode log. Returns whether the episode succeeded."""
    payload: dict[str, Any] = from_json(log_path.read_text(encoding="utf-8"))
    turns: list[dict[str, Any]] = payload["turns"]
    if not turns:
        raise ValueError(f"No turns found in log: {log_path}")

    if "initial_observation" in payload:
        first_obs = Observation.from_dict(payload["initial_observation"])
        initial_map = str(payload.get("initial_map", ""))
    else:
        first_obs = Observation.from_dict(turns[0]["observation"])
        initial_map = turns[0]["map"]

    task = first_obs.task

    dashboard = LiveDashboard(
        task=task,
        delay_seconds=delay_seconds,
        show_local_view=show_local_view,
    )

    total_reward = 0.0
    with dashboard:
        if initial_map:
            first_obs = Observation.from_dict({**first_obs.to_dict(), "full_map": initial_map})
        dashboard.on_reset(first_obs)
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        for entry in turns:
            obs = Observation.from_dict(entry["observation"])
            decision = AgentDecision.model_validate(entry["decision"])

            dashboard.on_thinking(obs)
            if delay_seconds > 0:
                time.sleep(delay_seconds * 0.5)

            dashboard.on_decision(obs, decision)
            if delay_seconds > 0:
                time.sleep(delay_seconds * 0.5)

            from virtual_world.agent.records import TurnRecord

            turn = TurnRecord(
                turn=entry["turn"],
                observation=entry["observation"],
                decision=entry["decision"],
                reward=entry["reward"],
                terminated=entry["terminated"],
                truncated=entry["truncated"],
                map=entry["map"],
            )
            total_reward += turn.reward
            dashboard.on_step(turn)

        from virtual_world.agent.records import EpisodeResult

        result = EpisodeResult(
            success=bool(payload.get("success", False)),
            steps=int(payload.get("steps", 0)),
            total_reward=total_reward,
            terminated=bool(payload.get("terminated", False)),
            truncated=bool(payload.get("truncated", False)),
            turns=[],
            log_path=log_path,
        )
        dashboard.on_complete(result)

    return bool(payload.get("success", False))
