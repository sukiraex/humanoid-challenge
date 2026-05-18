"""Agent loop: observe → reason → act until the episode ends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from virtual_world.agent.llm import LLMClient
from virtual_world.agent.schema import AgentDecision
from virtual_world.harness.env import GridWorldEnv
from virtual_world.harness.observation import Observation
from virtual_world.harness.serialization import to_json


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """Log entry for a single agent step."""

    turn: int
    observation: dict[str, Any]
    decision: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    map: str


@dataclass
class EpisodeResult:
    """Summary of a completed (or halted) agent episode."""

    success: bool
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    turns: list[TurnRecord] = field(default_factory=list)
    log_path: Path | None = None


class AgentLoop:
    """Runs the LLM agent against a :class:`GridWorldEnv` until done or max turns."""

    def __init__(
        self,
        env: GridWorldEnv,
        llm: LLMClient,
        *,
        max_retries: int = 2,
        history_window: int = 3,
    ) -> None:
        self._env = env
        self._llm = llm
        self._max_retries = max_retries
        self._history_window = history_window

    def run(self, *, log_dir: Path | None = None) -> EpisodeResult:
        observation, _info = self._env.reset()
        history: list[tuple[Observation, AgentDecision]] = []
        turns: list[TurnRecord] = []
        total_reward = 0.0
        terminated = False
        truncated = False
        turn_index = 0

        while not (terminated or truncated):
            decision = self._decide_with_retries(observation, history)
            observation, reward, terminated, truncated, _info = self._env.step(decision.action_id)
            total_reward += reward
            turn_index += 1

            turns.append(
                TurnRecord(
                    turn=turn_index,
                    observation=observation.to_dict(),
                    decision=decision.model_dump(),
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    map=self._env.render(),
                )
            )
            history.append((observation, decision))
            if len(history) > self._history_window:
                history = history[-self._history_window :]

        result = EpisodeResult(
            success=observation.success,
            steps=observation.step,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            turns=turns,
        )
        if log_dir is not None:
            result.log_path = self._write_log(log_dir, result)
        return result

    def _decide_with_retries(
        self,
        observation: Observation,
        history: list[tuple[Observation, AgentDecision]],
    ) -> AgentDecision:
        last_error: Exception | None = None
        for _attempt in range(self._max_retries + 1):
            try:
                return self._llm.decide(
                    observation,
                    self._env.action_space,
                    history=history or None,
                )
            except (ValueError, ValidationError) as exc:
                last_error = exc
        raise RuntimeError(f"LLM failed after {self._max_retries + 1} attempts") from last_error

    def _write_log(self, log_dir: Path, result: EpisodeResult) -> Path:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = log_dir / f"episode_{timestamp}.json"
        payload = {
            "success": result.success,
            "steps": result.steps,
            "total_reward": result.total_reward,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "turns": [asdict(turn) for turn in result.turns],
        }
        path.write_text(to_json(payload), encoding="utf-8")
        return path
