"""Hooks for observing the agent loop (dashboard, logging, tests)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from virtual_world.agent.records import EpisodeResult, TurnRecord
from virtual_world.agent.schema import AgentDecision
from virtual_world.harness.observation import Observation


@runtime_checkable
class AgentCallback(Protocol):
    """Optional observer for :class:`AgentLoop` lifecycle events."""

    def on_reset(self, observation: Observation) -> None: ...

    def on_thinking(self, observation: Observation) -> None: ...

    def on_decision(self, observation: Observation, decision: AgentDecision) -> None: ...

    def on_step(self, turn: TurnRecord) -> None: ...

    def on_complete(self, result: EpisodeResult) -> None: ...
