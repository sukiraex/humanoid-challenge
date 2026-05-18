"""Episode record types shared by the agent loop and dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
