"""Rich-based live terminal dashboard for agent episodes."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from virtual_world.agent.callbacks import AgentCallback
from virtual_world.agent.records import EpisodeResult, TurnRecord
from virtual_world.agent.schema import AgentDecision
from virtual_world.dashboard.render import (
    action_block,
    episode_banner,
    feedback_block,
    map_block,
    status_line,
    thought_block,
)
from virtual_world.harness.observation import Observation

if TYPE_CHECKING:
    from rich.console import Console, Group, RenderableType
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table


def _require_rich() -> None:
    try:
        import rich  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Live dashboard requires Rich: pip install 'virtual-world-agent[dashboard]'"
        ) from exc


class LiveDashboard(AgentCallback):
    """Real-time terminal UI that updates each agent turn."""

    def __init__(
        self,
        *,
        task: str,
        delay_seconds: float = 0.35,
        show_local_view: bool = True,
    ) -> None:
        _require_rich()
        from rich.console import Console
        from rich.live import Live

        self._task = task
        self._delay = delay_seconds
        self._show_local_view = show_local_view
        self._console = Console()
        self._live: Live | None = None

        self._observation: Observation | None = None
        self._decision: AgentDecision | None = None
        self._map = ""
        self._local_view = ""
        self._phase = "reset"
        self._total_reward = 0.0
        self._last_turn: TurnRecord | None = None

    def __enter__(self) -> LiveDashboard:
        from rich.live import Live

        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._live is not None:
            self._live.__exit__(*args)

    def on_reset(self, observation: Observation) -> None:
        self._observation = observation
        self._decision = None
        self._map = map_block(observation.full_map or self._map_from_obs(observation))
        self._local_view = self._format_local(observation)
        self._phase = "ready"
        self._total_reward = 0.0
        self._refresh()

    def on_thinking(self, observation: Observation) -> None:
        self._observation = observation
        self._decision = None
        self._phase = "thinking"
        self._refresh()

    def on_decision(self, observation: Observation, decision: AgentDecision) -> None:
        self._observation = observation
        self._decision = decision
        self._phase = "acting"
        self._refresh()

    def on_step(self, turn: TurnRecord) -> None:
        self._last_turn = turn
        self._total_reward += turn.reward
        obs = Observation.from_dict(turn.observation)
        self._observation = obs
        self._map = turn.map
        self._local_view = self._format_local(obs)
        self._phase = "done" if turn.terminated or turn.truncated else "observe"
        self._refresh()
        if self._delay > 0:
            time.sleep(self._delay)

    def on_complete(self, result: EpisodeResult) -> None:
        self._phase = episode_banner(result.success, result.steps, result.total_reward)
        self._refresh()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._build_layout())

    def _build_layout(self) -> RenderableType:
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table

        obs = self._observation
        status = status_line(
            obs,
            total_reward=self._total_reward,
            phase=self._phase,
        ) if obs else self._phase

        grid = Table.grid(expand=True)
        grid.add_column(ratio=2)
        grid.add_column(ratio=1)

        map_panel = Panel(
            self._map or "(no map)",
            title="World",
            border_style="cyan",
        )
        side = Group(
            Panel(status, title="Status", border_style="green"),
            Panel(
                action_block(self._decision),
                title="Action",
                border_style="yellow",
            ),
        )
        grid.add_row(map_panel, side)

        if self._show_local_view and self._local_view:
            grid.add_row(
                Panel(self._local_view, title="Local view (@ = you)", border_style="blue"),
                Panel(
                    thought_block(self._decision, phase=self._phase),
                    title="Agent reasoning",
                    border_style="magenta",
                ),
            )
        else:
            grid.add_row(
                Panel(
                    thought_block(self._decision, phase=self._phase),
                    title="Agent reasoning",
                    border_style="magenta",
                ),
            )

        feedback = feedback_block(obs) if obs else "—"
        root = Group(
            Panel(self._task, title="Task", border_style="white"),
            grid,
            Panel(feedback, title="Environment feedback", border_style="dim"),
        )
        return root

    @staticmethod
    def _format_local(observation: Observation) -> str:
        return "\n".join("".join(row) for row in observation.local_view)

    @staticmethod
    def _map_from_obs(observation: Observation) -> str:
        return observation.full_map or LiveDashboard._format_local(observation)
