"""Command-line entry point for running the LLM grid-world agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from virtual_world.agent.llm import create_llm_client
from virtual_world.agent.loop import AgentLoop
from virtual_world.agent.scenarios import SCENARIOS, get_scenario
from virtual_world.dashboard.replay import replay_episode
from virtual_world.harness import EnvConfig, GridWorldEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an LLM agent in the virtual grid world.",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="simple_goal",
        help="Scenario layout and task (default: simple_goal)",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "openai", "anthropic"],
        default="mock",
        help="LLM provider (default: mock — no API key required)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (OPENAI_API_KEY / ANTHROPIC_API_KEY in env)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for JSON episode logs (default: logs/)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable writing episode logs",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Show a Rich live dashboard (thoughts, map, actions)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Seconds between turns in live/replay mode (default: 0.35)",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        metavar="LOG.json",
        help="Replay a saved episode log with the live dashboard",
    )
    return parser


def _run_live_episode(
    env: GridWorldEnv,
    llm: object,
    *,
    task: str,
    log_dir: Path | None,
    delay: float,
) -> int:
    from virtual_world.dashboard import LiveDashboard

    dashboard = LiveDashboard(task=task, delay_seconds=delay)
    with dashboard:
        result = AgentLoop(env, llm).run(  # type: ignore[arg-type]
            log_dir=log_dir,
            callbacks=[dashboard],
        )

    if result.log_path:
        print(f"Log written to: {result.log_path}")
    return 0 if result.success else 1


def _print_plain_summary(result: object) -> int:
    from virtual_world.agent.records import EpisodeResult

    assert isinstance(result, EpisodeResult)
    status = "SUCCESS" if result.success else "FAILED"
    print(f"\n=== Episode {status} ===")
    print(f"Steps: {result.steps}  Reward: {result.total_reward:.3f}")
    print(f"Terminated: {result.terminated}  Truncated: {result.truncated}")
    if result.log_path:
        print(f"Log written to: {result.log_path}")

    for turn in result.turns:
        decision = turn.decision
        print(
            f"\n[Turn {turn.turn}] action={decision['action_id']}\n"
            f"  Reasoning: {decision['reasoning']}\n"
            f"  Feedback: {turn.observation['feedback']}"
        )
        print(turn.map)

    return 0 if result.success else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.replay is not None:
        if not args.replay.is_file():
            print(f"Replay file not found: {args.replay}", file=sys.stderr)
            return 2
        if not args.live:
            print("Note: --replay uses the live dashboard; pass --live to animate.", file=sys.stderr)
        success = replay_episode(args.replay, delay_seconds=args.delay)
        return 0 if success else 1

    scenario = get_scenario(args.scenario)
    env = GridWorldEnv(
        EnvConfig(
            layout=scenario.layout,
            task=scenario.task,
            max_steps=scenario.max_steps,
            include_full_map=args.live,
        ),
    )

    script = list(scenario.mock_script) if args.provider == "mock" else None
    llm = create_llm_client(args.provider, script=script, model=args.model)
    log_dir = None if args.no_log else args.log_dir

    if args.live:
        return _run_live_episode(
            env,
            llm,
            task=scenario.task,
            log_dir=log_dir,
            delay=args.delay,
        )

    result = AgentLoop(env, llm).run(log_dir=log_dir)
    return _print_plain_summary(result)


if __name__ == "__main__":
    sys.exit(main())
