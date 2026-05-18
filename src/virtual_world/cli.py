"""Command-line entry point for running the LLM grid-world agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from virtual_world.agent.llm import create_llm_client
from virtual_world.agent.loop import AgentLoop
from virtual_world.agent.scenarios import SCENARIOS, get_scenario
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = get_scenario(args.scenario)

    env = GridWorldEnv(
        EnvConfig(
            layout=scenario.layout,
            task=scenario.task,
            max_steps=scenario.max_steps,
        ),
    )

    script = list(scenario.mock_script) if args.provider == "mock" else None
    llm = create_llm_client(args.provider, script=script, model=args.model)

    loop = AgentLoop(env, llm)
    log_dir = None if args.no_log else args.log_dir
    result = loop.run(log_dir=log_dir)

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


if __name__ == "__main__":
    sys.exit(main())
