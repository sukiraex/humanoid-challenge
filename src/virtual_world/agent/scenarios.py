"""Named scenarios for demos, tests, and CLI runs."""

from __future__ import annotations

from dataclasses import dataclass

from virtual_world.layout import MINI_KEY_DOOR_SCENARIO, SIMPLE_GOAL_SCENARIO


@dataclass(frozen=True, slots=True)
class Scenario:
    """Bundled layout, task description, and optional mock action script."""

    name: str
    layout: str
    task: str
    mock_script: tuple[str, ...]
    max_steps: int = 50


SCENARIOS: dict[str, Scenario] = {
    "simple_goal": Scenario(
        name="simple_goal",
        layout=SIMPLE_GOAL_SCENARIO,
        task="Move east along the corridor until you stand on the goal tile (G).",
        mock_script=("move_east", "move_east", "move_east", "move_east", "wait"),
        max_steps=30,
    ),
    "key_door": Scenario(
        name="key_door",
        layout=MINI_KEY_DOOR_SCENARIO,
        task=(
            "Pick up the key (k), unlock the locked door (D) in front of you, "
            "then reach the goal (G)."
        ),
        mock_script=(
            "move_east",
            "pick_up",
            "move_west",
            "turn_right",
            "turn_right",
            "use",
            "move_south",
            "move_east",
            "move_east",
            "wait",
        ),
        max_steps=30,
    ),
}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        available = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario {name!r}. Choose from: {available}") from exc
