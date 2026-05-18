# Virtual World Agent

Production-oriented LLM agent harness for a text-based grid world. Built incrementally; see milestones below.

## Step 2 (current): Gym-like harness

`GridWorldEnv` wraps the grid world with a familiar `reset` / `step` API, structured `Observation` objects (JSON + LLM prompts), and an `ActionSpace` that decodes action IDs or dicts from structured LLM output.

### Harness quick start

```python
from virtual_world import EnvConfig, GridWorldEnv
from virtual_world.layout import MINI_KEY_DOOR_SCENARIO

env = GridWorldEnv(
    EnvConfig(
        layout=MINI_KEY_DOOR_SCENARIO,
        task="Pick up the key, unlock the door, and reach the goal.",
        max_steps=50,
    )
)
obs, info = env.reset()
print(obs.to_prompt())
print(info["action_space"])  # catalog for tool / JSON schema

obs, reward, done, truncated, info = env.step("move_east")
print(obs.feedback, reward, done)
```

### Agent action format (for Step 3)

The LLM should emit one of:

```json
{"action_id": "move_east"}
```

```json
{"action": "move", "direction": "east"}
```

Valid `action_id` values: `move_north`, `move_south`, `move_east`, `move_west`, `turn_left`, `turn_right`, `pick_up`, `use`, `wait`.

### Observation serialization

```python
obs.to_dict()   # JSON-ready dict
obs.to_json()   # pretty-printed JSON string
obs.to_prompt() # natural-language block for the model
env.export_state_json()  # full episode snapshot for logs
```

## Step 1: Grid world environment

The `virtual_world` package provides a typed 2D grid with walls, goals, keys, locked doors, and obstacles. Maps are defined as ASCII art and parsed at load time.

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
pytest -v
```

### Quick manual demo

```python
from virtual_world import Action, Direction, VirtualWorld
from virtual_world.layout import SIMPLE_GOAL_SCENARIO

world = VirtualWorld.from_layout(SIMPLE_GOAL_SCENARIO)
print(world.reset().ascii_map)

for _ in range(5):
    result = world.step(Action.move(Direction.EAST))
    print(result.state.message)

print(world.render())
```

### Map legend

| Char | Meaning        |
|------|----------------|
| `.`  | Floor          |
| `#`  | Wall           |
| `G`  | Goal           |
| `@`  | Agent start    |
| `k`  | Key (id: main) |
| `D`  | Locked door    |
| `O`  | Obstacle       |

### Roadmap

1. Grid world
2. **Gym-like harness** (this step)
3. LLM agent core (structured outputs)
4. CLI dashboard / visualizer
