# Virtual World Agent

Production-oriented LLM agent harness for a text-based grid world. Built incrementally; see milestones below.

## Step 4 (current): Live CLI dashboard

A Rich-powered terminal dashboard shows the map, agent reasoning, chosen action, and environment feedback in real time.

### Install dashboard support

```bash
pip install -e ".[dashboard]"
# or: pip install -e ".[all]"
```

### Live run (mock or real LLM)

```bash
vw-agent --scenario key_door --provider mock --live
vw-agent --scenario key_door --provider openai --live --delay 0.5
```

### Replay a saved episode log

```bash
vw-agent --replay examples/episode_20260518T165941Z.json --delay 0.4
```

Press `Ctrl+C` to exit. Episode JSON logs still save to `logs/` unless `--no-log`.

---

## Step 3: LLM agent core

The agent observes the world, reasons with chain-of-thought JSON, and selects actions in a loop until success or step limit.

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
pytest -v
```

### Run without an API key (mock provider)

```bash
vw-agent --scenario simple_goal --provider mock
vw-agent --scenario key_door --provider mock
```

Episode logs are written to `logs/episode_<timestamp>.json` (reasoning, actions, ASCII maps per turn).

### Run with a real LLM

```bash
pip install -e ".[openai]"      # or: pip install -e ".[anthropic]"
copy .env.example .env          # add your API key
vw-agent --scenario key_door --provider openai --model gpt-4o-mini
# vw-agent --scenario key_door --provider anthropic
```

### Structured decision format

The model must return JSON:

```json
{
  "reasoning": "The key is one cell east. I will move there and pick it up.",
  "action_id": "move_east"
}
```

`action_id` must be one of: `move_north`, `move_south`, `move_east`, `move_west`, `turn_left`, `turn_right`, `pick_up`, `use`, `wait`.

### Programmatic usage

```python
from virtual_world import AgentLoop, EnvConfig, GridWorldEnv, MockLLMClient
from virtual_world.agent.scenarios import get_scenario

scenario = get_scenario("key_door")
env = GridWorldEnv(EnvConfig(layout=scenario.layout, task=scenario.task))
llm = MockLLMClient(list(scenario.mock_script))
result = AgentLoop(env, llm).run(log_dir="logs")
print(result.success, result.log_path)
```

## Step 2: Gym-like harness

`GridWorldEnv` wraps the grid world with `reset` / `step`, structured `Observation` objects (JSON + LLM prompts), and `ActionSpace` decoding.

```python
from virtual_world import EnvConfig, GridWorldEnv
obs, info = env.reset()
obs, reward, done, truncated, info = env.step("move_east")
```

## Step 1: Grid world environment

Typed 2D grid with walls, goals, keys, locked doors, and obstacles. Maps are ASCII art.

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
2. Gym-like harness
3. LLM agent core
4. **Live CLI dashboard** (this step)
