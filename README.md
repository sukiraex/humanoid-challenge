# Virtual World Agent

An LLM agent harness for a text-based grid world. The agent perceives its environment, reasons in structured JSON, chooses discrete actions, and completes goal-directed tasks (reach a goal, pick up keys, unlock doors).

The core of this project is the **harness**: a clean boundary between the simulation and any language model.

## Requirements

- Python 3.11+
- Optional: OpenAI or Anthropic API key for live LLM runs

## Quick start

```bash
# Clone and enter the repo, then:
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -e ".[dev,dashboard]"
pytest -v
```

## How to run

### 1. Mock agent (no API key)

Runs a scripted policy so you can verify the full pipeline offline:

```bash
vw-agent --scenario simple_goal --provider mock
vw-agent --scenario key_door --provider mock
```

### 2. Live terminal dashboard

Shows the map, reasoning, action, and feedback updating each turn:

```bash
vw-agent --scenario key_door --provider mock --live
vw-agent --scenario key_door --provider mock --live --delay 0.6
```

### 3. Real LLM (OpenAI or Anthropic)

```bash
pip install -e ".[openai]"       # or: pip install -e ".[anthropic]"
copy .env.example .env           # Windows
# cp .env.example .env           # macOS / Linux
```

Add your key to `.env`:

```env
OPENAI_API_KEY=sk-...
AGENT_MODEL=gpt-4o-mini
```

Run:

```bash
vw-agent --scenario key_door --provider openai
vw-agent --scenario key_door --provider openai --live
```

Anthropic:

```env
ANTHROPIC_API_KEY=sk-ant-...
AGENT_MODEL=claude-sonnet-4-20250514
```

```bash
vw-agent --scenario key_door --provider anthropic
```

### 4. Replay a saved episode log

```bash
vw-agent --replay examples/sample_episode_key_door.json --delay 0.4
```

### 5. Run tests

```bash
pytest -v
```

### CLI reference

| Flag | Description |
|------|-------------|
| `--scenario` | `simple_goal` or `key_door` |
| `--provider` | `mock`, `openai`, or `anthropic` |
| `--live` | Rich live dashboard |
| `--delay` | Seconds between turns (default `0.35`) |
| `--log-dir` | Where JSON logs are saved (default `logs/`) |
| `--no-log` | Skip writing a log file |
| `--replay` | Replay a saved `episode_*.json` |
| `--model` | Override model name from `.env` |

## Example input and output (for submission)

### Example input

The agent receives a natural-language **task** plus a structured **observation** each turn. The model must respond with JSON:

```json
{
  "reasoning": "The key is one cell east. I will move there and pick it up.",
  "action_id": "move_east"
}
```

A representative observation prompt looks like:

```
## Task
Pick up the key (k), unlock the locked door (D) in front of you, then reach the goal (G).

## Status
Step: 2 / 30
Position: (2, 1), facing north
Keys held: (none)
Feedback: Moved east.

## Local view (@ = you)
...
```

### Example output (episode log)

A full run is saved under **`examples/`**:

| File | Description |
|------|-------------|
| [`examples/sample_episode_key_door.json`](examples/sample_episode_key_door.json) | **Primary submission example** — successful key → door → goal episode |
| `examples/episode_*.json` | Additional timestamped runs |

Each log contains:

- `success`, `steps`, `total_reward`
- `initial_observation` and `initial_map` (starting state)
- `turns[]` — per turn: `decision` (reasoning + `action_id`), `observation`, `feedback`, ASCII `map`

Replay visually:

```bash
vw-agent --replay examples/sample_episode_key_door.json --delay 0.4
```

New logs are written to `logs/` on every run unless `--no-log` is passed.

## Scenarios

| Name | Task |
|------|------|
| `simple_goal` | Walk east to the goal tile `G` |
| `key_door` | Pick up key `k`, unlock door `D`, reach `G` |

### Map legend

| Char | Meaning |
|------|---------|
| `.` | Floor |
| `#` | Wall |
| `G` | Goal |
| `@` | Agent (in local view) |
| `k` | Key |
| `D` / `d` | Locked / unlocked door |
| `O` | Obstacle |

## Design notes

### Architecture

The system is split into four layers:

1. **`VirtualWorld`** — simulation (physics, entities, win conditions)
2. **`GridWorldEnv`** — Gym-like `reset` / `step` API
3. **`AgentLoop` + `LLMClient`** — observe → reason → act loop
4. **`LiveDashboard`** — real-time CLI visualization

This keeps the world independent from any specific model or UI.

### How represent observations

Observations are a structured `Observation` dataclass (JSON-serializable) plus a `to_prompt()` text view for the LLM.

Include:

- **Task text** — explicit goal so the model knows *why* it is acting
- **Egocentric local view** (5×5 grid, `@` = agent) — enough spatial context without giving a full map cheat sheet by default
- **Position, facing, inventory, step count** — state the model cannot infer from one local slice alone
- **Environment feedback** — the result of the last action (“Blocked: wall”, “Picked up key 'main'”) so the model can correct mistakes
- **Optional full map** — enabled for the live dashboard (`include_full_map=True`)

I deliberately separate **structured data** (for logging, tests, tool schemas) from **prompt text** (for the model). The harness owns both; the LLM never sees raw Python objects.

### Why this action space

I use **9 discrete `action_id` strings** instead of free-form text:

`move_north`, `move_south`, `move_east`, `move_west`, `turn_left`, `turn_right`, `pick_up`, `use`, `wait`

Reasons:

- **Validatability** — every action is checked against `ActionSpace` before `env.step()`; invalid moves never reach the simulator
- **Simple JSON schema** — one `action_id` field is easy to enforce with `response_format=json_object` (OpenAI) or prompt instructions
- **Enough expressiveness** — movement is absolute (cardinal), which is easier for LLMs than relative “move forward” while still requiring `turn_*` + `use` for doors
- **Gym familiarity** — discrete action IDs map cleanly to a classic RL interface

The model outputs chain-of-thought in `reasoning` but only **commits** via `action_id`, separating deliberation from execution.

### What worked

- **Structured JSON decisions** (Pydantic validation) drastically reduced parse errors vs. free-text commands
- **Feedback field** in observations helped mock and live agents recover from blocked moves
- **Mock provider** made CI, demos, and submission reproducible without API costs
- **Episode JSON logs** double as submission artifacts and replay input for the dashboard
- **Incremental harness** (`VirtualWorld` → `GridWorldEnv` → `AgentLoop`) kept each layer testable in isolation

### What did not / limitations

- **Partial observability** — a 5×5 local view alone is hard for larger mazes; the `key_door` mini scenario was sized for reliability. Full-map mode helps debugging but makes the task easier.
- **Real LLM variance** — without API keys in CI, most automated tests use `MockLLMClient`; live models may need prompt tuning or more steps on `KEY_AND_DOOR_SCENARIO` (the large maze).
- **No spatial memory** — the agent does not persist a learned map; it relies on the prompt and short history window (last 3 turns).
- **Text-only UI** — Rich terminal dashboard rather than pygame; fits the ASCII world and keeps dependencies light.

## Project structure

```
src/virtual_world/
  world.py, layout.py, entities.py   # Simulation
  harness/                           # GridWorldEnv, Observation, ActionSpace
  agent/                             # AgentLoop, LLM clients, scenarios
  dashboard/                         # Live Rich UI + replay
  cli.py                             # vw-agent entry point
examples/                            # Sample episode logs (submission)
tests/                               # 31 pytest tests
```

## Install extras

```bash
pip install -e ".[dev]"         # pytest
pip install -e ".[dashboard]"   # Rich live UI
pip install -e ".[openai]"      # OpenAI API
pip install -e ".[anthropic]"   # Anthropic API
pip install -e ".[all]"         # everything
```
