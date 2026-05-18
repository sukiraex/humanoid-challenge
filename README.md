# Virtual World Agent

Production-oriented LLM agent harness for a text-based grid world. Built incrementally; see milestones below.

## Step 1 (current): Grid world environment

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

1. **Grid world** (this step)
2. Gym-like harness (observation / action API)
3. LLM agent core (structured outputs)
4. CLI dashboard / visualizer
