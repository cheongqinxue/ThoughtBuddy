# `experiments/` — virtual-world experiment plugins

Each experiment registers agent tools via `register_experiment(name, create_tools)`.
Import `experiments` (or a specific experiment package) to register; `get_experiment_tools(settings)`
returns tools when `config.yaml` → `experiment.active` is set.

## Layout

| Path | Role |
|---|---|
| [`registry.py`](registry.py) | `register_experiment`, `list_experiments`, `get_experiment_tools` |
| [`chess/`](chess/) | Chess board + Stockfish opponent + browser UI |
| [`chess/service.py`](chess/service.py) | Session-backed games for agent + human players |
| [`chess/routes.py`](chess/routes.py) | HTTP API + UI route (`/thoughtbuddy/chess/ui`) |
| [`chess/static/index.html`](chess/static/index.html) | Live watch, human play, replay UI |
| [`chess/engine.py`](chess/engine.py) | Stockfish UCI integration, path resolution |
| [`chess/opponent.py`](chess/opponent.py) | Automatic opponent replies after agent moves |
| [`chess/validate.py`](chess/validate.py) | Fail-fast startup check when chess is active |
| [`../run.sh`](../../run.sh) | Repo-root experiment stack entrypoint — `./run.sh config/experiments/<run>.yaml start` |
| [`runner/`](runner/) | Generic stack orchestration Python for `run.sh` |
| [`chess/setup.py`](chess/setup.py) | Chess experiment setup entrypoint (preflight, config sync, analyze) |
| [`chess/plugin/`](chess/plugin/) | Agent tools, UI, Stockfish, materialization |
| [`../config/experiments-template/chess.yaml`](../../config/experiments-template/chess.yaml) | Copy-and-tweak experiment run template (LLM profile, ports, chess knobs). |

## Adding an experiment

1. Create `experiments/<name>/tools.py` with `create_tools(settings) -> list[BaseTool]`.
2. In `experiments/<name>/__init__.py`: `register_experiment("<name>", create_tools)`.
3. Import the package from `experiments/__init__.py`.
4. Set `experiment.active: <name>` in `config.yaml`.
5. Chess only: run `./scripts/install_stockfish` before enabling the experiment.
6. Chess UI: open `/thoughtbuddy/chess/ui` while `langgraph dev` is running.

AGENTS.md: [Phase 7 — First experiment plugin](../AGENTS.md#phase-7--first-experiment-plugin)

## Tests

| Test file | Covers |
|---|---|
| `tests/test_experiments.py` | Registry, config wiring |
| `tests/test_chess_experiment.py` | Chess session + tools |
| `tests/test_chess_engine.py` | Stockfish path resolution + opponent helpers |
| `tests/test_chess_stockfish_integration.py` | Live Stockfish move/reply (integration) |
| `tests/test_chess_sessions.py` | Session store + game service + tools |
