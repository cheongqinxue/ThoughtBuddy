# `experiments/chess/plugin/` — chess virtual-world plugin

Loaded when `config.yaml` → `experiment.active: chess`. Registers `world_chess_*` tools, HTTP UI, and heartbeat task materialization.

| File | Role |
|---|---|
| [`tools.py`](tools.py) | Agent tools: `world_chess_status`, `world_chess_move`, `world_chess_reset` |
| [`service.py`](service.py) | `ChessGameService` — sessions, moves, SQLite |
| [`game.py`](game.py) | `ChessSession` domain model |
| [`engine.py`](engine.py) | Stockfish UCI subprocess |
| [`opponent.py`](opponent.py) | Agent vs engine turn logic |
| [`materialize.py`](materialize.py) | `chess_start` / `chess_continue` queue tasks |
| [`routes.py`](routes.py) | REST + browser UI under `/thoughtbuddy/chess/` |
| [`validate.py`](validate.py) | Startup validation (engine reachable) |
| [`static/`](static/) | Chess UI assets |

Experiment setup: [`../setup.py`](../setup.py). Stack runner: repo-root [`run.sh`](../../../../run.sh) + [`../../runner/`](../../runner/).
