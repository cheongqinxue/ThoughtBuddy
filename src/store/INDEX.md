# `src/store/` — SQLite data access

CRUD layer over `db/schema.py` tables. No business logic for heartbeat timing or graph invoke.

## AGENTS.md

- [Alert registry](#scheduling-architecture-target) → `alerts.py`
- [Task queue](#scheduling-architecture-target) → `tasks.py`
- [Message archive](#message-archive--context-compaction-target) → `messages.py`, `compaction.py`

## Files

| File | Table(s) | Responsibility |
|---|---|---|
| [`alerts.py`](alerts.py) | `alerts` | Create/update/list alerts; cron + one-shot; `get_due_alerts`, `list_upcoming` |
| [`tasks.py`](tasks.py) | `tasks` | Enqueue user tasks, pending/running/completed/ignored lifecycle, stale reaper |
| [`messages.py`](messages.py) | `message_log`, `message_log_fts` | Append-only archive; datetime pagination; BM25 search |
| [`compaction.py`](compaction.py) | `compaction_points` | Latest summary + boundary seq per thread |

## Task status model

Matches AGENTS.md: `pending` → `running` (heartbeat) → `completed` | `ignored` (agent via `update_task`). No `deferred`.

## Key constraints

- `UNIQUE(alert_id, occurrence_at)` on tasks — idempotent materialization (AGENTS.md)
- `message_log` never deletes rows — full history; agent reads via `history_read` / `history_search`

## Callers

| Module | Stores used |
|---|---|
| `scheduling/materialize.py`, `heartbeat.py` | alerts, tasks |
| `scheduling/schedule_map.py` | alerts, tasks |
| `tools/scheduling.py` | alerts, tasks (own DB connection) |
| `context/build.py`, `compact.py` | messages, compaction |
| `runtime/wake.py` | messages |
| `webapp.py` | tasks, alerts (ops routes) |

## Tests

| Test file | Covers |
|---|---|
| `tests/test_scheduling.py` | tasks, alerts via materialization |
| `tests/test_scheduling_tools.py` | alert/task tool CRUD |
| `tests/test_context.py` | messages, compaction |

## Gaps / redundancies

- **No `GoalStore`** — AGENTS.md north star mentions long-term goals; no store module yet.
- **`Alert` / `Task` dataclasses** — parallel to DB rows; no ORM.
