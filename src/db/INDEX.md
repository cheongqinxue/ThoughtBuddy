# `src/db/` — SQLite infrastructure

Single-file SQLite for alerts, tasks, and message archive. WAL + busy timeout configured at connect time.

## AGENTS.md

- [Persistence stack](#persistence-stack-chosen)
- [Scheduling architecture](#scheduling-architecture-target) — `data/thoughtbuddy.db`
- [Concurrency & failure modes](#concurrency--failure-modes-v1)

## Files

| File | Role |
|---|---|
| [`connection.py`](connection.py) | `connect(path)` — open DB, run `init_db`, set pragmas (`WAL`, `busy_timeout`) |
| [`schema.py`](schema.py) | DDL + `init_db()` migrations (`SCHEMA_VERSION = 6`) |
| [`__init__.py`](__init__.py) | Package marker |

## Tables (schema v3)

| Table | Store module | AGENTS.md |
|---|---|---|
| `alerts` | `store/alerts.py` | [Alert registry](#scheduling-architecture-target) |
| `tasks` | `store/tasks.py` | [Task queue](#scheduling-architecture-target), `UNIQUE(alert_id, occurrence_at)` |
| `message_log` | `store/messages.py` | [Message archive](#message-archive--context-compaction-target) |
| `compaction_points` | `store/compaction.py` | [Compaction](#message-archive--context-compaction-target) |
| `schema_migrations` | internal | — |

## Not in schema (documented gaps)

| AGENTS.md mentions | Status |
|---|---|
| `goals` table | **Not implemented** — long-term goal store still future work |
| Progress journal | **Not implemented** — no table |

## Callers

- `runtime/service.py` — primary runtime connection
- `tools/scheduling.py` — opens its own connection per tool call via `connect(db_path)`

## Tests

- Indirectly via `tests/test_scheduling.py`, `tests/test_context.py`, `tests/test_scheduling_tools.py` (temp DB paths)

## Gaps / redundancies

- **Dual connection pattern:** runtime holds one long-lived `conn`; scheduling tools open/close per invocation. Acceptable for v1; could unify later.
- **No SQLAlchemy** — raw sqlite3 by design (AGENTS.md persistence stack).
