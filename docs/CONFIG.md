# ThoughtBuddy configuration reference

Single source of truth for **how config is loaded** and **where values are read** in the codebase. Use this when adding settings or hunting stray config reads.

## Loading model

| Source | Contains | Loader |
|---|---|---|
| `config.yaml` | All non-sensitive app settings | `Settings.load()` |
| `.env` | API keys and secrets only | `Settings.load()` |
| `Settings(...)` constructor | Unit tests and explicit overrides | Direct instantiation |

**Do not** add new per-setting environment variables. **Do not** use LangGraph `RunnableConfig.configurable` for app settings.

```python
from config import Settings

settings = Settings.load()  # config.yaml + .env
```

Optional overrides for tests:

```python
settings = Settings.load(config_path=tmp_path / "config.yaml", env_file="/dev/null")
settings = Settings(openai_api_key="test-key", ...)  # fully explicit
```

## Secrets (`.env` only)

| Env var | Settings field | Used by |
|---|---|---|
| `OPENAI_API_KEY` | `openai_api_key` | `src/llm.py` → ChatOpenAI |
| `LANGSEARCH_API_KEY` | `langsearch_api_key` | `src/tools/websearch/providers/langsearch.py` |
| `EVENT_WEBHOOK_SECRET` | `event_webhook_secret` | `POST /thoughtbuddy/tasks/event` |
| `LANGFUSE_PUBLIC_KEY` | _(SDK env only)_ | `langfuse` SDK via `utils/observability.py` |
| `LANGFUSE_SECRET_KEY` | _(SDK env only)_ | `langfuse` SDK via `utils/observability.py` |
| `LANGFUSE_BASE_URL` | _(SDK env only)_ | Optional; override via `observability.langfuse.base_url` |

**LangGraph-only (not in Settings):**

| Env var | Used by |
|---|---|
| `LANGSMITH_API_KEY` | `langgraph dev` via `langgraph.json` → `"env": ".env"` |

Document every secret in `.env-template`. Never commit `.env`.

## Non-sensitive settings (`config.yaml`)

See committed `config.yaml` for defaults. Field mapping lives in `src/config.py` → `_YAML_FIELD_PATHS`.

| YAML section | Settings fields |
|---|---|
| `llm` | `openai_base_url`, `agent_model`, `tokenizer_model`, `enable_thinking`, `llm_temperature`, `llm_max_tokens`, `llm_max_retries`, `llm_context_window_tokens` |
| `logging` | `log_level`, `retry_log_each_attempt`, `retry_max_attempts`, `retry_min_wait_seconds`, `retry_max_wait_seconds` |
| `runtime` | `thoughtbuddy_db_path`, `thoughtbuddy_lock_path`, `heartbeat_interval_seconds`, `stale_task_seconds`, `default_thread_id`, `upcoming_alerts_window_seconds` |
| `context` | `context_retain_messages`, `context_compact_token_threshold` |
| `webfetch` | `webfetch_timeout_seconds`, `webfetch_network_idle_timeout_ms`, `webfetch_post_load_wait_ms`, `webfetch_max_chars` |
| `memory` | `memory_store_path`, `memory_collection_name`, `memory_embedding_model`, `memory_embedding_device`, `memory_recall_multiplier` |
| `scratchpad` | `scratchpad_directory` |
| `monitor` | `monitor_max_concurrent`, `monitor_default_timeout_seconds`, `monitor_min_expected_seconds`, `monitor_min_expected_heartbeats`, `monitor_log_tail_bytes` |
| `experiment` | `active_experiment` |
| `experiment.chess` | `chess_engine_path`, `chess_agent_color`, `chess_agent_generation`, `chess_skill_level`, `chess_move_time_seconds`, `chess_engine_timeout_seconds` |
| `observability.langfuse` | `langfuse_enabled`, `langfuse_base_url`, `langfuse_log_min_level` |

## Config consumption map

Every place that reads `Settings` or loads configuration:

| File | How config is obtained | Fields used |
|---|---|---|
| `src/agent.py` | `Settings.load()` or injected | all (via `create_chat_model`, `get_tools`) |
| `src/runtime/service.py` | `Settings.load()` or injected | db path, lock, heartbeat, stale task |
| `src/scheduling/heartbeat.py` | injected `Settings` | compaction, wake thread id, upcoming window |
| `src/runtime/wake.py` | injected `Settings` | context compaction, default thread id |
| `src/llm.py` | injected `Settings` | `openai_*`, `agent_model`, `llm_temperature`, `llm_max_tokens`, `llm_max_retries` |
| `src/tools/__init__.py` | injected `Settings` | passed to tool factories + `active_experiment` |
| `src/experiments/registry.py` | injected `Settings` | `active_experiment` |
| `src/tools/websearch/` | injected `Settings` | `langsearch_api_key`, `websearch_provider`, shared retry settings |
| `src/tools/webfetch/` | injected `Settings` | webfetch timeouts/limits, shared retry settings |
| `src/utils/api_retry.py` | injected `Settings` | `retry_*` via `create_api_retry()` |
| `src/utils/observability.py` | injected `Settings` + `LANGFUSE_*` env | wake tracing, log bridge (v1 Langfuse) |
| `src/runtime/wake.py` | injected `Settings` | Langfuse callbacks on graph invoke |
| `src/agent.py` | `Settings.load()` in `make_graph` | Studio default Langfuse callbacks; `run_agent` direct traces |
| `src/tools/scheduling.py` | injected `Settings` | `thoughtbuddy_db_path` |
| `src/context/compact.py` | injected `Settings` | context thresholds |
| `src/memory/__init__.py` | `Settings.load()` or injected | memory store paths/model |
| `src/utils/anchored_edit.py` | path argument | `read_file`, `edit_file` for any markdown path |
| `src/tools/scratchpad.py` | injected `Settings` | `scratchpad_directory` → `SCRATCHPAD.md` |
| `src/tools/progress.py` | injected `Settings` | `scratchpad_directory` → `PROGRESS.md` |
| `src/webapp.py` | via `create_runtime()` → `Settings.load()` | indirect |

## LangGraph checkpoint config (not app settings)

These use LangGraph's `config={"configurable": {"thread_id": ...}}` for **conversation checkpointing** only. Values come from `Settings.default_thread_id`, not from LangGraph server config.

| File | Purpose |
|---|---|
| `src/agent.py` → `run_agent()` | per-turn thread id |
| `src/runtime/wake.py` → `invoke_wake_cycle()` | wake-cycle thread id |
| `src/scheduling/heartbeat.py` | passes thread id to wake cycle |

`make_graph(_config: RunnableConfig)` in `src/agent.py` **ignores** LangGraph runtime config and calls `create_agent()` → `Settings.load()`. Do not reintroduce app settings through LangGraph configurable fields.

## Test fixtures

| File | Pattern |
|---|---|
| `tests/conftest.py` | `Settings(...)` with fake credentials |
| `tests/test_config.py` | `Settings.load(config_path=..., env_file=...)` |
| `tests/test_memory.py` | `Settings(...)` with temp paths |
| `tests/test_scheduling_tools.py` | `Settings(...)` with temp db |
| `tests/integration/test_websearch_live.py` | `Settings.load()` + real `.env` |

## Adding a new setting

1. Add field to `Settings` in `src/config.py` with a pydantic default.
2. Add YAML path mapping to `_YAML_FIELD_PATHS`.
3. Add value to committed `config.yaml`.
4. If secret, add env var to `_SECRET_ENV_VARS` and `.env-template` only.
5. Update this file's tables.
6. Pass via injected `Settings` — do not read env/yaml in feature code.

## Removed / deprecated patterns

- ~~`Settings.from_env()`~~ — removed; use `Settings.load()`.
- ~~Per-setting env vars (`HEARTBEAT_INTERVAL_SECONDS`, etc.)~~ — use `config.yaml`.
- ~~LangGraph configurable for app settings~~ — not used; checkpoint thread id only.
- ~~GCP / EXTRACT_MODEL vars in `.env-template`~~ — were unused legacy; removed.
