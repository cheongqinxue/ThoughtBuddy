# `src/utils/` — shared helpers

Cross-cutting utilities with no ThoughtBuddy domain logic. Imported by tools, stores, and scheduling.

## AGENTS.md

- [API retries with tenacity](#8-api-retries-with-tenacity) → `api_retry.py`

## Files

| File | Role | Used by |
|---|---|---|
| [`api_retry.py`](api_retry.py) | `ApiRetry`, `create_api_retry`, retry classification helpers | `tools/websearch/providers/langsearch.py` |
| [`logging/`](logging/) | Bootstrap, secret redaction, wake/thread context | `webapp.py`, `runtime/wake.py` |
| [`observability.py`](observability.py) | Wake traces, log bridge, flush/shutdown (v1: Langfuse) | `runtime/wake.py`, `logging/config.py`, integration tests |
| [`time.py`](time.py) | UTC parse/format helpers | `store/*`, `scheduling/*` |
| [`anchored_edit.py`](anchored_edit.py) | `read_file`, `edit_file`, `apply_edit` — anchored markdown deltas | `tools/scratchpad.py`, `tools/progress.py` |
| [`__init__.py`](__init__.py) | Package marker | — |

## `api_retry.py`

Tenacity-backed retry for external HTTP:

- **Retry:** timeouts, network errors, 429, 5xx
- **Do not retry:** 400, 401, 403, 404, 422

## Tests

| Test file | Covers |
|---|---|
| `tests/test_api_retry.py` | Retry classification |
| `tests/test_anchored_edit.py` | Anchored file edit logic |

## Gaps / redundancies

- **LangGraph Studio session ids** — Studio traces use default callbacks; per-thread session mapping depends on Studio config.
- **Time helpers** — small; could stay here or move to `store` if only used there (currently shared widely — correct placement).

## Candidates for future utils

- Structured logging bootstrap (if `LOG_LEVEL` wired at startup)
- Shared JSON schema helpers if tool payloads grow
