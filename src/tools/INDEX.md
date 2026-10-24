# `src/tools/` — agent tool registry

All LangChain tools exposed to the ReAct agent. Provider-specific code stays in submodules; LLM sees generic tool names.

## AGENTS.md

- [Agent scheduling tools](#agent-scheduling-tools-target)
- [Generic tool names](#3-generic-tool-names-swappable-providers)
- [Observe the world](#wake-cycle-target-behavior) — `websearch`, `webfetch`
- [Persistent scratchpad](#persistent-scratchpad) — `edit_scratchpad`, `edit_progress`

## Files

| File | Agent tools | Backend |
|---|---|---|
| [`__init__.py`](__init__.py) | `get_tools()` registry | Aggregates registered factories; applies `wrap_tool_with_boundary` to every tool |
| [`tool_boundary.py`](tool_boundary.py) | `wrap_tool_with_boundary`, `run_tool_boundary` | Applied centrally in `get_tools()` |
| [`scheduling.py`](scheduling.py) | `create_alert`, `update_alert`, `update_task`, `list_alerts` | SQLite via `store/` |
| [`scratchpad.py`](scratchpad.py) | `edit_scratchpad` | `.data/scratchpad/SCRATCHPAD.md` |
| [`progress.py`](progress.py) | `edit_progress` | `.data/scratchpad/PROGRESS.md` |
| [`anchored_edit.py`](anchored_edit.py) | — | Shared factory for anchored edits |
| [`websearch/`](websearch/) | `websearch` | Swappable backend; v1: [`providers/langsearch.py`](websearch/providers/langsearch.py) |
| [`webfetch/`](webfetch/) | `webfetch` | Playwright + trafilatura; SSRF guard in `url_safety.py` |

## Registration pattern

Each module exports:

```python
def create_tools(settings: Settings) -> list[BaseTool]:
    ...
```

Register new modules with `register_tools(create_tools)` — no manual boundary wrapper needed.

## Scheduling tools detail

| Tool | AGENTS.md section |
|---|---|
| `create_alert` | [Triage rules](#agent-scheduling-tools-target) — reschedule via new one-shot |
| `update_alert` | Agent must know `alert_id` from `list_alerts` |
| `update_task` | `completed` / `ignored` + `disposition_reason` |
| `list_alerts` | [Schedule map](#list_alerts--schedule-map) — `schedule_map.py` |

## Scratchpad & progress tools

| Tool | File | Notes |
|---|---|---|
| `edit_scratchpad` | `SCRATCHPAD.md` | Front-of-mind notes; always-in-context (planned) |
| `edit_progress` | `PROGRESS.md` | Progress journal; always-in-context (planned) |

Both use anchored deltas via `utils/anchored_edit.py` (`pre`, `post`, `original`, `new`; `<BOF>` / `<EOF>`).

## World observation tools

| Tool | When to use | Depends on |
|---|---|---|
| `websearch` | Discovery, recent facts, citations | `.env` → `LANGSEARCH_API_KEY` |
| `webfetch` | Full page read after search or user URL | Playwright Chromium, `config.yaml` → `webfetch` |

## Not yet registered

| Capability | Module | Gap |
|---|---|---|
| Vector memory | `memory/` | No `tools/memory.py` factory |

## Tests

| Test file | Covers |
|---|---|
| `tests/test_tools.py` | Registry tool names |
| `tests/test_tool_boundary.py` | Tool error JSON boundary |
| `tests/test_scheduling_tools.py` | Scheduling tools |
| `tests/test_websearch.py` | websearch tool + LangSearch provider |
| `tests/test_webfetch.py`, `tests/test_url_safety.py` | PageFetcher + webfetch + URL safety |
| `tests/test_anchored_edit.py` | Anchored edit utils + edit tools |
| `tests/integration/test_websearch_live.py` | Live websearch (LangSearch provider) |

## Gaps / redundancies

- **`scheduling.py` opens its own DB connection** per tool call — separate from runtime `conn` (see `db/INDEX.md`).
- **websearch optional** — omitted when `LANGSEARCH_API_KEY` is unset; add providers under `websearch/providers/`.
