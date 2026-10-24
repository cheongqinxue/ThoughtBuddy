# `src/context/` — message archive → model context

Builds what the LLM sees each wake from SQLite `message_log`, scratchpad files, and `compaction_points`. LangGraph checkpoint messages are replaced per wake; SQLite is the source of truth for history.

## AGENTS.md

- [Message archive & context compaction](#message-archive--context-compaction-target)
- [Persistent scratchpad](#persistent-scratchpad)
- Config: `config.yaml` → `context` (`retain_messages`, `compact_token_threshold`)

## Files

| File | Exports / role |
|---|---|
| [`build.py`](build.py) | `build_agent_context()`, `build_transcript_messages()` |
| [`system.py`](system.py) | `build_system_prompt()` — operating environment + tools |
| [`compact.py`](compact.py) | `maybe_compact_context()`, `load_compaction_prompt()` — reads `_compact.md` |
| [`serialize.py`](serialize.py) | LangChain message ↔ JSON for `message_log.content_json` |
| [`tokens.py`](tokens.py) | Token estimate for compaction threshold |

## Dependencies

| Module | Why |
|---|---|
| `store/messages.py` | Read/write `message_log` |
| `store/compaction.py` | Read/write `compaction_points` |
| `utils/anchored_edit.py` | Read `_goals.md`, `PROGRESS.md`, `SCRATCHPAD.md`, `_compact.md` |
| `config.Settings` | Retain count, compact threshold, scratchpad directory |
| `tools.get_tools()` | Tool names/descriptions for system prompt |

## Callers

| Caller | When |
|---|---|
| `runtime/wake.py` → `invoke_wake_cycle()` | Append wake msg → maybe compact → `build_agent_context()` → invoke → archive new turn |

## Context assembly order

1. **System prompt** — harness, pillars, tools
2. **Goals** — `_goals.md`
3. **Owner instructions** — `_system.md`
4. **Plan summary** — DAG snapshot
5. **World snapshot** — observations + virtual state
6. **Knowledge recall** — top-K from backend
7. **Progress** — `PROGRESS.md`
8. **Scratchpad** — `SCRATCHPAD.md`
9. **Memory checkpoint** — latest compaction summary (if any)
10. **Verbatim transcript** — messages after compaction boundary
11. **Open tasks** — pending task snapshot JSON

Compaction summarizes through boundary `t − N`; the verbatim tail may grow beyond `N` until the next compaction.

## Tests

| Test file | Covers |
|---|---|
| `tests/test_context.py` | Archive, compaction, full `build_agent_context` |

## Gaps / redundancies

- **FTS / search** on `message_log` — `history_search` (FTS5 BM25) + `history_read` (datetime pagination)
- **Compaction summarization** uses the same chat `model` as the agent — no dedicated small summarizer model in config.
- Token counts use `llm.tokens.count_messages_tokens()` (HuggingFace `apply_chat_template` for `llm.agent_model`); `context/tokens.py` delegates there.
- Pre-wake budget check in `budget.py` compares full assembled context against `llm.context_window_tokens - llm.max_tokens`; overflow raises `CriticalWakeError` and halts the runtime (no heartbeat retry loop).
