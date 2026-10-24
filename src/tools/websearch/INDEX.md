# `tools/websearch/` — websearch agent tool

Agent-facing tool name: **`websearch`**. Provider implementations live under `providers/`.

## Layout

| File | Role |
|---|---|
| [`__init__.py`](__init__.py) | `create_tools()` — registers `websearch` |
| [`backend.py`](backend.py) | `WebSearchBackend` protocol + `create_websearch_backend()` |
| [`format.py`](format.py) | Normalize provider payloads → LLM citations |
| [`schemas.py`](schemas.py) | `WebSearchInput` args schema |
| [`providers/langsearch.py`](providers/langsearch.py) | LangSearch HTTP client (v1 default) |

## Adding a provider

1. Implement `WebSearchBackend` in `providers/<name>.py` (return SearchResponse-shaped dict).
2. Branch in `create_websearch_backend()` on `settings.websearch_provider`.
3. Add credentials to `.env-template` and wire `Settings` if needed.
