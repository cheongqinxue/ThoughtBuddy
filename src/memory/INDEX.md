# `src/memory/` — vector memory (ChromaDB)

Disk-persistent semantic memory for facts the agent may need later. **Utility layer only** — not yet an agent tool or wake-cycle step.

## AGENTS.md

- [Long-term memory (vector store)](#long-term-memory-vector-store)
- Config: `config.yaml` → `memory` section
- Distinct from SQLite `message_log` (conversation archive)

## Files

| File | Role |
|---|---|
| [`__init__.py`](__init__.py) | Public API: `write_memory`, `search_memories`, `recall_memories`, `get_memory_store` |
| [`store.py`](store.py) | `MemoryStore` — ChromaDB persistent client, upsert/query |
| [`embeddings.py`](embeddings.py) | `SentenceTransformerEmbedder` (GPU), `DeterministicEmbedder` (tests) |
| [`rerank.py`](rerank.py) | `MemoryReranker` protocol, `VectorScoreReranker`, `ColbertReranker` stub |
| [`types.py`](types.py) | `MemoryRecord`, `MemoryCandidate`, `MemorySearchResult` |

## Retrieval pipeline

```mermaid
flowchart LR
Q[query] --> RECALL[vector recall k × multiplier]
RECALL --> RERANK[optional MemoryReranker]
RERANK --> OUT[top_k results]
```

Default reranker preserves vector order. `ColbertReranker` raises `NotImplementedError` — hook for future last-mile reranking.

## Storage layout

- Path: `config.yaml` → `memory.store_path` (default `data/memories/chroma`)
- Collection: `memory.collection_name`
- Embeddings: local `sentence-transformers` (`memory.embedding_model`)

## Callers

| Consumer | Status |
|---|---|
| Agent tools | **Not wired** |
| Wake cycle | **Not wired** |
| Tests | `tests/test_memory.py` |

## Tests

| Test file | Covers |
|---|---|
| `tests/test_memory.py` | Write, search, reranker stub, helpers |

## Gaps / AGENTS.md alignment

- **Not in wake cycle step 2 (Review progress)** — AGENTS.md mentions progress journal / goals; memory store is separate experimental layer.
- **No agent tool** — `write_memory` / `search_memories` ready for `tools/memory.py` factory.
- **Two memory systems:** conversation archive (SQLite) vs semantic memory (Chroma) — document clearly for future contributors.
