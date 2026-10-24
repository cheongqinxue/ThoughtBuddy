"""Persistent vector memory utilities for ThoughtBuddy."""

from __future__ import annotations

from typing import Any

from config import Settings
from memory.rerank import MemoryReranker
from memory.store import MemoryStore
from memory.types import MemoryRecord, MemorySearchResult

_store_cache: dict[str, MemoryStore] = {}


def get_memory_store(settings: Settings | None = None) -> MemoryStore:
    """Return a memory store for the configured Chroma path.

    Args:
        settings: Optional settings override. Loads from environment when omitted.

    Returns:
        Initialized memory store instance.
    """
    resolved = settings or Settings.load()
    cache_key = resolved.memory_store_path
    cached = _store_cache.get(cache_key)
    if cached is not None:
        return cached

    store = MemoryStore(resolved)
    _store_cache[cache_key] = store
    return store


def write_memory(
    text: str,
    *,
    metadata: dict[str, Any] | None = None,
    memory_id: str | None = None,
    settings: Settings | None = None,
) -> MemoryRecord:
    """Persist a memory for later semantic retrieval.

    Args:
        text: Memory body text.
        metadata: Optional metadata stored with the memory.
        memory_id: Optional stable identifier.
        settings: Optional settings override.

    Returns:
        Stored memory record.
    """
    return get_memory_store(settings).write(
        text,
        metadata=metadata,
        memory_id=memory_id,
    )


def search_memories(
    query: str,
    *,
    top_k: int = 5,
    recall_k: int | None = None,
    reranker: MemoryReranker | None = None,
    settings: Settings | None = None,
) -> list[MemorySearchResult]:
    """Search stored memories by semantic similarity.

    Vector recall returns a larger candidate pool; an optional reranker can
    refine ordering (for example ColBERT in a later phase).

    Args:
        query: Natural-language search query.
        top_k: Number of final results to return.
        recall_k: Optional vector recall pool size.
        reranker: Optional reranker implementation.
        settings: Optional settings override.

    Returns:
        Ranked memory search results.
    """
    return get_memory_store(settings).search(
        query,
        top_k=top_k,
        recall_k=recall_k,
        reranker=reranker,
    )


def recall_memories(
    query: str,
    *,
    k: int = 10,
    settings: Settings | None = None,
) -> list[MemorySearchResult]:
    """Recall memories using vector search only.

    Args:
        query: Natural-language search query.
        k: Number of candidates to return.
        settings: Optional settings override.

    Returns:
        Vector-ranked memory search results without reranking metadata.
    """
    candidates = get_memory_store(settings).recall(query, k=k)
    return [
        MemorySearchResult(
            memory_id=candidate.memory_id,
            text=candidate.text,
            metadata=candidate.metadata,
            vector_score=candidate.vector_score,
        )
        for candidate in candidates
    ]


def clear_memory_store_cache() -> None:
    """Clear cached memory store instances (primarily for tests)."""
    _store_cache.clear()
