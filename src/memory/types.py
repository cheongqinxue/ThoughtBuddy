"""Shared types for persistent agent memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryRecord:
    """A memory written to the vector store."""

    memory_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryCandidate:
    """A memory recalled by vector search before optional reranking."""

    memory_id: str
    text: str
    metadata: dict[str, Any]
    vector_score: float


@dataclass(frozen=True)
class MemorySearchResult:
    """A memory returned after recall and optional reranking."""

    memory_id: str
    text: str
    metadata: dict[str, Any]
    vector_score: float
    rerank_score: float | None = None
