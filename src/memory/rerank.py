"""Reranking hooks for memory retrieval."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from memory.types import MemoryCandidate, MemorySearchResult


@runtime_checkable
class MemoryReranker(Protocol):
    """Optional second-stage reranker (e.g. ColBERT) over vector recall."""

    def rerank(
        self,
        query: str,
        candidates: list[MemoryCandidate],
        *,
        top_k: int,
    ) -> list[MemorySearchResult]:
        """Rerank recalled candidates and return the best matches.

        Args:
            query: Original search query.
            candidates: Vector-recalled candidates with similarity scores.
            top_k: Number of results to return.

        Returns:
            Ranked memory search results.
        """


class VectorScoreReranker:
    """Default reranker that preserves vector similarity order."""

    def rerank(
        self,
        query: str,
        candidates: list[MemoryCandidate],
        *,
        top_k: int,
    ) -> list[MemorySearchResult]:
        """Return the top vector matches without additional scoring.

        Args:
            query: Original search query (unused).
            candidates: Vector-recalled candidates.
            top_k: Number of results to return.

        Returns:
            Top candidates as search results ordered by vector score.
        """
        del query
        ordered = sorted(candidates, key=lambda candidate: candidate.vector_score, reverse=True)
        return [
            MemorySearchResult(
                memory_id=candidate.memory_id,
                text=candidate.text,
                metadata=candidate.metadata,
                vector_score=candidate.vector_score,
                rerank_score=None,
            )
            for candidate in ordered[:top_k]
        ]


class ColbertReranker:
    """Placeholder for future ColBERT last-mile reranking.

    Implements the ``MemoryReranker`` protocol but is not wired yet. Swap this
    in via ``search_memories(..., reranker=ColbertReranker(...))`` once a local
    ColBERT model is configured.
    """

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        """Store ColBERT configuration for future use.

        Args:
            model_name: ColBERT checkpoint identifier.
            device: Optional torch device override.
        """
        self._model_name = model_name
        self._device = device

    def rerank(
        self,
        query: str,
        candidates: list[MemoryCandidate],
        *,
        top_k: int,
    ) -> list[MemorySearchResult]:
        """Raise until ColBERT integration is implemented."""
        del query, candidates, top_k
        msg = (
            "ColBERT reranking is not implemented yet. "
            f"Configured model: {self._model_name} (device={self._device})."
        )
        raise NotImplementedError(msg)
