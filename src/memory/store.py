"""ChromaDB-backed persistent memory store."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from config import Settings
from memory.embeddings import Embedder, build_embedder
from memory.rerank import MemoryReranker, VectorScoreReranker
from memory.types import MemoryCandidate, MemoryRecord, MemorySearchResult

logger = logging.getLogger(__name__)

_METADATA_JSON_PREFIX = "__json__:"


class MemoryStore:
    """Disk-persistent vector memory backed by ChromaDB."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        chroma_client: ClientAPI | None = None,
    ) -> None:
        """Initialize the memory store.

        Args:
            settings: Application settings.
            embedder: Optional embedder override (used in tests).
            chroma_client: Optional Chroma client override (used in tests).
        """
        self._settings = settings
        self._embedder = embedder or build_embedder(
            settings.memory_embedding_model,
            device=settings.memory_embedding_device or None,
        )
        self._client = chroma_client or self._build_persistent_client(settings.memory_store_path)
        self._collection = self._get_or_create_collection(
            settings.memory_collection_name,
        )
        self._default_reranker = VectorScoreReranker()

    def write(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> MemoryRecord:
        """Write a memory to the vector store.

        Args:
            text: Memory body text to embed and persist.
            metadata: Optional scalar metadata stored alongside the memory.
            memory_id: Optional stable identifier. Generated when omitted.

        Returns:
            Persisted memory record.
        """
        normalized_text = text.strip()
        if not normalized_text:
            msg = "Memory text must not be empty."
            raise ValueError(msg)

        resolved_id = memory_id or str(uuid.uuid4())
        stored_metadata = _sanitize_metadata(metadata or {})
        embedding = self._embedder.embed_documents([normalized_text])[0]

        upsert_kwargs: dict[str, Any] = {
            "ids": [resolved_id],
            "documents": [normalized_text],
            "embeddings": [embedding],
        }
        if stored_metadata:
            upsert_kwargs["metadatas"] = [stored_metadata]
        self._collection.upsert(**upsert_kwargs)
        logger.debug("Stored memory %s", resolved_id)
        return MemoryRecord(
            memory_id=resolved_id,
            text=normalized_text,
            metadata=_deserialize_metadata(stored_metadata),
        )

    def recall(self, query: str, *, k: int = 10) -> list[MemoryCandidate]:
        """Recall candidate memories using vector similarity only.

        Args:
            query: Natural-language search query.
            k: Maximum number of candidates to return.

        Returns:
            Recalled memory candidates ordered by vector score.
        """
        normalized_query = query.strip()
        if not normalized_query:
            msg = "Search query must not be empty."
            raise ValueError(msg)

        if k <= 0:
            return []

        if self.count() == 0:
            return []

        query_embedding = self._embedder.embed_query(normalized_query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        candidates: list[MemoryCandidate] = []
        for memory_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            candidates.append(
                MemoryCandidate(
                    memory_id=memory_id,
                    text=document or "",
                    metadata=_deserialize_metadata(metadata or {}),
                    vector_score=_distance_to_similarity(distance),
                ),
            )

        return sorted(candidates, key=lambda candidate: candidate.vector_score, reverse=True)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        recall_k: int | None = None,
        reranker: MemoryReranker | None = None,
    ) -> list[MemorySearchResult]:
        """Search memories using vector recall and optional reranking.

        Args:
            query: Natural-language search query.
            top_k: Number of final results to return.
            recall_k: Vector recall pool size. Defaults to ``top_k * multiplier``.
            reranker: Optional reranker (e.g. future ColBERT). Defaults to vector order.

        Returns:
            Ranked memory search results.
        """
        if top_k <= 0:
            return []

        resolved_recall_k = recall_k or top_k * self._settings.memory_recall_multiplier
        resolved_recall_k = max(resolved_recall_k, top_k)
        candidates = self.recall(query, k=resolved_recall_k)
        active_reranker = reranker or self._default_reranker
        return active_reranker.rerank(query, candidates, top_k=top_k)

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Fetch a memory by identifier.

        Args:
            memory_id: Memory identifier.

        Returns:
            Memory record when found, otherwise ``None``.
        """
        results = self._collection.get(
            ids=[memory_id],
            include=["documents", "metadatas"],
        )
        ids = results.get("ids") or []
        if not ids:
            return None

        document = (results.get("documents") or [""])[0]
        metadata = (results.get("metadatas") or [{}])[0]
        return MemoryRecord(
            memory_id=memory_id,
            text=document or "",
            metadata=_deserialize_metadata(metadata or {}),
        )

    def count(self) -> int:
        """Return the number of stored memories."""
        return self._collection.count()

    @staticmethod
    def _build_persistent_client(path: str) -> ClientAPI:
        store_path = Path(path)
        store_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(store_path))

    def _get_or_create_collection(self, name: str) -> Collection:
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert metadata values into Chroma-compatible scalar fields.

    Complex values are JSON-encoded into strings with a reserved prefix.

    Args:
        metadata: Raw metadata mapping.

    Returns:
        Metadata mapping safe for Chroma persistence.
    """
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
            continue
        sanitized[key] = f"{_METADATA_JSON_PREFIX}{json.dumps(value, sort_keys=True)}"
    return sanitized


def _deserialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Restore metadata values previously encoded for Chroma storage.

    Args:
        metadata: Stored metadata mapping.

    Returns:
        Deserialized metadata mapping.
    """
    restored: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, str) and value.startswith(_METADATA_JSON_PREFIX):
            restored[key] = json.loads(value.removeprefix(_METADATA_JSON_PREFIX))
            continue
        restored[key] = value
    return restored


def _distance_to_similarity(distance: float | None) -> float:
    """Convert Chroma cosine distance to a higher-is-better similarity score.

    Args:
        distance: Chroma distance for cosine space.

    Returns:
        Similarity score in ``[0, 1]`` when distance is valid.
    """
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - distance)
