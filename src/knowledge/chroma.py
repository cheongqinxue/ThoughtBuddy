"""Chroma-backed knowledge backend (v1 default)."""

from __future__ import annotations

from typing import Any

from config import Settings
from knowledge.backend import KnowledgeBackend, KnowledgeEntry, KnowledgeSearchHit
from memory import get_memory_store, write_memory
from memory.store import MemoryStore


class ChromaKnowledgeBackend:
    """Knowledge backend implemented with ChromaDB vector storage."""

    def __init__(self, store: MemoryStore) -> None:
        """Initialize the backend.

        Args:
            store: Underlying memory store.
        """
        self._store = store

    def add(self, text: str, *, metadata: dict[str, Any] | None = None) -> KnowledgeEntry:
        """Store a knowledge entry.

        Args:
            text: Knowledge body text.
            metadata: Optional metadata.

        Returns:
            Stored knowledge entry.
        """
        meta = dict(metadata or {})
        meta.setdefault("superseded_by", None)
        record = self._store.write(text, metadata=meta)
        return KnowledgeEntry(id=record.memory_id, text=record.text, metadata=record.metadata)

    def edit(
        self,
        entry_id: str,
        *,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
        supersede: bool = False,
    ) -> KnowledgeEntry:
        """Update or supersede a knowledge entry.

        Args:
            entry_id: Existing entry id.
            text: Replacement text when superseding or rewriting.
            metadata: Metadata updates or replacement metadata.
            supersede: When true, mark the old entry superseded and write a new one.

        Returns:
            Resulting active knowledge entry.
        """
        if supersede:
            new_meta = dict(metadata or {})
            new_meta["supersedes_id"] = entry_id
            new_meta.setdefault("superseded_by", None)
            body = text or ""
            if not body:
                msg = "Supersede requires replacement text."
                raise ValueError(msg)
            new_entry = self.add(body, metadata=new_meta)
            self._mark_superseded(entry_id, new_entry.id)
            return new_entry

        merged_meta = dict(metadata or {})
        merged_meta.setdefault("superseded_by", None)
        body = text or ""
        if not body:
            msg = "Edit requires text when not superseding."
            raise ValueError(msg)
        record = self._store.write(body, metadata=merged_meta, memory_id=entry_id)
        return KnowledgeEntry(id=record.memory_id, text=record.text, metadata=record.metadata)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_superseded: bool = False,
    ) -> list[KnowledgeSearchHit]:
        """Search knowledge entries.

        Args:
            query: Natural-language query.
            top_k: Maximum results.
            include_superseded: Include entries marked superseded.

        Returns:
            Ranked knowledge hits.
        """
        results = self._store.search(query, top_k=top_k)
        hits: list[KnowledgeSearchHit] = []
        for result in results:
            metadata = result.metadata
            if not include_superseded and metadata.get("superseded_by"):
                continue
            hits.append(
                KnowledgeSearchHit(
                    id=result.memory_id,
                    text=result.text,
                    metadata=metadata,
                    score=result.vector_score,
                ),
            )
        return hits

    def _mark_superseded(self, old_id: str, new_id: str) -> None:
        existing = self._store.get(old_id)
        if existing is None:
            return
        metadata = dict(existing.metadata)
        metadata["superseded_by"] = new_id
        self._store.write(existing.text, metadata=metadata, memory_id=old_id)


def build_chroma_backend(settings: Settings) -> KnowledgeBackend:
    """Build the default Chroma knowledge backend.

    Args:
        settings: Application settings.

    Returns:
        Configured knowledge backend.
    """
    return ChromaKnowledgeBackend(get_memory_store(settings))
