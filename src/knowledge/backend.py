"""Knowledge backend protocol — storage/retrieval abstracted from agent tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class KnowledgeEntry:
    """One knowledge record."""

    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class KnowledgeSearchHit:
    """One knowledge search result."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float | None = None


@runtime_checkable
class KnowledgeBackend(Protocol):
    """Protocol for swappable knowledge storage and retrieval."""

    def add(self, text: str, *, metadata: dict[str, Any] | None = None) -> KnowledgeEntry:
        """Store a knowledge entry."""

    def edit(
        self,
        entry_id: str,
        *,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
        supersede: bool = False,
    ) -> KnowledgeEntry:
        """Update or supersede a knowledge entry."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_superseded: bool = False,
    ) -> list[KnowledgeSearchHit]:
        """Search knowledge entries relevant to a query."""
