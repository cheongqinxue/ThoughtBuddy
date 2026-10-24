"""Tests for knowledge backend and tools."""

from __future__ import annotations

import json

import pytest

from config import Settings
from knowledge.chroma import ChromaKnowledgeBackend
from memory.embeddings import DeterministicEmbedder
from memory.store import MemoryStore


@pytest.fixture
def knowledge_backend(tmp_path, settings: Settings) -> ChromaKnowledgeBackend:
    """Deterministic knowledge backend for tests."""
    settings = settings.model_copy(update={"memory_store_path": str(tmp_path / "chroma")})
    store = MemoryStore(settings, embedder=DeterministicEmbedder(embedding_dim=8))
    return ChromaKnowledgeBackend(store)


class TestKnowledgeBackend:
    """KnowledgeBackend behavior."""

    def test_add_and_search(self, knowledge_backend: ChromaKnowledgeBackend) -> None:
        """Added knowledge is searchable."""
        knowledge_backend.add("Bishops move diagonally.", metadata={"tags": ["chess"]})
        hits = knowledge_backend.search("bishop movement", top_k=3)
        assert hits
        assert "diagonally" in hits[0].text

    def test_supersede_excludes_old_entry(self, knowledge_backend: ChromaKnowledgeBackend) -> None:
        """Superseded entries are excluded from default search."""
        original = knowledge_backend.add("Old rule about castling.")
        knowledge_backend.edit(
            original.id,
            text="Updated castling rule.",
            supersede=True,
        )
        hits = knowledge_backend.search("castling", top_k=5)
        texts = " ".join(hit.text for hit in hits)
        assert "Updated castling rule" in texts
        assert "Old rule" not in texts


class TestKnowledgeTools:
    """Knowledge agent tools."""

    def test_knowledge_add_and_search_tools(
        self,
        tmp_path,
        settings: Settings,
        monkeypatch,
    ) -> None:
        """Tools delegate to backend without leaking provider names."""
        from knowledge import tools as knowledge_tools_module

        settings = settings.model_copy(update={"memory_store_path": str(tmp_path / "chroma")})

        def _build_backend(_settings: Settings):
            store = MemoryStore(_settings, embedder=DeterministicEmbedder(embedding_dim=8))
            return ChromaKnowledgeBackend(store)

        monkeypatch.setattr(knowledge_tools_module, "build_chroma_backend", _build_backend)
        tools = {tool.name: tool for tool in knowledge_tools_module.create_tools(settings)}
        add_payload = json.loads(
            tools["knowledge_add"].invoke({"text": "Knights move in an L-shape.", "tags": ["chess"]}),
        )
        assert add_payload["ok"] is True
        search_payload = json.loads(tools["knowledge_search"].invoke({"query": "knight", "top_k": 3}))
        assert search_payload
