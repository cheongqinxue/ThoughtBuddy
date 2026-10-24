"""Tests for persistent vector memory utilities."""

from __future__ import annotations

import uuid

import chromadb
import pytest

import memory as memory_module
from config import Settings
from memory import clear_memory_store_cache, search_memories, write_memory
from memory.embeddings import DeterministicEmbedder
from memory.rerank import ColbertReranker, VectorScoreReranker
from memory.store import MemoryStore
from memory.types import MemoryCandidate


@pytest.fixture
def memory_settings(tmp_path) -> Settings:
    """Settings pointing at a temporary Chroma store."""
    return Settings(
        openai_base_url="http://localhost:8000/v1",
        openai_api_key="test-key",
        agent_model="test-model",
        langsearch_api_key="test-langsearch-key",
        memory_store_path=str(tmp_path / "chroma"),
        memory_collection_name=f"test_memories_{uuid.uuid4().hex}",
        memory_recall_multiplier=4,
    )


@pytest.fixture
def memory_store(memory_settings: Settings) -> MemoryStore:
    """Return an isolated in-memory Chroma store for tests."""
    clear_memory_store_cache()
    client = chromadb.EphemeralClient()
    store = MemoryStore(
        memory_settings,
        embedder=DeterministicEmbedder(),
        chroma_client=client,
    )
    yield store
    clear_memory_store_cache()


class TestMemoryStore:
    """Memory store read/write tests."""

    def test_write_and_get_memory(self, memory_store: MemoryStore) -> None:
        """Persist and fetch a memory by id."""
        record = memory_store.write(
            "User prefers concise morning summaries.",
            metadata={"topic": "preferences"},
            memory_id="mem-1",
        )

        fetched = memory_store.get("mem-1")

        assert record.memory_id == "mem-1"
        assert fetched is not None
        assert fetched.text == record.text
        assert fetched.metadata == {"topic": "preferences"}

    def test_write_rejects_empty_text(self, memory_store: MemoryStore) -> None:
        """Reject empty memory bodies."""
        with pytest.raises(ValueError, match="empty"):
            memory_store.write("   ")

    def test_search_returns_best_match(self, memory_store: MemoryStore) -> None:
        """Return the most relevant stored memory for a query."""
        memory_store.write(
            "User prefers concise morning summaries every day.",
            memory_id="pref",
        )
        memory_store.write(
            "Database backup runs nightly at 2am UTC on Sundays.",
            memory_id="ops",
        )

        results = memory_store.search(
            "morning summaries preference",
            top_k=1,
        )

        assert len(results) == 1
        assert results[0].memory_id == "pref"

    def test_recall_uses_larger_candidate_pool(self, memory_store: MemoryStore) -> None:
        """Recall more candidates than the final top_k before reranking."""
        for index in range(6):
            memory_store.write(f"Memory number {index} about topic alpha", memory_id=f"m{index}")

        candidates = memory_store.recall("topic alpha", k=5)
        reranker = VectorScoreReranker()
        final = reranker.rerank(
            "topic alpha",
            candidates,
            top_k=2,
        )

        assert len(candidates) == 5
        assert len(final) == 2

    def test_metadata_roundtrip_for_complex_values(self, memory_store: MemoryStore) -> None:
        """Store and restore non-scalar metadata values."""
        memory_store.write(
            "Structured metadata example.",
            metadata={"tags": ["alpha", "beta"], "priority": 2},
            memory_id="meta-1",
        )

        fetched = memory_store.get("meta-1")

        assert fetched is not None
        assert fetched.metadata == {"tags": ["alpha", "beta"], "priority": 2}


class TestMemoryUtilities:
    """High-level memory helper tests."""

    def test_write_and_search_helpers(self, memory_settings: Settings) -> None:
        """Use module helpers against an isolated store."""
        clear_memory_store_cache()
        store = MemoryStore(
            memory_settings,
            embedder=DeterministicEmbedder(),
            chroma_client=chromadb.EphemeralClient(),
        )
        memory_module._store_cache[memory_settings.memory_store_path] = store

        write_memory(
            "Project codename is ThoughtBuddy.",
            memory_id="project-name",
            settings=memory_settings,
        )
        results = search_memories("ThoughtBuddy codename", top_k=1, settings=memory_settings)

        assert results[0].memory_id == "project-name"
        clear_memory_store_cache()


class TestRerankers:
    """Reranker protocol tests."""

    def test_colbert_reranker_is_not_implemented_yet(self) -> None:
        """ColBERT reranker remains a future extension point."""
        reranker = ColbertReranker("colbert-qa")
        candidates = [
            MemoryCandidate(
                memory_id="1",
                text="Example",
                metadata={},
                vector_score=0.9,
            ),
        ]

        with pytest.raises(NotImplementedError, match="ColBERT"):
            reranker.rerank("example", candidates, top_k=1)
