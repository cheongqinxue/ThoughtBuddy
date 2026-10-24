"""Tests for world observation store."""

from __future__ import annotations

from db.connection import connect
from store.world import WorldStore


class TestWorldStore:
    """World observations and virtual state."""

    def test_append_and_list_observations(self, tmp_path) -> None:
        """Observations persist and list newest-first."""
        conn = connect(tmp_path / "test.db")
        store = WorldStore(conn)
        store.append_observation(
            "t1",
            source="websearch",
            payload={"query": "chess openings"},
            summary="search results",
        )
        conn.commit()
        observations = store.list_recent("t1", limit=5)
        assert len(observations) == 1
        assert observations[0].source == "websearch"
        conn.close()

    def test_virtual_state_roundtrip(self, tmp_path) -> None:
        """Virtual state keys persist."""
        conn = connect(tmp_path / "test.db")
        store = WorldStore(conn)
        store.set_state("t1", "chess.fen", "startpos-fen")
        conn.commit()
        assert store.get_state("t1", "chess.fen") == "startpos-fen"
        assert store.list_state("t1") == {"chess.fen": "startpos-fen"}
        conn.close()
