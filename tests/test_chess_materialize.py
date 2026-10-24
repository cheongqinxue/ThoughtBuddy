"""Tests for chess experiment task materialization."""

from __future__ import annotations

from config import Settings
from db.connection import connect
from experiments.chess.plugin.materialize import materialize_chess_tasks
from store.chess import ChessStore
from store.tasks import TaskStore


class TestChessMaterialize:
    """Chess wake task materialization."""

    def test_enqueues_start_when_no_active_session(self, tmp_path, settings: Settings) -> None:
        """Queue an initial chess task when no agent game exists."""
        conn = connect(tmp_path / "chess.db")
        active = settings.model_copy(update={"active_experiment": "chess"})

        created = materialize_chess_tasks(conn, active)
        conn.commit()

        assert created == 1
        pending = TaskStore(conn).list_pending()
        assert pending[0].payload["kind"] == "chess_start"

    def test_enqueues_continue_on_agent_turn(self, tmp_path, settings: Settings) -> None:
        """Queue a continuation task when the agent is on move."""
        conn = connect(tmp_path / "chess.db")
        active = settings.model_copy(update={"active_experiment": "chess"})
        ChessStore(conn).create_session(
            opponent_type="agent",
            player_color="white",
            agent_generation=0,
        )
        conn.commit()

        created = materialize_chess_tasks(conn, active)
        conn.commit()

        assert created == 1
        pending = TaskStore(conn).list_pending()
        assert pending[0].payload["kind"] == "chess_continue"

    def test_skips_duplicate_continue_tasks(self, tmp_path, settings: Settings) -> None:
        """Do not flood the queue with duplicate chess continuation tasks."""
        conn = connect(tmp_path / "chess.db")
        active = settings.model_copy(update={"active_experiment": "chess"})
        session = ChessStore(conn).create_session(
            opponent_type="agent",
            player_color="white",
            agent_generation=0,
        )
        TaskStore(conn).enqueue_user_task(
            message="Continue the active chess game vs Stockfish. Check status and make your move.",
            payload={"kind": "chess_continue", "session_id": session.id},
        )
        conn.commit()

        created = materialize_chess_tasks(conn, active)
        conn.commit()

        assert created == 0
        assert len(TaskStore(conn).list_pending()) == 1
