"""Tests for chess session storage and game service."""

from __future__ import annotations

import json
from unittest.mock import patch

import chess
import pytest

from config import Settings
from db.connection import connect
from experiments.chess.plugin.service import ChessGameError, ChessGameService
from store.chess import ChessStore


@pytest.fixture
def chess_conn(tmp_path):
    """SQLite connection with chess schema."""
    conn = connect(tmp_path / "chess.db")
    yield conn
    conn.close()


@pytest.fixture
def chess_settings(tmp_path, settings: Settings) -> Settings:
    """Settings for chess service tests."""
    return settings.model_copy(
        update={
            "active_experiment": "chess",
            "thoughtbuddy_db_path": str(tmp_path / "chess.db"),
            "chess_engine_path": str(tmp_path / "missing-engine"),
            "chess_agent_color": "white",
            "chess_agent_generation": 0,
        },
    )


class TestChessStore:
    """Chess session persistence."""

    def test_create_and_list_sessions(self, chess_conn) -> None:
        """Create human and agent sessions with distinct labels."""
        store = ChessStore(chess_conn)
        human = store.create_session(opponent_type="human", player_color="white")
        agent = store.create_session(
            opponent_type="agent",
            player_color="white",
            agent_generation=0,
        )

        sessions = store.list_sessions(limit=10)
        assert {session.id for session in sessions} == {human.id, agent.id}
        assert human.stockfish_color == "black"
        assert agent.agent_generation == 0

    def test_append_moves_in_order(self, chess_conn) -> None:
        """Moves are stored in ply order."""
        store = ChessStore(chess_conn)
        session = store.create_session(opponent_type="human", player_color="white")
        store.append_move(
            session_id=session.id,
            uci="e2e4",
            san="e4",
            player_role="human",
            fen_after=session.current_fen,
        )
        store.append_move(
            session_id=session.id,
            uci="e7e5",
            san="e5",
            player_role="stockfish",
            fen_after=session.current_fen,
        )
        moves = store.list_moves(session.id)
        assert [move.ply for move in moves] == [1, 2]
        assert moves[0].player_role == "human"
        assert moves[1].player_role == "stockfish"


class TestChessGameService:
    """Session-backed chess orchestration."""

    def test_create_human_session(self, chess_conn, chess_settings: Settings) -> None:
        """Human sessions are labeled for the UI."""
        service = ChessGameService(chess_conn, chess_settings)
        session = service.create_human_session(player_color="white")
        assert session["opponent_type"] == "human"
        assert session["label"] == "Human vs Stockfish"

    def test_agent_session_tracks_generation(self, chess_conn, chess_settings: Settings) -> None:
        """Agent sessions store the configured generation."""
        service = ChessGameService(chess_conn, chess_settings)
        session = service.create_agent_session(force_new=True)
        assert session["opponent_type"] == "agent"
        assert session["agent_generation"] == 0
        assert session["label"] == "Agent Gen 0 vs Stockfish"

    def test_apply_human_move_records_both_sides(
        self,
        chess_conn,
        chess_settings: Settings,
    ) -> None:
        """Human move triggers a mocked Stockfish reply."""
        service = ChessGameService(chess_conn, chess_settings)
        session = service.create_human_session(player_color="white")
        mock_move = chess.Move.from_uci("e7e5")

        def _opponent_reply(session, _settings, *, player_color: str):
            return session.move("e7e5")

        with patch(
            "experiments.chess.plugin.service.play_opponent_move",
            side_effect=_opponent_reply,
        ):
            detail = service.apply_human_move(session["id"], "e2e4")

        roles = [move["player_role"] for move in detail["moves"]]
        assert roles == ["human", "stockfish"]

    def test_apply_agent_move_rejects_wrong_session_type(
        self,
        chess_conn,
        chess_settings: Settings,
    ) -> None:
        """Agent moves cannot be applied to human sessions."""
        service = ChessGameService(chess_conn, chess_settings)
        session = service.create_human_session()
        with pytest.raises(ChessGameError, match="not a agent game"):
            service.apply_agent_move(session["id"], "e2e4")

    def test_reset_agent_session_starts_new_game(
        self,
        chess_conn,
        chess_settings: Settings,
    ) -> None:
        """Reset abandons the prior agent session."""
        service = ChessGameService(chess_conn, chess_settings)
        first = service.create_agent_session(force_new=True)
        second = service.reset_agent_session()
        assert first["id"] != second["id"]
        abandoned = ChessStore(chess_conn).get_session(first["id"])
        assert abandoned.status == "abandoned"


class TestChessToolsWithSessions:
    """Agent tools backed by sessions."""

    def test_world_chess_move_uses_agent_session(
        self,
        tmp_path,
        chess_settings: Settings,
    ) -> None:
        """Move tool writes to the active agent session."""
        chess_settings = chess_settings.model_copy(
            update={"thoughtbuddy_db_path": str(tmp_path / "chess.db")},
        )
        connect(tmp_path / "chess.db")
        from experiments.chess.plugin.tools import create_tools

        tools = {tool.name: tool for tool in create_tools(chess_settings)}
        with patch(
            "experiments.chess.plugin.service.play_opponent_move",
            side_effect=lambda session, _settings, *, player_color: session.move("e7e5"),
        ):
            payload = json.loads(tools["world_chess_move"].invoke({"uci": "e2e4"}))  # type: ignore[union-attr]

        assert payload["ok"] is True
        assert payload["move"] == "e2e4"
        assert payload["opponent_move"] == "e7e5"
        assert payload["session_id"]
