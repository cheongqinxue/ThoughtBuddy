"""Tests for the chess experiment."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from config import Settings
from db.connection import connect
from experiments.chess.plugin.game import ChessSession
from tools.tool_boundary import wrap_tool_with_boundary

from experiments.chess.plugin.tools import create_tools
from store.chess import ChessStore


class TestChessSession:
    """Chess board session behavior."""

    def test_starts_at_standard_position(self) -> None:
        """New session uses the standard starting FEN."""
        status = ChessSession().status()
        assert status.turn == "white"
        assert status.is_game_over is False
        assert status.legal_move_count == 20
        assert "e2e4" in status.legal_moves

    def test_legal_move_updates_turn(self) -> None:
        """Applying a legal move advances the side to move."""
        session = ChessSession()
        result = session.move("e2e4")
        assert result["ok"] is True
        assert result["status"]["turn"] == "black"

    def test_illegal_move_rejected(self) -> None:
        """Reject moves that are not legal in the current position."""
        session = ChessSession()
        result = session.move("e2e5")
        assert result["ok"] is False
        assert "Illegal move" in result["error"]
        assert result["status"]["turn"] == "white"


class TestChessTools:
    """Chess experiment agent tools."""

    @pytest.fixture
    def chess_tools(self, tmp_path, settings: Settings) -> dict[str, object]:
        """Chess tools backed by a temp database."""
        settings = settings.model_copy(update={"thoughtbuddy_db_path": str(tmp_path / "test.db")})
        tools = [wrap_tool_with_boundary(tool) for tool in create_tools(settings)]
        return {tool.name: tool for tool in tools}

    def test_world_chess_status_returns_json_board(self, chess_tools: dict[str, object]) -> None:
        """Status tool returns parseable JSON with FEN and legal moves."""
        status_tool = chess_tools["world_chess_status"]
        payload = json.loads(status_tool.invoke({}))  # type: ignore[union-attr]
        assert payload["turn"] == "white"
        assert "rnbqkbnr" in payload["fen"]
        assert isinstance(payload["legal_moves"], list)

    def test_world_chess_move_persists_to_world_store(
        self,
        tmp_path,
        settings: Settings,
        chess_tools: dict[str, object],
    ) -> None:
        """Moves persist FEN and log a world observation."""
        move_tool = chess_tools["world_chess_move"]
        with patch(
            "experiments.chess.plugin.service.play_opponent_move",
            side_effect=lambda session, _settings, **_: session.move("e7e5"),
        ):
            payload = json.loads(move_tool.invoke({"uci": "e2e4"}))  # type: ignore[union-attr]
        assert payload["ok"] is True
        assert payload["opponent_move"] == "e7e5"

        conn = connect(tmp_path / "test.db")
        store = ChessStore(conn)
        sessions = store.list_sessions(opponent_type="agent", limit=5)
        assert sessions
        moves = store.list_moves(sessions[0].id)
        assert any(move.player_role == "agent" for move in moves)
        conn.close()

    def test_world_chess_move_rejects_when_not_agent_turn(
        self,
        chess_tools: dict[str, object],
    ) -> None:
        """Reject moves when Stockfish is on move."""
        move_tool = chess_tools["world_chess_move"]
        with patch("experiments.chess.plugin.service.play_opponent_move", return_value=None):
            payload = json.loads(move_tool.invoke({"uci": "e2e4"}))  # type: ignore[union-attr]
        assert payload["ok"] is True

        blocked = json.loads(move_tool.invoke({"uci": "g1f3"}))  # type: ignore[union-attr]
        assert blocked["ok"] is False
        assert "Not your turn" in blocked["message"]

    def test_world_chess_move_rejects_illegal_move(self, chess_tools: dict[str, object]) -> None:
        """Move tool returns an error for illegal moves."""
        move_tool = chess_tools["world_chess_move"]
        payload = json.loads(move_tool.invoke({"uci": "e2e5"}))  # type: ignore[union-attr]
        assert payload["ok"] is False
        assert "Illegal move" in payload["message"]
