"""Live Stockfish integration tests for the chess experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import Settings
from db.connection import connect
from experiments.chess.plugin.engine import ChessEngineError, resolve_engine_path
from experiments.chess.plugin.tools import create_tools
from store.chess import ChessStore

pytestmark = pytest.mark.integration


def _stockfish_settings(tmp_path: Path, settings: Settings) -> Settings | None:
    """Build settings when a Stockfish binary is available."""
    try:
        resolve_engine_path(".data/stockfish/engine")
    except ChessEngineError:
        pytest.skip("Run ./scripts/install_stockfish first")

    return settings.model_copy(
        update={
            "active_experiment": "chess",
            "thoughtbuddy_db_path": str(tmp_path / "test.db"),
            "chess_engine_path": ".data/stockfish/engine",
            "chess_agent_color": "white",
            "chess_move_time_seconds": 0.05,
        },
    )


class TestChessStockfishIntegration:
    """End-to-end chess move with a real Stockfish reply."""

    def test_agent_move_triggers_stockfish_reply(
        self,
        tmp_path: Path,
        settings: Settings,
    ) -> None:
        """Agent move should produce an automatic opponent reply."""
        resolved = _stockfish_settings(tmp_path, settings)
        if resolved is None:
            pytest.skip("Stockfish unavailable")

        tools = {tool.name: tool for tool in create_tools(resolved)}
        move_tool = tools["world_chess_move"]
        payload = json.loads(move_tool.invoke({"uci": "e2e4"}))  # type: ignore[union-attr]

        assert payload["ok"] is True
        assert payload["move"] == "e2e4"
        assert payload["opponent_move"]
        assert payload["status"]["turn"] == "white"

        conn = connect(tmp_path / "test.db")
        store = ChessStore(conn)
        sessions = store.list_sessions(opponent_type="agent", limit=1)
        assert sessions
        roles = {move.player_role for move in store.list_moves(sessions[0].id)}
        assert "agent" in roles
        assert "stockfish" in roles
        conn.close()
