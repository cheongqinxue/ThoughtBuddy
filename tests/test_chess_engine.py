"""Tests for Stockfish engine helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import pytest

from config import Settings
from experiments.chess.plugin.engine import ChessEngineError, choose_opponent_move, resolve_engine_path
from experiments.chess.plugin.opponent import is_agent_turn, play_opponent_move
from experiments.chess.plugin.game import ChessSession
from experiments.chess.plugin.validate import validate_chess_experiment


class TestResolveEnginePath:
    """Engine path resolution."""

    def test_resolve_relative_project_binary(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolve a project-local Stockfish binary path."""
        engine = tmp_path / ".data" / "stockfish" / "engine"
        engine.parent.mkdir(parents=True)
        engine.write_text("#!/bin/sh\n", encoding="utf-8")
        engine.chmod(0o755)
        monkeypatch.chdir(tmp_path)

        resolved = resolve_engine_path(".data/stockfish/engine")
        assert resolved == engine

    def test_missing_engine_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raise a helpful error when Stockfish is missing."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ChessEngineError, match="Install it with ./scripts/install_stockfish"):
            resolve_engine_path("missing-stockfish")


class TestChessOpponent:
    """Opponent turn handling."""

    def test_is_agent_turn_on_white_start(self, settings: Settings) -> None:
        """White starts when the agent plays white."""
        session = ChessSession()
        settings = settings.model_copy(update={"chess_agent_color": "white"})
        assert is_agent_turn(session, settings) is True

    def test_play_opponent_move_replies_after_agent_move(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stockfish move is applied after the agent moves."""
        session = ChessSession()
        session.move("e2e4")
        settings = settings.model_copy(
            update={
                "chess_engine_path": str(tmp_path / "engine"),
                "chess_agent_color": "white",
            },
        )
        mock_move = chess.Move.from_uci("e7e5")

        with patch(
            "experiments.chess.plugin.opponent.choose_opponent_move",
            return_value=mock_move,
        ):
            result = play_opponent_move(session, settings, player_color="white")

        assert result is not None
        assert result["ok"] is True
        assert result["move"] == "e7e5"
        assert session.status().turn == "white"


class TestValidateChessExperiment:
    """Startup validation for chess."""

    def test_noop_when_chess_inactive(self, settings: Settings) -> None:
        """Skip validation when chess is not active."""
        validate_chess_experiment(settings.model_copy(update={"active_experiment": ""}))

    def test_requires_stockfish_when_chess_active(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fail fast when chess is active but Stockfish is missing."""
        monkeypatch.chdir(tmp_path)
        active = settings.model_copy(update={"active_experiment": "chess"})
        with pytest.raises(ChessEngineError, match="Install it with ./scripts/install_stockfish"):
            validate_chess_experiment(active)

    def test_validates_when_stockfish_present(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pass validation when Stockfish starts successfully."""
        engine = tmp_path / ".data" / "stockfish" / "engine"
        engine.parent.mkdir(parents=True)
        engine.write_text("#!/bin/sh\n", encoding="utf-8")
        engine.chmod(0o755)
        monkeypatch.chdir(tmp_path)

        active = settings.model_copy(
            update={
                "active_experiment": "chess",
                "chess_engine_path": ".data/stockfish/engine",
            },
        )

        mock_engine = MagicMock()
        mock_engine.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.__exit__ = MagicMock(return_value=False)
        with patch(
            "experiments.chess.plugin.engine.chess.engine.SimpleEngine.popen_uci",
            return_value=mock_engine,
        ):
            validate_chess_experiment(active)

        mock_engine.ping.assert_called_once()


class TestChooseOpponentMove:
    """Engine move selection with mocked UCI."""

    def test_choose_opponent_move_uses_stockfish(
        self,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """Return the move selected by Stockfish."""
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.move = chess.Move.from_uci("d2d4")
        mock_engine.play.return_value = mock_result
        mock_engine.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.__exit__ = MagicMock(return_value=False)

        settings = settings.model_copy(update={"chess_engine_path": str(tmp_path / "engine")})

        @contextmanager
        def _mock_open(_settings: Settings):
            yield mock_engine

        with patch("experiments.chess.plugin.engine.open_stockfish", _mock_open):
            move = choose_opponent_move(chess.Board(), settings)

        assert move.uci() == "d2d4"
