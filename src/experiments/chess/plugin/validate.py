"""Startup validation for the chess experiment."""

from __future__ import annotations

from config import Settings
from experiments.chess.plugin.engine import ChessEngineError, validate_stockfish


def validate_chess_experiment(settings: Settings) -> None:
    """Ensure the chess experiment is runnable when active.

    Args:
        settings: Application settings.

    Raises:
        ChessEngineError: When Stockfish is required but unavailable.
        ValueError: When chess configuration is invalid.
    """
    if settings.active_experiment.strip() != "chess":
        return

    agent_color = settings.chess_agent_color.strip().lower()
    if agent_color not in {"white", "black"}:
        msg = f"Invalid experiment.chess.agent_color: {settings.chess_agent_color!r}"
        raise ValueError(msg)

    validate_stockfish(settings)
