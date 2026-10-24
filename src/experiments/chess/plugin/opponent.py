"""Automatic Stockfish replies for the chess experiment."""

from __future__ import annotations

from typing import Any

import chess

from config import Settings
from experiments.chess.plugin.engine import ChessEngineError, choose_opponent_move
from experiments.chess.plugin.game import ChessSession


def opponent_color(agent_color: str) -> str:
    """Return the opponent color for a configured agent color.

    Args:
        agent_color: ``white`` or ``black``.

    Returns:
        Opposite side name.

    Raises:
        ValueError: When ``agent_color`` is invalid.
    """
    normalized = agent_color.strip().lower()
    if normalized == "white":
        return "black"
    if normalized == "black":
        return "white"
    msg = f"Invalid chess agent color: {agent_color!r} (expected 'white' or 'black')"
    raise ValueError(msg)


def is_player_turn(session: ChessSession, player_color: str) -> bool:
    """Return whether the human/agent side is on move.

    Args:
        session: Current chess session.
        player_color: Side the human or agent plays (``white`` or ``black``).

    Returns:
        True when the non-engine player may move.
    """
    return session.status().turn == player_color.strip().lower()


def is_agent_turn(session: ChessSession, settings: Settings) -> bool:
    """Return whether the configured agent side is on move.

    Args:
        session: Current chess session.
        settings: Application settings.

    Returns:
        True when the agent may play.
    """
    return is_player_turn(session, settings.chess_agent_color)


def play_opponent_move(
    session: ChessSession,
    settings: Settings,
    *,
    player_color: str,
) -> dict[str, Any] | None:
    """Play one Stockfish move when it is the engine's turn.

    Args:
        session: Current chess session.
        settings: Application settings.
        player_color: Side the human or agent plays (``white`` or ``black``).

    Returns:
        Move result payload, or ``None`` when no opponent move was required.

    Raises:
        ChessEngineError: When Stockfish fails to produce a legal move.
    """
    status = session.status()
    if status.is_game_over or is_player_turn(session, player_color):
        return None

    board = chess.Board(status.fen)
    move = choose_opponent_move(board, settings)
    result = session.move(move.uci())
    if not result.get("ok"):
        msg = f"Stockfish produced illegal move {move.uci()}: {result.get('error')}"
        raise ChessEngineError(msg)
    return result
