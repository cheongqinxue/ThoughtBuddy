"""In-memory chess session for the chess experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess


@dataclass(frozen=True)
class ChessStatus:
    """Snapshot of the current chess position."""

    fen: str
    turn: str
    is_check: bool
    is_checkmate: bool
    is_stalemate: bool
    is_game_over: bool
    legal_move_count: int
    legal_moves: list[str]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable status payload.

        Returns:
            Status fields as a plain dictionary.
        """
        return {
            "fen": self.fen,
            "turn": self.turn,
            "is_check": self.is_check,
            "is_checkmate": self.is_checkmate,
            "is_stalemate": self.is_stalemate,
            "is_game_over": self.is_game_over,
            "legal_move_count": self.legal_move_count,
            "legal_moves": self.legal_moves,
        }


class ChessSession:
    """Mutable chess board the agent interacts with via experiment tools."""

    def __init__(self) -> None:
        """Initialize a new game at the standard starting position."""
        self._board = chess.Board()

    def reset(self) -> None:
        """Reset the board to the standard starting position."""
        self._board = chess.Board()

    def load_fen(self, fen: str) -> None:
        """Load a position from FEN notation.

        Args:
            fen: FEN string.

        Raises:
            ValueError: When FEN is invalid.
        """
        self._board = chess.Board(fen)

    def status(self) -> ChessStatus:
        """Return the current board status.

        Returns:
            Position snapshot including FEN, turn, and legal moves.
        """
        legal_moves = [move.uci() for move in self._board.legal_moves]
        return ChessStatus(
            fen=self._board.fen(),
            turn="white" if self._board.turn == chess.WHITE else "black",
            is_check=self._board.is_check(),
            is_checkmate=self._board.is_checkmate(),
            is_stalemate=self._board.is_stalemate(),
            is_game_over=self._board.is_game_over(),
            legal_move_count=len(legal_moves),
            legal_moves=legal_moves,
        )

    def move(self, uci: str) -> dict[str, Any]:
        """Apply a move in UCI notation (e.g. ``e2e4``).

        Args:
            uci: Move in UCI format.

        Returns:
            Result payload with ``ok`` flag, optional ``error``, and updated ``status``.
        """
        normalized = uci.strip().lower()
        try:
            move = chess.Move.from_uci(normalized)
        except ValueError:
            return {
                "ok": False,
                "error": f"Invalid UCI move: {uci!r}",
            }

        if move not in self._board.legal_moves:
            return {
                "ok": False,
                "error": f"Illegal move: {uci!r}",
                "status": self.status().as_dict(),
            }

        self._board.push(move)
        return {
            "ok": True,
            "move": normalized,
            "status": self.status().as_dict(),
        }


_session: ChessSession | None = None


def get_session() -> ChessSession:
    """Return the process-wide chess session, creating it if needed.

    Returns:
        Shared ``ChessSession`` instance.
    """
    global _session
    if _session is None:
        _session = ChessSession()
    return _session


def reset_session() -> ChessSession:
    """Reset and return the process-wide chess session.

    Returns:
        Fresh ``ChessSession`` at the starting position.
    """
    global _session
    _session = ChessSession()
    return _session
