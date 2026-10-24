"""Chess game orchestration for agent tools and HTTP APIs."""

from __future__ import annotations

import sqlite3
from typing import Any, Literal

import chess

from config import Settings
from experiments.chess.plugin.game import ChessSession
from experiments.chess.plugin.opponent import is_agent_turn, play_opponent_move
from store.chess import ChessMoveRecord, ChessSessionRecord, ChessStore, session_to_dict


class ChessGameError(Exception):
    """Raised when a chess action is invalid."""


class ChessGameService:
    """Manage session-backed chess games against Stockfish."""

    def __init__(self, conn: sqlite3.Connection, settings: Settings) -> None:
        """Initialize the service.

        Args:
            conn: SQLite connection.
            settings: Application settings.
        """
        self._conn = conn
        self._settings = settings
        self._store = ChessStore(conn)

    def list_sessions(
        self,
        *,
        opponent_type: Literal["human", "agent"] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List stored chess sessions.

        Args:
            opponent_type: Optional opponent filter.
            limit: Maximum sessions to return.

        Returns:
            JSON-ready session summaries.
        """
        sessions = self._store.list_sessions(opponent_type=opponent_type, limit=limit)
        return [session_to_dict(session) for session in sessions]

    def get_session_detail(self, session_id: str) -> dict[str, Any]:
        """Return one session with full move history.

        Args:
            session_id: Session identifier.

        Returns:
            JSON-ready session payload including moves.
        """
        session = self._store.get_session(session_id)
        moves = self._store.list_moves(session_id)
        return session_to_dict(session, moves=moves)

    def get_active_agent_session(self) -> dict[str, Any] | None:
        """Return the active agent session for the configured generation.

        Returns:
            Session detail payload, or ``None`` when no active game exists.
        """
        session = self._store.get_active_agent_session(self._settings.chess_agent_generation)
        if session is None:
            return None
        moves = self._store.list_moves(session.id)
        return session_to_dict(session, moves=moves)

    def create_human_session(self, *, player_color: Literal["white", "black"] = "white") -> dict[str, Any]:
        """Start a human vs Stockfish session.

        Args:
            player_color: Color the human plays.

        Returns:
            JSON-ready session payload.
        """
        session = self._store.create_session(
            opponent_type="human",
            player_color=player_color,
        )
        if player_color == "black":
            return self._apply_stockfish_reply(session.id)
        return session_to_dict(session, moves=[])

    def create_agent_session(self, *, force_new: bool = False) -> dict[str, Any]:
        """Start or resume the active agent session for this generation.

        Args:
            force_new: Abandon any active agent session and start fresh.

        Returns:
            JSON-ready session payload.
        """
        generation = self._settings.chess_agent_generation
        existing = self._store.get_active_agent_session(generation)
        if existing is not None and not force_new:
            live = self._session_from_record(existing)
            if live.status().is_game_over:
                self._store.abandon_session(existing.id)
            else:
                return self.get_session_detail(existing.id)

        if existing is not None and force_new:
            self._store.abandon_session(existing.id)

        player_color = self._settings.chess_agent_color.strip().lower()
        if player_color not in {"white", "black"}:
            msg = f"Invalid chess agent color: {self._settings.chess_agent_color!r}"
            raise ChessGameError(msg)

        session = self._store.create_session(
            opponent_type="agent",
            player_color=player_color,  # type: ignore[arg-type]
            agent_generation=generation,
        )
        if player_color == "black":
            return self._apply_stockfish_reply(session.id)
        return session_to_dict(session, moves=[])

    def apply_human_move(self, session_id: str, uci: str) -> dict[str, Any]:
        """Apply a human move and let Stockfish reply when appropriate.

        Args:
            session_id: Session identifier.
            uci: Move in UCI notation.

        Returns:
            Updated session detail payload.
        """
        return self._apply_player_move(
            session_id,
            uci,
            expected_role="human",
        )

    def apply_agent_move(self, session_id: str, uci: str) -> dict[str, Any]:
        """Apply an agent move and let Stockfish reply when appropriate.

        Args:
            session_id: Session identifier.
            uci: Move in UCI notation.

        Returns:
            Move result payload for agent tools.
        """
        detail = self._apply_player_move(
            session_id,
            uci,
            expected_role="agent",
        )
        moves = detail.get("moves") or []
        agent_move = next(
            (move["uci"] for move in reversed(moves) if move["player_role"] == "agent"),
            uci,
        )
        stockfish_move = next(
            (move["uci"] for move in reversed(moves) if move["player_role"] == "stockfish"),
            None,
        )
        return {
            "ok": True,
            "session_id": session_id,
            "move": agent_move,
            "opponent_move": stockfish_move,
            "status": self._status_dict(detail["current_fen"]),
            "session": detail,
        }

    def get_agent_status(self) -> dict[str, Any]:
        """Return status for the active agent session, creating one if needed.

        Returns:
            Board status payload for agent tools.
        """
        session_detail = self.create_agent_session(force_new=False)
        board = chess.Board(session_detail["current_fen"])
        return {
            "session_id": session_detail["id"],
            **self._status_dict(board.fen()),
            "label": session_detail["label"],
            "agent_generation": self._settings.chess_agent_generation,
        }

    def reset_agent_session(self) -> dict[str, Any]:
        """Abandon the current agent session and start a new game.

        Returns:
            JSON-ready new session payload.
        """
        return self.create_agent_session(force_new=True)

    def _apply_player_move(
        self,
        session_id: str,
        uci: str,
        *,
        expected_role: Literal["human", "agent"],
    ) -> dict[str, Any]:
        """Apply a human or agent move to a session.

        Args:
            session_id: Session identifier.
            uci: Move in UCI notation.
            expected_role: Required non-engine role for this action.

        Returns:
            Updated session detail payload.

        Raises:
            ChessGameError: When the session or move is invalid.
        """
        record = self._store.get_session(session_id)
        if record.opponent_type != expected_role:
            msg = f"Session {session_id} is not a {expected_role} game"
            raise ChessGameError(msg)
        if record.status != "active":
            msg = f"Session {session_id} is not active"
            raise ChessGameError(msg)

        session = self._session_from_record(record)
        if session.status().is_game_over:
            msg = "Game is already over"
            raise ChessGameError(msg)

        if not self._is_player_turn(record, session):
            msg = "Not your turn"
            raise ChessGameError(msg)

        if expected_role == "agent" and not is_agent_turn(session, self._settings):
            msg = f"Agent ({self._settings.chess_agent_color}) is not on move"
            raise ChessGameError(msg)

        board = chess.Board(session.status().fen)
        move = self._parse_move(board, uci)
        san = board.san(move)
        result = session.move(uci)
        if not result.get("ok"):
            msg = str(result.get("error", "Illegal move"))
            raise ChessGameError(msg)

        generation = self._settings.chess_agent_generation if expected_role == "agent" else None
        self._store.append_move(
            session_id=session_id,
            uci=move.uci(),
            san=san,
            player_role=expected_role,
            fen_after=session.status().fen,
            agent_generation=generation,
        )
        self._finalize_if_over(session_id, session)

        if not session.status().is_game_over:
            return self._apply_stockfish_reply(session_id, session=session)

        return self.get_session_detail(session_id)

    def _apply_stockfish_reply(
        self,
        session_id: str,
        *,
        session: ChessSession | None = None,
    ) -> dict[str, Any]:
        """Let Stockfish move when it is the engine's turn.

        Args:
            session_id: Session identifier.
            session: Optional in-memory session; loaded from store when omitted.

        Returns:
            Updated session detail payload.
        """
        record = self._store.get_session(session_id)
        session = session or self._session_from_record(record)
        if session.status().is_game_over:
            return self.get_session_detail(session_id)

        fen_before_opponent = session.status().fen
        opponent_result = play_opponent_move(
            session,
            self._settings,
            player_color=record.player_color,
        )
        if opponent_result is not None and opponent_result.get("ok"):
            stockfish_move = opponent_result.get("move")
            if isinstance(stockfish_move, str):
                board_before = chess.Board(fen_before_opponent)
                move_obj = chess.Move.from_uci(stockfish_move)
                san = board_before.san(move_obj)
                self._store.append_move(
                    session_id=session_id,
                    uci=stockfish_move,
                    san=san,
                    player_role="stockfish",
                    fen_after=session.status().fen,
                )
            self._finalize_if_over(session_id, session)

        return self.get_session_detail(session_id)

    def _finalize_if_over(self, session_id: str, session: ChessSession) -> None:
        """Mark session completed when the board reaches game over.

        Args:
            session_id: Session identifier.
            session: In-memory chess session.
        """
        status = session.status()
        if not status.is_game_over:
            self._store.update_session_fen(session_id, current_fen=status.fen)
            return

        result = self._game_result(status)
        self._store.update_session_fen(
            session_id,
            current_fen=status.fen,
            status="completed",
            result=result,
        )

    @staticmethod
    def _game_result(status) -> str:
        """Map board status to a stored result label.

        Args:
            status: Chess status snapshot.

        Returns:
            Result label string.
        """
        if status.is_checkmate:
            winner = "black" if status.turn == "white" else "white"
            return f"{winner}_win"
        return "draw"

    @staticmethod
    def _session_from_record(record: ChessSessionRecord) -> ChessSession:
        """Hydrate an in-memory session from a stored record.

        Args:
            record: Persisted session row.

        Returns:
            Loaded chess session.
        """
        session = ChessSession()
        session.load_fen(record.current_fen)
        return session

    @staticmethod
    def _is_player_turn(record: ChessSessionRecord, session: ChessSession) -> bool:
        """Return whether the human/agent side is on move.

        Args:
            record: Persisted session row.
            session: In-memory chess session.

        Returns:
            True when the non-engine player may move.
        """
        turn = session.status().turn
        return turn == record.player_color

    @staticmethod
    def _parse_move(board: chess.Board, uci: str) -> chess.Move:
        """Parse and validate a UCI move.

        Args:
            board: Current board position.
            uci: Move in UCI notation.

        Returns:
            Parsed move.

        Raises:
            ChessGameError: When the move is invalid or illegal.
        """
        normalized = uci.strip().lower()
        try:
            move = chess.Move.from_uci(normalized)
        except ValueError as exc:
            msg = f"Invalid UCI move: {uci!r}"
            raise ChessGameError(msg) from exc
        if move not in board.legal_moves:
            msg = f"Illegal move: {uci!r}"
            raise ChessGameError(msg)
        return move

    @staticmethod
    def _status_dict(fen: str) -> dict[str, Any]:
        """Build a JSON status payload from FEN.

        Args:
            fen: Board FEN string.

        Returns:
            Status mapping compatible with agent tools.
        """
        session = ChessSession()
        session.load_fen(fen)
        return session.status().as_dict()
