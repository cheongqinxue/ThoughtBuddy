"""Chess session and move persistence."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from utils.time import utc_now_str as _utc_now

ChessOpponentType = Literal["human", "agent"]
ChessSessionStatus = Literal["active", "completed", "abandoned"]
ChessMoveRole = Literal["human", "agent", "stockfish"]
ChessPlayerColor = Literal["white", "black"]

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@dataclass(frozen=True)
class ChessSessionRecord:
    """Persisted chess game session."""

    id: str
    opponent_type: ChessOpponentType
    agent_generation: int | None
    player_color: ChessPlayerColor
    stockfish_color: ChessPlayerColor
    status: ChessSessionStatus
    result: str | None
    starting_fen: str
    current_fen: str
    created_at: str
    updated_at: str
    completed_at: str | None


@dataclass(frozen=True)
class ChessMoveRecord:
    """One move within a chess session."""

    id: str
    session_id: str
    ply: int
    uci: str
    san: str | None
    player_role: ChessMoveRole
    agent_generation: int | None
    fen_after: str
    created_at: str


class ChessStore:
    """CRUD operations for chess sessions and move history."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the store.

        Args:
            conn: SQLite connection.
        """
        self._conn = conn

    def create_session(
        self,
        *,
        opponent_type: ChessOpponentType,
        player_color: ChessPlayerColor,
        agent_generation: int | None = None,
        starting_fen: str = STARTING_FEN,
        session_id: str | None = None,
    ) -> ChessSessionRecord:
        """Create a new chess session.

        Args:
            opponent_type: Whether the non-engine player is human or agent.
            player_color: Color played by the human or agent.
            agent_generation: Agent generation label when ``opponent_type`` is agent.
            starting_fen: Initial board FEN.
            session_id: Optional explicit session id.

        Returns:
            Created session row.
        """
        now = _utc_now()
        session_id = session_id or str(uuid.uuid4())
        stockfish_color: ChessPlayerColor = "black" if player_color == "white" else "white"
        self._conn.execute(
            """
            INSERT INTO chess_sessions (
                id, opponent_type, agent_generation, player_color, stockfish_color,
                status, result, starting_fen, current_fen,
                created_at, updated_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, 'active', NULL, ?, ?, ?, ?, NULL)
            """,
            (
                session_id,
                opponent_type,
                agent_generation,
                player_color,
                stockfish_color,
                starting_fen,
                starting_fen,
                now,
                now,
            ),
        )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> ChessSessionRecord:
        """Load one session by id.

        Args:
            session_id: Session identifier.

        Returns:
            Session row.

        Raises:
            KeyError: When the session does not exist.
        """
        row = self._conn.execute(
            "SELECT * FROM chess_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            msg = f"Chess session not found: {session_id}"
            raise KeyError(msg)
        return _session_from_row(row)

    def list_sessions(
        self,
        *,
        opponent_type: ChessOpponentType | None = None,
        agent_generation: int | None = None,
        status: ChessSessionStatus | None = None,
        limit: int = 50,
    ) -> list[ChessSessionRecord]:
        """List chess sessions newest first.

        Args:
            opponent_type: Optional filter by opponent type.
            agent_generation: Optional filter by agent generation.
            status: Optional filter by session status.
            limit: Maximum rows to return.

        Returns:
            Matching session rows.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if opponent_type is not None:
            clauses.append("opponent_type = ?")
            params.append(opponent_type)
        if agent_generation is not None:
            clauses.append("agent_generation = ?")
            params.append(agent_generation)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT * FROM chess_sessions
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [_session_from_row(row) for row in rows]

    def get_active_agent_session(self, agent_generation: int) -> ChessSessionRecord | None:
        """Return the active agent session for one generation, if any.

        Args:
            agent_generation: Agent generation identifier.

        Returns:
            Active session row, or ``None``.
        """
        row = self._conn.execute(
            """
            SELECT * FROM chess_sessions
            WHERE opponent_type = 'agent'
              AND agent_generation = ?
              AND status = 'active'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (agent_generation,),
        ).fetchone()
        return _session_from_row(row) if row else None

    def update_session_fen(
        self,
        session_id: str,
        *,
        current_fen: str,
        status: ChessSessionStatus | None = None,
        result: str | None = None,
    ) -> ChessSessionRecord:
        """Update the live position for a session.

        Args:
            session_id: Session identifier.
            current_fen: Latest board FEN.
            status: Optional new status.
            result: Optional game result label.

        Returns:
            Updated session row.
        """
        now = _utc_now()
        completed_at: str | None = None
        if status == "completed":
            completed_at = now

        if status is None and result is None:
            self._conn.execute(
                """
                UPDATE chess_sessions
                SET current_fen = ?, updated_at = ?
                WHERE id = ?
                """,
                (current_fen, now, session_id),
            )
        else:
            self._conn.execute(
                """
                UPDATE chess_sessions
                SET current_fen = ?, status = COALESCE(?, status),
                    result = COALESCE(?, result),
                    updated_at = ?, completed_at = COALESCE(?, completed_at)
                WHERE id = ?
                """,
                (current_fen, status, result, now, completed_at, session_id),
            )
        return self.get_session(session_id)

    def abandon_session(self, session_id: str) -> ChessSessionRecord:
        """Mark a session abandoned.

        Args:
            session_id: Session identifier.

        Returns:
            Updated session row.
        """
        now = _utc_now()
        self._conn.execute(
            """
            UPDATE chess_sessions
            SET status = 'abandoned', updated_at = ?, completed_at = ?
            WHERE id = ?
            """,
            (now, now, session_id),
        )
        return self.get_session(session_id)

    def append_move(
        self,
        *,
        session_id: str,
        uci: str,
        san: str | None,
        player_role: ChessMoveRole,
        fen_after: str,
        agent_generation: int | None = None,
        move_id: str | None = None,
    ) -> ChessMoveRecord:
        """Append one move to a session.

        Args:
            session_id: Session identifier.
            uci: Move in UCI notation.
            san: Optional SAN notation.
            player_role: Who played the move.
            fen_after: Board FEN after the move.
            agent_generation: Agent generation when role is agent.
            move_id: Optional explicit move id.

        Returns:
            Created move row.
        """
        ply_row = self._conn.execute(
            "SELECT COALESCE(MAX(ply), 0) + 1 FROM chess_moves WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        ply = int(ply_row[0])
        now = _utc_now()
        move_id = move_id or str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO chess_moves (
                id, session_id, ply, uci, san, player_role,
                agent_generation, fen_after, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                move_id,
                session_id,
                ply,
                uci,
                san,
                player_role,
                agent_generation,
                fen_after,
                now,
            ),
        )
        return self.get_move(move_id)

    def list_moves(self, session_id: str) -> list[ChessMoveRecord]:
        """Return moves for a session in ply order.

        Args:
            session_id: Session identifier.

        Returns:
            Ordered move rows.
        """
        rows = self._conn.execute(
            """
            SELECT * FROM chess_moves
            WHERE session_id = ?
            ORDER BY ply ASC
            """,
            (session_id,),
        ).fetchall()
        return [_move_from_row(row) for row in rows]

    def get_move(self, move_id: str) -> ChessMoveRecord:
        """Load one move by id.

        Args:
            move_id: Move identifier.

        Returns:
            Move row.

        Raises:
            KeyError: When the move does not exist.
        """
        row = self._conn.execute(
            "SELECT * FROM chess_moves WHERE id = ?",
            (move_id,),
        ).fetchone()
        if row is None:
            msg = f"Chess move not found: {move_id}"
            raise KeyError(msg)
        return _move_from_row(row)


def _session_from_row(row: sqlite3.Row) -> ChessSessionRecord:
    """Convert a SQLite row to ``ChessSessionRecord``."""
    return ChessSessionRecord(
        id=row["id"],
        opponent_type=row["opponent_type"],
        agent_generation=row["agent_generation"],
        player_color=row["player_color"],
        stockfish_color=row["stockfish_color"],
        status=row["status"],
        result=row["result"],
        starting_fen=row["starting_fen"],
        current_fen=row["current_fen"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


def _move_from_row(row: sqlite3.Row) -> ChessMoveRecord:
    """Convert a SQLite row to ``ChessMoveRecord``."""
    return ChessMoveRecord(
        id=row["id"],
        session_id=row["session_id"],
        ply=row["ply"],
        uci=row["uci"],
        san=row["san"],
        player_role=row["player_role"],
        agent_generation=row["agent_generation"],
        fen_after=row["fen_after"],
        created_at=row["created_at"],
    )


def session_to_dict(session: ChessSessionRecord, *, moves: list[ChessMoveRecord] | None = None) -> dict[str, Any]:
    """Serialize a session for JSON APIs.

    Args:
        session: Session row.
        moves: Optional move list to include.

    Returns:
        JSON-serializable mapping.
    """
    payload: dict[str, Any] = {
        "id": session.id,
        "opponent_type": session.opponent_type,
        "agent_generation": session.agent_generation,
        "player_color": session.player_color,
        "stockfish_color": session.stockfish_color,
        "status": session.status,
        "result": session.result,
        "starting_fen": session.starting_fen,
        "current_fen": session.current_fen,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "completed_at": session.completed_at,
        "label": _session_label(session),
    }
    if moves is not None:
        payload["moves"] = [move_to_dict(move) for move in moves]
    return payload


def move_to_dict(move: ChessMoveRecord) -> dict[str, Any]:
    """Serialize a move for JSON APIs.

    Args:
        move: Move row.

    Returns:
        JSON-serializable mapping.
    """
    return {
        "id": move.id,
        "session_id": move.session_id,
        "ply": move.ply,
        "uci": move.uci,
        "san": move.san,
        "player_role": move.player_role,
        "agent_generation": move.agent_generation,
        "fen_after": move.fen_after,
        "created_at": move.created_at,
    }


def _session_label(session: ChessSessionRecord) -> str:
    """Build a human-readable session label."""
    if session.opponent_type == "agent":
        generation = session.agent_generation if session.agent_generation is not None else 0
        return f"Agent Gen {generation} vs Stockfish"
    return "Human vs Stockfish"
