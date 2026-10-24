"""Chess experiment task materialization for the heartbeat."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from config import Settings
from experiments.chess.plugin.service import ChessGameService
from store.tasks import TaskStore

_CHESS_START_MESSAGE = (
    "Play chess vs Stockfish. Check board status, start or resume the agent game, "
    "and make your move if it is legal."
)
_CHESS_CONTINUE_MESSAGE = (
    "Continue the active chess game vs Stockfish. Check status and make your move."
)


def materialize_chess_tasks(conn: sqlite3.Connection, settings: Settings) -> int:
    """Enqueue chess wake tasks when the agent should act.

    Args:
        conn: SQLite connection participating in the caller's transaction.
        settings: Application settings.

    Returns:
        Number of new tasks enqueued.
    """
    if settings.active_experiment != "chess":
        return 0

    task_store = TaskStore(conn)
    service = ChessGameService(conn, settings)
    session = service.get_active_agent_session()
    if session is None:
        return _maybe_enqueue_start(conn, task_store)

    if session["status"] != "active" or not _is_agent_turn(session):
        return 0

    if _has_inflight_chess_task(conn, task_store, kind="chess_continue", session_id=session["id"]):
        return 0

    task_store.enqueue_user_task(
        message=_CHESS_CONTINUE_MESSAGE,
        payload={"kind": "chess_continue", "session_id": session["id"]},
    )
    return 1


def _maybe_enqueue_start(conn: sqlite3.Connection, task_store: TaskStore) -> int:
    """Enqueue a one-shot start task when no chess work is already queued.

    Args:
        conn: SQLite connection.
        task_store: Task queue store.

    Returns:
        ``1`` when a start task was enqueued, otherwise ``0``.
    """
    if _has_inflight_chess_task(conn, task_store, kind="chess_start"):
        return 0
    if _has_inflight_chess_task(conn, task_store, kind="chess_continue"):
        return 0

    task_store.enqueue_user_task(
        message=_CHESS_START_MESSAGE,
        payload={"kind": "chess_start"},
    )
    return 1


def _has_inflight_chess_task(
    conn: sqlite3.Connection,
    task_store: TaskStore,
    *,
    kind: str,
    session_id: str | None = None,
) -> bool:
    """Return whether a matching chess task is already pending or running.

    Args:
        conn: SQLite connection.
        task_store: Task queue store.
        kind: Chess task payload kind.
        session_id: Optional session id for continuation tasks.

    Returns:
        True when a matching inflight task exists.
    """
    for task in task_store.list_pending():
        if _matches_chess_task(task.payload, kind=kind, session_id=session_id):
            return True

    rows = conn.execute(
        """
        SELECT payload
        FROM tasks
        WHERE status = 'running'
        """,
    ).fetchall()
    for row in rows:
        payload = json.loads(row["payload"])
        if _matches_chess_task(payload, kind=kind, session_id=session_id):
            return True
    return False


def _matches_chess_task(
    payload: dict[str, Any],
    *,
    kind: str,
    session_id: str | None,
) -> bool:
    """Return whether a payload matches the requested chess task kind."""
    if payload.get("kind") != kind:
        return False
    if session_id is None:
        return True
    return payload.get("session_id") == session_id


def _is_agent_turn(session: dict[str, Any]) -> bool:
    """Return whether the agent side is on move for a session snapshot."""
    turn = session["current_fen"].split()[1]
    turn_color = "white" if turn == "w" else "black"
    return turn_color == session["player_color"]
