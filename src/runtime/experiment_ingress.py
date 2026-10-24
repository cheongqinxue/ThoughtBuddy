"""Compose experiment-specific task materialization for the heartbeat."""

from __future__ import annotations

import sqlite3

from config import Settings


def materialize_experiment_tasks(conn: sqlite3.Connection, settings: Settings) -> int:
    """Materialize domain-specific wake tasks for the active experiment.

    Args:
        conn: SQLite connection participating in the caller's transaction.
        settings: Application settings.

    Returns:
        Number of new tasks enqueued.
    """
    if settings.active_experiment == "chess":
        from experiments.chess.plugin.materialize import materialize_chess_tasks

        return materialize_chess_tasks(conn, settings)
    return 0
