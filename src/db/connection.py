"""SQLite connection helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from db.schema import init_db


def ensure_data_dir(db_path: Path | str) -> None:
    """Create parent directories for the database file if needed.

    Args:
        db_path: Path to the SQLite database file.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open a SQLite connection with ThoughtBuddy pragmas and schema.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Initialized SQLite connection.
    """
    ensure_data_dir(db_path)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn
