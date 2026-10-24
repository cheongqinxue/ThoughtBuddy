"""Monitored subprocess persistence."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from utils.time import utc_now_str as _utc_now

MonitorStatus = Literal[
    "running",
    "completed",
    "failed",
    "timed_out",
    "interrupted",
    "cancelled",
]


@dataclass(frozen=True)
class MonitoredProcess:
    """Persisted monitored subprocess row."""

    id: str
    thread_id: str
    command: list[str]
    description: str
    status: MonitorStatus
    pid: int | None
    pgid: int | None
    log_dir: str
    timeout_seconds: int
    exit_code: int | None
    signal_name: str | None
    started_at: str
    finished_at: str | None
    created_at: str
    updated_at: str


class MonitorStore:
    """CRUD for monitored subprocesses."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the store.

        Args:
            conn: SQLite connection.
        """
        self._conn = conn

    def create(
        self,
        *,
        thread_id: str,
        command: list[str],
        description: str,
        log_dir: str,
        timeout_seconds: int,
        pid: int,
        pgid: int,
        monitor_id: str | None = None,
    ) -> MonitoredProcess:
        """Insert a running monitored process row.

        Args:
            thread_id: Conversation thread id.
            command: argv passed to the subprocess.
            description: Agent-provided summary.
            log_dir: Directory for stdout/stderr logs.
            timeout_seconds: Kill after this many seconds.
            pid: Subprocess pid.
            pgid: Process group id for group kill.
            monitor_id: Optional explicit monitor id.

        Returns:
            Created monitor row.
        """
        now = _utc_now()
        resolved_id = monitor_id or str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO monitored_processes (
                id, thread_id, command_json, description, status, pid, pgid,
                log_dir, timeout_seconds, exit_code, signal_name,
                started_at, finished_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, NULL, NULL, ?, NULL, ?, ?)
            """,
            (
                resolved_id,
                thread_id,
                json.dumps(command),
                description,
                pid,
                pgid,
                log_dir,
                timeout_seconds,
                now,
                now,
                now,
            ),
        )
        row = self.get(resolved_id)
        if row is None:
            msg = f"Monitor not found after insert: {resolved_id}"
            raise RuntimeError(msg)
        return row

    def get(self, monitor_id: str) -> MonitoredProcess | None:
        """Return a monitor by id.

        Args:
            monitor_id: Monitor identifier.

        Returns:
            Monitor row if found.
        """
        row = self._conn.execute(
            """
            SELECT id, thread_id, command_json, description, status, pid, pgid,
                   log_dir, timeout_seconds, exit_code, signal_name,
                   started_at, finished_at, created_at, updated_at
            FROM monitored_processes
            WHERE id = ?
            """,
            (monitor_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_monitor(row)

    def count_running(self) -> int:
        """Return the number of monitors in ``running`` status.

        Returns:
            Running monitor count.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM monitored_processes WHERE status = 'running'",
        ).fetchone()
        return int(row["count"])

    def list_running(self) -> list[MonitoredProcess]:
        """Return all running monitors ordered by start time.

        Returns:
            Running monitor rows.
        """
        rows = self._conn.execute(
            """
            SELECT id, thread_id, command_json, description, status, pid, pgid,
                   log_dir, timeout_seconds, exit_code, signal_name,
                   started_at, finished_at, created_at, updated_at
            FROM monitored_processes
            WHERE status = 'running'
            ORDER BY started_at ASC
            """,
        ).fetchall()
        return [self._row_to_monitor(row) for row in rows]

    def list_recent(self, *, thread_id: str, limit: int = 10) -> list[MonitoredProcess]:
        """Return recent monitors for a thread (running first, then by time).

        Args:
            thread_id: Conversation thread id.
            limit: Maximum rows to return.

        Returns:
            Monitor rows for context display.
        """
        rows = self._conn.execute(
            """
            SELECT id, thread_id, command_json, description, status, pid, pgid,
                   log_dir, timeout_seconds, exit_code, signal_name,
                   started_at, finished_at, created_at, updated_at
            FROM monitored_processes
            WHERE thread_id = ?
            ORDER BY
                CASE status WHEN 'running' THEN 0 ELSE 1 END,
                started_at DESC
            LIMIT ?
            """,
            (thread_id, limit),
        ).fetchall()
        return [self._row_to_monitor(row) for row in rows]

    def finalize(
        self,
        monitor_id: str,
        *,
        status: MonitorStatus,
        exit_code: int | None = None,
        signal_name: str | None = None,
    ) -> MonitoredProcess:
        """Mark a monitor terminal.

        Args:
            monitor_id: Monitor identifier.
            status: Terminal status.
            exit_code: Process exit code when available.
            signal_name: Signal name when killed by signal.

        Returns:
            Updated monitor row.

        Raises:
            ValueError: When the monitor is missing or not running.
        """
        monitor = self.get(monitor_id)
        if monitor is None:
            msg = f"Monitor not found: {monitor_id}"
            raise ValueError(msg)
        if monitor.status != "running":
            msg = f"Monitor {monitor_id} is not running ({monitor.status})."
            raise ValueError(msg)

        now = _utc_now()
        self._conn.execute(
            """
            UPDATE monitored_processes
            SET status = ?,
                exit_code = ?,
                signal_name = ?,
                finished_at = ?,
                updated_at = ?,
                pid = NULL,
                pgid = NULL
            WHERE id = ?
            """,
            (status, exit_code, signal_name, now, now, monitor_id),
        )
        updated = self.get(monitor_id)
        if updated is None:
            msg = f"Monitor not found after finalize: {monitor_id}"
            raise RuntimeError(msg)
        return updated

    @staticmethod
    def _row_to_monitor(row: sqlite3.Row) -> MonitoredProcess:
        command_raw = json.loads(row["command_json"])
        command = [str(part) for part in command_raw]
        return MonitoredProcess(
            id=row["id"],
            thread_id=row["thread_id"],
            command=command,
            description=row["description"],
            status=row["status"],
            pid=row["pid"],
            pgid=row["pgid"],
            log_dir=row["log_dir"],
            timeout_seconds=int(row["timeout_seconds"]),
            exit_code=row["exit_code"],
            signal_name=row["signal_name"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
