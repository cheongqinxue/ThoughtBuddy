"""Task queue persistence."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from utils.time import utc_now_str as _utc_now

TaskSource = Literal["alert", "user", "event"]
TaskStatus = Literal["pending", "running", "completed", "ignored"]
TerminalTaskStatus = Literal["completed", "ignored"]


@dataclass(frozen=True)
class Task:
    """Persisted task row."""

    id: str
    alert_id: str | None
    occurrence_at: str | None
    source: TaskSource
    status: TaskStatus
    payload: dict[str, Any]
    disposition_reason: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None


class TaskStore:
    """CRUD operations for the task queue."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the store.

        Args:
            conn: SQLite connection.
        """
        self._conn = conn

    def enqueue_alert_task(
        self,
        *,
        alert_id: str,
        occurrence_at: str,
        payload: dict[str, Any],
        task_id: str | None = None,
    ) -> Task | None:
        """Enqueue a task materialized from an alert occurrence.

        Args:
            alert_id: Source alert identifier.
            occurrence_at: ISO-8601 occurrence timestamp used for deduplication.
            payload: Task payload copied from the alert.
            task_id: Optional explicit task id.

        Returns:
            Created task, or None if the occurrence already exists.
        """
        now = _utc_now()
        task_id = task_id or str(uuid.uuid4())
        try:
            self._conn.execute(
                """
                INSERT INTO tasks (
                    id, alert_id, occurrence_at, source, status, payload,
                    disposition_reason, created_at, updated_at, started_at, completed_at
                ) VALUES (?, ?, ?, 'alert', 'pending', ?, NULL, ?, ?, NULL, NULL)
                """,
                (task_id, alert_id, occurrence_at, json.dumps(payload), now, now),
            )
        except sqlite3.IntegrityError:
            return None
        return Task(
            id=task_id,
            alert_id=alert_id,
            occurrence_at=occurrence_at,
            source="alert",
            status="pending",
            payload=payload,
            disposition_reason=None,
            created_at=now,
            updated_at=now,
            started_at=None,
            completed_at=None,
        )

    def enqueue_event_task(
        self,
        *,
        message: str,
        event_key: str,
        occurrence_at: str,
        payload: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> Task | None:
        """Enqueue a push/event task (deduped by event key + occurrence).

        Args:
            message: Human-readable wake instruction.
            event_key: Stable event identifier for deduplication.
            occurrence_at: ISO-8601 occurrence timestamp (dedup pair with event_key).
            payload: Optional structured payload fields.
            task_id: Optional explicit task id.

        Returns:
            Created task, or None if the occurrence already exists.
        """
        now = _utc_now()
        task_id = task_id or str(uuid.uuid4())
        body = {"message": message, **(payload or {})}
        try:
            self._conn.execute(
                """
                INSERT INTO tasks (
                    id, alert_id, occurrence_at, event_key, source, status, payload,
                    disposition_reason, created_at, updated_at, started_at, completed_at
                ) VALUES (?, NULL, ?, ?, 'event', 'pending', ?, NULL, ?, ?, NULL, NULL)
                """,
                (task_id, occurrence_at, event_key, json.dumps(body), now, now),
            )
        except sqlite3.IntegrityError:
            return None
        return Task(
            id=task_id,
            alert_id=None,
            occurrence_at=occurrence_at,
            source="event",
            status="pending",
            payload=body,
            disposition_reason=None,
            created_at=now,
            updated_at=now,
            started_at=None,
            completed_at=None,
        )

    def enqueue_user_task(
        self,
        *,
        message: str,
        payload: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> Task:
        """Enqueue a user-initiated task.

        Args:
            message: User message text.
            payload: Optional extra payload fields.
            task_id: Optional explicit task id.

        Returns:
            Created task row.
        """
        now = _utc_now()
        task_id = task_id or str(uuid.uuid4())
        body = {"message": message, **(payload or {})}
        self._conn.execute(
            """
            INSERT INTO tasks (
                id, alert_id, occurrence_at, source, status, payload,
                disposition_reason, created_at, updated_at, started_at, completed_at
            ) VALUES (?, NULL, NULL, 'user', 'pending', ?, NULL, ?, ?, NULL, NULL)
            """,
            (task_id, json.dumps(body), now, now),
        )
        return Task(
            id=task_id,
            alert_id=None,
            occurrence_at=None,
            source="user",
            status="pending",
            payload=body,
            disposition_reason=None,
            created_at=now,
            updated_at=now,
            started_at=None,
            completed_at=None,
        )

    def get(self, task_id: str) -> Task | None:
        """Return a task by id.

        Args:
            task_id: Task identifier.

        Returns:
            Task row if found, otherwise None.
        """
        row = self._conn.execute(
            """
            SELECT id, alert_id, occurrence_at, source, status, payload,
                   disposition_reason, created_at, updated_at, started_at, completed_at
            FROM tasks
            WHERE id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def update_status(
        self,
        task_id: str,
        *,
        status: TerminalTaskStatus,
        disposition_reason: str | None = None,
    ) -> Task:
        """Set a terminal task status.

        Args:
            task_id: Task identifier.
            status: Terminal status (`completed` or `ignored`).
            disposition_reason: Optional explanation for the disposition.

        Returns:
            Updated task row.

        Raises:
            ValueError: If the task is missing or not updatable.
        """
        task = self.get(task_id)
        if task is None:
            msg = f"Task not found: {task_id}"
            raise ValueError(msg)
        if task.status not in {"pending", "running"}:
            msg = f"Task {task_id} is already terminal ({task.status})."
            raise ValueError(msg)

        now = _utc_now()
        self._conn.execute(
            """
            UPDATE tasks
            SET status = ?,
                disposition_reason = ?,
                updated_at = ?,
                completed_at = ?
            WHERE id = ?
            """,
            (status, disposition_reason, now, now, task_id),
        )
        updated = self.get(task_id)
        if updated is None:
            msg = f"Task not found after update: {task_id}"
            raise ValueError(msg)
        return updated

    def list_pending(self) -> list[Task]:
        """Return pending tasks oldest first.

        Returns:
            Pending tasks ordered by creation time.
        """
        rows = self._conn.execute(
            """
            SELECT id, alert_id, occurrence_at, source, status, payload,
                   disposition_reason, created_at, updated_at, started_at, completed_at
            FROM tasks
            WHERE status = 'pending'
            ORDER BY created_at ASC
            """,
        ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def mark_running(self, task_ids: list[str]) -> None:
        """Mark tasks as running for the current wake cycle.

        Args:
            task_ids: Task identifiers to update.
        """
        if not task_ids:
            return
        now = _utc_now()
        placeholders = ",".join("?" for _ in task_ids)
        self._conn.execute(
            f"""
            UPDATE tasks
            SET status = 'running', started_at = ?, updated_at = ?
            WHERE id IN ({placeholders}) AND status = 'pending'
            """,
            (now, now, *task_ids),
        )

    def revert_running_to_pending(self, task_ids: list[str]) -> int:
        """Reset running tasks back to pending after a failed wake cycle.

        Args:
            task_ids: Task identifiers to revert.

        Returns:
            Number of tasks updated.
        """
        if not task_ids:
            return 0
        now = _utc_now()
        placeholders = ",".join("?" for _ in task_ids)
        cursor = self._conn.execute(
            f"""
            UPDATE tasks
            SET status = 'pending',
                started_at = NULL,
                updated_at = ?
            WHERE id IN ({placeholders}) AND status = 'running'
            """,
            (now, *task_ids),
        )
        return cursor.rowcount

    def count_by_status(self) -> dict[str, int]:
        """Return task counts grouped by status.

        Returns:
            Mapping of status to count.
        """
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status",
        ).fetchall()
        return {row["status"]: row["count"] for row in rows}

    def reap_stale_running(self, *, stale_after_seconds: int) -> int:
        """Reset long-running tasks back to pending.

        Args:
            stale_after_seconds: Age threshold for stale running tasks.

        Returns:
            Number of tasks reset.
        """
        now = _utc_now()
        cursor = self._conn.execute(
            """
            UPDATE tasks
            SET status = 'pending',
                updated_at = ?,
                started_at = NULL
            WHERE status = 'running'
              AND started_at <= datetime('now', ?)
            """,
            (now, f"-{stale_after_seconds} seconds"),
        )
        return cursor.rowcount

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            alert_id=row["alert_id"],
            occurrence_at=row["occurrence_at"],
            source=row["source"],
            status=row["status"],
            payload=json.loads(row["payload"]),
            disposition_reason=row["disposition_reason"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )
