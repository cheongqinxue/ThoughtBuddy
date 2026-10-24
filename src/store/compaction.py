"""Compaction point persistence."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass

from utils.time import utc_now_str


@dataclass(frozen=True)
class CompactionPoint:
    """Summary boundary for a thread."""

    id: str
    thread_id: str
    boundary_seq: int
    retain_count: int
    summary: str
    created_at: str


class CompactionStore:
    """Store compaction summaries and boundaries."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the store.

        Args:
            conn: SQLite connection.
        """
        self._conn = conn

    def get_latest(self, thread_id: str) -> CompactionPoint | None:
        """Return the most recent compaction for a thread.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            Latest compaction point, if any.
        """
        row = self._conn.execute(
            """
            SELECT id, thread_id, boundary_seq, retain_count, summary, created_at
            FROM compaction_points
            WHERE thread_id = ?
            ORDER BY created_at DESC, boundary_seq DESC
            LIMIT 1
            """,
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_point(row)

    def create(
        self,
        thread_id: str,
        *,
        boundary_seq: int,
        retain_count: int,
        summary: str,
        point_id: str | None = None,
    ) -> CompactionPoint:
        """Record a new compaction point.

        Args:
            thread_id: Conversation thread identifier.
            boundary_seq: Last sequence included in the summary window.
            retain_count: N verbatim messages kept after the boundary.
            summary: Compressed summary text.
            point_id: Optional explicit id.

        Returns:
            Created compaction point.
        """
        now = utc_now_str()
        point_id = point_id or str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO compaction_points (
                id, thread_id, boundary_seq, retain_count, summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (point_id, thread_id, boundary_seq, retain_count, summary, now),
        )
        return CompactionPoint(
            id=point_id,
            thread_id=thread_id,
            boundary_seq=boundary_seq,
            retain_count=retain_count,
            summary=summary,
            created_at=now,
        )

    @staticmethod
    def _row_to_point(row: sqlite3.Row) -> CompactionPoint:
        return CompactionPoint(
            id=row["id"],
            thread_id=row["thread_id"],
            boundary_seq=row["boundary_seq"],
            retain_count=row["retain_count"],
            summary=row["summary"],
            created_at=row["created_at"],
        )
