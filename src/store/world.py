"""SQLite store for world observations and virtual state."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

from utils.time import utc_now_str


@dataclass(frozen=True)
class WorldObservation:
    """One timestamped observation of the world."""

    id: str
    thread_id: str
    source: str
    payload: dict[str, Any]
    summary: str | None
    created_at: str


class WorldStore:
    """Persist external and virtual world observations."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the store.

        Args:
            conn: SQLite connection.
        """
        self._conn = conn

    def append_observation(
        self,
        thread_id: str,
        *,
        source: str,
        payload: dict[str, Any],
        summary: str | None = None,
        observation_id: str | None = None,
    ) -> WorldObservation:
        """Append an observation record.

        Args:
            thread_id: Conversation thread identifier.
            source: Observation channel (e.g. websearch, virtual).
            payload: Raw observation payload.
            summary: Optional short summary.
            observation_id: Optional stable id.

        Returns:
            Persisted observation.
        """
        observation = WorldObservation(
            id=observation_id or str(uuid.uuid4()),
            thread_id=thread_id,
            source=source,
            payload=payload,
            summary=summary,
            created_at=utc_now_str(),
        )
        self._conn.execute(
            """
            INSERT INTO world_observations (id, thread_id, source, payload, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                observation.id,
                observation.thread_id,
                observation.source,
                json.dumps(observation.payload),
                observation.summary,
                observation.created_at,
            ),
        )
        return observation

    def list_recent(self, thread_id: str, *, limit: int = 10) -> list[WorldObservation]:
        """Return recent observations for a thread.

        Args:
            thread_id: Conversation thread identifier.
            limit: Maximum observations to return.

        Returns:
            Observations newest-first.
        """
        rows = self._conn.execute(
            """
            SELECT id, thread_id, source, payload, summary, created_at
            FROM world_observations
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (thread_id, limit),
        ).fetchall()
        return [self._row_to_observation(row) for row in rows]

    def get_state(self, thread_id: str, state_key: str) -> str | None:
        """Read one virtual-world state value.

        Args:
            thread_id: Conversation thread identifier.
            state_key: State key (e.g. chess.fen).

        Returns:
            Stored value, or ``None`` when missing.
        """
        row = self._conn.execute(
            """
            SELECT value FROM world_state
            WHERE thread_id = ? AND state_key = ?
            """,
            (thread_id, state_key),
        ).fetchone()
        return row[0] if row else None

    def set_state(self, thread_id: str, state_key: str, value: str) -> None:
        """Upsert one virtual-world state value.

        Args:
            thread_id: Conversation thread identifier.
            state_key: State key.
            value: Serialized state value.
        """
        self._conn.execute(
            """
            INSERT INTO world_state (thread_id, state_key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id, state_key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (thread_id, state_key, value, utc_now_str()),
        )

    def list_state(self, thread_id: str) -> dict[str, str]:
        """Return all virtual-world state keys for a thread.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            Mapping of state key to value.
        """
        rows = self._conn.execute(
            """
            SELECT state_key, value FROM world_state
            WHERE thread_id = ?
            ORDER BY state_key
            """,
            (thread_id,),
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    @staticmethod
    def _row_to_observation(row: tuple[Any, ...]) -> WorldObservation:
        return WorldObservation(
            id=row[0],
            thread_id=row[1],
            source=row[2],
            payload=json.loads(row[3]),
            summary=row[4],
            created_at=row[5],
        )
