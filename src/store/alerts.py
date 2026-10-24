"""Alert registry persistence."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from utils.time import utc_now_str as _utc_now

ScheduleKind = Literal["cron", "once"]


@dataclass(frozen=True)
class Alert:
    """Persisted alert definition."""

    id: str
    description: str
    schedule_kind: ScheduleKind
    cron_expr: str | None
    run_at: str | None
    next_fire_at: str
    payload: dict[str, Any]
    enabled: bool
    created_at: str
    updated_at: str


class AlertStore:
    """CRUD operations for the alert registry."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the store.

        Args:
            conn: SQLite connection.
        """
        self._conn = conn

    def create_cron_alert(
        self,
        *,
        description: str,
        cron_expr: str,
        next_fire_at: str,
        payload: dict[str, Any] | None = None,
        alert_id: str | None = None,
    ) -> Alert:
        """Create a recurring cron alert.

        Args:
            description: Human-readable alert summary.
            cron_expr: Cron expression for recurrence.
            next_fire_at: ISO-8601 UTC timestamp of the next fire time.
            payload: Optional JSON payload copied into materialized tasks.
            alert_id: Optional explicit alert id.

        Returns:
            Created alert row.
        """
        return self._insert_alert(
            alert_id=alert_id or str(uuid.uuid4()),
            description=description,
            schedule_kind="cron",
            cron_expr=cron_expr,
            run_at=None,
            next_fire_at=next_fire_at,
            payload=payload or {},
        )

    def create_once_alert(
        self,
        *,
        description: str,
        run_at: str,
        payload: dict[str, Any] | None = None,
        alert_id: str | None = None,
    ) -> Alert:
        """Create a one-shot alert.

        Args:
            description: Human-readable alert summary.
            run_at: ISO-8601 UTC timestamp when the alert should fire.
            payload: Optional JSON payload copied into materialized tasks.
            alert_id: Optional explicit alert id.

        Returns:
            Created alert row.
        """
        return self._insert_alert(
            alert_id=alert_id or str(uuid.uuid4()),
            description=description,
            schedule_kind="once",
            cron_expr=None,
            run_at=run_at,
            next_fire_at=run_at,
            payload=payload or {},
        )

    def get(self, alert_id: str) -> Alert | None:
        """Return an alert by id.

        Args:
            alert_id: Alert identifier.

        Returns:
            Alert row if found, otherwise None.
        """
        row = self._conn.execute(
            """
            SELECT id, description, schedule_kind, cron_expr, run_at, next_fire_at,
                   payload, enabled, created_at, updated_at
            FROM alerts
            WHERE id = ?
            """,
            (alert_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_alert(row)

    def update_alert(
        self,
        alert_id: str,
        *,
        enabled: bool | None = None,
        description: str | None = None,
        cron_expr: str | None = None,
        run_at: str | None = None,
        next_fire_at: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Alert:
        """Update mutable alert fields.

        Args:
            alert_id: Alert identifier.
            enabled: Optional enabled flag.
            description: Optional new description.
            cron_expr: Optional cron expression for cron alerts.
            run_at: Optional run time for one-shot alerts.
            next_fire_at: Optional next fire override.
            payload: Optional payload replacement.

        Returns:
            Updated alert row.

        Raises:
            ValueError: If the alert does not exist.
        """
        existing = self.get(alert_id)
        if existing is None:
            msg = f"Alert not found: {alert_id}"
            raise ValueError(msg)

        fields: dict[str, Any] = {"updated_at": _utc_now()}
        if enabled is not None:
            fields["enabled"] = 1 if enabled else 0
        if description is not None:
            fields["description"] = description
        if cron_expr is not None:
            fields["cron_expr"] = cron_expr
        if run_at is not None:
            fields["run_at"] = run_at
        if next_fire_at is not None:
            fields["next_fire_at"] = next_fire_at
        if payload is not None:
            fields["payload"] = json.dumps(payload)

        assignments = ", ".join(f"{column} = ?" for column in fields)
        values = list(fields.values()) + [alert_id]
        self._conn.execute(
            f"UPDATE alerts SET {assignments} WHERE id = ?",
            values,
        )
        updated = self.get(alert_id)
        if updated is None:
            msg = f"Alert not found after update: {alert_id}"
            raise ValueError(msg)
        return updated

    def list_enabled(self) -> list[Alert]:
        """Return all enabled alerts ordered by next fire time.

        Returns:
            Enabled alerts.
        """
        rows = self._conn.execute(
            """
            SELECT id, description, schedule_kind, cron_expr, run_at, next_fire_at,
                   payload, enabled, created_at, updated_at
            FROM alerts
            WHERE enabled = 1
            ORDER BY next_fire_at ASC
            """,
        ).fetchall()
        return [self._row_to_alert(row) for row in rows]

    def list_upcoming(self, *, within_seconds: int) -> list[Alert]:
        """Return enabled alerts firing within a time window.

        Args:
            within_seconds: Lookahead window in seconds from now.

        Returns:
            Alerts with next_fire_at inside the window.
        """
        rows = self._conn.execute(
            """
            SELECT id, description, schedule_kind, cron_expr, run_at, next_fire_at,
                   payload, enabled, created_at, updated_at
            FROM alerts
            WHERE enabled = 1
              AND next_fire_at <= datetime('now', ?)
            ORDER BY next_fire_at ASC
            """,
            (f"+{within_seconds} seconds",),
        ).fetchall()
        return [self._row_to_alert(row) for row in rows]

    def get_due_alerts(self) -> list[Alert]:
        """Return enabled alerts whose next fire time is due.

        Returns:
            Alerts ready for materialization.
        """
        rows = self._conn.execute(
            """
            SELECT id, description, schedule_kind, cron_expr, run_at, next_fire_at,
                   payload, enabled, created_at, updated_at
            FROM alerts
            WHERE enabled = 1
              AND next_fire_at <= datetime('now')
            ORDER BY next_fire_at ASC
            """,
        ).fetchall()
        return [self._row_to_alert(row) for row in rows]

    def update_next_fire_at(self, alert_id: str, next_fire_at: str) -> None:
        """Update the next fire timestamp for an alert.

        Args:
            alert_id: Alert identifier.
            next_fire_at: New ISO-8601 UTC timestamp.
        """
        self._conn.execute(
            """
            UPDATE alerts
            SET next_fire_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (next_fire_at, _utc_now(), alert_id),
        )

    def disable(self, alert_id: str) -> None:
        """Disable an alert after it has fired or been cancelled.

        Args:
            alert_id: Alert identifier.
        """
        self._conn.execute(
            """
            UPDATE alerts
            SET enabled = 0, updated_at = ?
            WHERE id = ?
            """,
            (_utc_now(), alert_id),
        )

    def _insert_alert(
        self,
        *,
        alert_id: str,
        description: str,
        schedule_kind: ScheduleKind,
        cron_expr: str | None,
        run_at: str | None,
        next_fire_at: str,
        payload: dict[str, Any],
    ) -> Alert:
        now = _utc_now()
        self._conn.execute(
            """
            INSERT INTO alerts (
                id, description, schedule_kind, cron_expr, run_at, next_fire_at,
                payload, enabled, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                alert_id,
                description,
                schedule_kind,
                cron_expr,
                run_at,
                next_fire_at,
                json.dumps(payload),
                now,
                now,
            ),
        )
        return Alert(
            id=alert_id,
            description=description,
            schedule_kind=schedule_kind,
            cron_expr=cron_expr,
            run_at=run_at,
            next_fire_at=next_fire_at,
            payload=payload,
            enabled=True,
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> Alert:
        return Alert(
            id=row["id"],
            description=row["description"],
            schedule_kind=row["schedule_kind"],
            cron_expr=row["cron_expr"],
            run_at=row["run_at"],
            next_fire_at=row["next_fire_at"],
            payload=json.loads(row["payload"]),
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
