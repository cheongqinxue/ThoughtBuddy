"""Build schedule-map views for agent planning."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from croniter import croniter

from scheduling.cron_utils import next_cron_fire
from store.alerts import Alert, AlertStore
from store.tasks import Task, TaskStore
from utils.time import format_utc, parse_utc, utc_now_str


def build_schedule_map(
    conn: sqlite3.Connection,
    *,
    window_seconds: int = 3600,
) -> dict[str, Any]:
    """Build a timeline-oriented schedule map for the agent.

    Args:
        conn: SQLite connection.
        window_seconds: Lookahead window from now for upcoming fires.

    Returns:
        Schedule map with pending tasks, upcoming fires, and standing alerts.
    """
    as_of = utc_now_str()
    window_start = parse_utc(as_of)
    window_end = window_start + timedelta(seconds=window_seconds)

    alert_store = AlertStore(conn)
    task_store = TaskStore(conn)
    standing_alerts = alert_store.list_enabled()
    pending_tasks = task_store.list_pending()

    return {
        "as_of": as_of,
        "window_seconds": window_seconds,
        "pending_tasks": [_pending_task_entry(task) for task in pending_tasks],
        "upcoming_fires": _collect_upcoming_fires(
            conn,
            standing_alerts,
            window_start=window_start,
            window_end=window_end,
        ),
        "standing_alerts": [_standing_alert_entry(alert) for alert in standing_alerts],
    }


def _pending_task_entry(task: Task) -> dict[str, Any]:
    summary = task.payload.get("description") or task.payload.get("message") or task.id
    return {
        "task_id": task.id,
        "source": task.source,
        "created_at": task.created_at,
        "alert_id": task.alert_id,
        "summary": summary,
    }


def _standing_alert_entry(alert: Alert) -> dict[str, Any]:
    return {
        "alert_id": alert.id,
        "description": alert.description,
        "schedule_kind": alert.schedule_kind,
        "cron_expr": alert.cron_expr,
        "run_at": alert.run_at,
        "next_fire_at": alert.next_fire_at,
        "enabled": alert.enabled,
    }


def _collect_upcoming_fires(
    conn: sqlite3.Connection,
    alerts: list[Alert],
    *,
    window_start: datetime,
    window_end: datetime,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for alert in alerts:
        for fires_at in _fires_in_window(alert, window_start, window_end):
            entries.append(
                {
                    "alert_id": alert.id,
                    "description": alert.description,
                    "fires_at": fires_at,
                    "schedule_kind": alert.schedule_kind,
                    "already_materialized": _is_materialized(conn, alert.id, fires_at),
                },
            )
    entries.sort(key=lambda item: item["fires_at"])
    return entries


def _fires_in_window(
    alert: Alert,
    window_start: datetime,
    window_end: datetime,
) -> list[str]:
    if alert.schedule_kind == "once":
        fire = parse_utc(alert.next_fire_at)
        if window_start <= fire <= window_end:
            return [alert.next_fire_at]
        return []

    if alert.cron_expr is None:
        return []

    fires: list[str] = []
    cursor = parse_utc(alert.next_fire_at)
    safety = 0
    while cursor <= window_end and safety < 500:
        if cursor >= window_start:
            fires.append(format_utc(cursor))
        cursor = next_cron_fire(alert.cron_expr, cursor)
        safety += 1
    return fires


def _is_materialized(conn: sqlite3.Connection, alert_id: str, occurrence_at: str) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM tasks
        WHERE alert_id = ? AND occurrence_at = ?
        LIMIT 1
        """,
        (alert_id, occurrence_at),
    ).fetchone()
    return row is not None


def compute_initial_next_fire(cron_expr: str, *, base: datetime | None = None) -> str:
    """Compute the first next fire time for a new cron alert.

    Args:
        cron_expr: Cron expression.
        base: Optional base time; defaults to now UTC.

    Returns:
        Stored UTC timestamp string for the next fire.
    """
    start = base or datetime.now(tz=UTC).replace(microsecond=0)
    if not croniter.is_valid(cron_expr):
        msg = f"Invalid cron expression: {cron_expr}"
        raise ValueError(msg)
    nxt = croniter(cron_expr, start).get_next(datetime)
    if nxt.tzinfo is None:
        nxt = nxt.replace(tzinfo=UTC)
    return format_utc(nxt.astimezone(UTC))
