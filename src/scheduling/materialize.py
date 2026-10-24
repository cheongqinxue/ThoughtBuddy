"""Alert to task materialization."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from croniter import croniter

from scheduling.cron_utils import next_cron_fire
from store.alerts import Alert, AlertStore
from store.tasks import TaskStore
from utils.time import format_utc, parse_utc, utc_now_str


def materialize_due_alerts(conn: sqlite3.Connection) -> int:
    """Materialize all due alert occurrences into the task queue.

    Args:
        conn: SQLite connection participating in the caller's transaction.

    Returns:
        Number of new tasks enqueued.
    """
    alert_store = AlertStore(conn)
    task_store = TaskStore(conn)
    created = 0
    now = parse_utc(utc_now_str())

    for alert in alert_store.get_due_alerts():
        created += _materialize_alert(alert_store, task_store, alert, now)

    return created


def _materialize_alert(
    alert_store: AlertStore,
    task_store: TaskStore,
    alert: Alert,
    now: datetime,
) -> int:
    created = 0

    if alert.schedule_kind == "once":
        occurrence_at = alert.next_fire_at
        task = task_store.enqueue_alert_task(
            alert_id=alert.id,
            occurrence_at=occurrence_at,
            payload={
                "description": alert.description,
                **alert.payload,
            },
        )
        if task is not None:
            created += 1
        alert_store.disable(alert.id)
        return created

    if alert.cron_expr is None:
        msg = f"Cron alert {alert.id} is missing cron_expr"
        raise ValueError(msg)

    next_fire = parse_utc(alert.next_fire_at)
    while next_fire <= now:
        occurrence_at = format_utc(next_fire)
        task = task_store.enqueue_alert_task(
            alert_id=alert.id,
            occurrence_at=occurrence_at,
            payload={
                "description": alert.description,
                **alert.payload,
            },
        )
        if task is not None:
            created += 1
        next_fire = next_cron_fire(alert.cron_expr, next_fire)
        alert_store.update_next_fire_at(alert.id, format_utc(next_fire))

    return created
