"""Monitored subprocess context block."""

from __future__ import annotations

import json
import sqlite3

from config import Settings
from store.monitors import MonitorStore

MONITORS_PREFIX = "Monitored processes:\n"


def build_monitors_snapshot(conn: sqlite3.Connection, settings: Settings) -> str:
    """Build the monitored-processes context block.

    Args:
        conn: SQLite connection.
        settings: Application settings.

    Returns:
        JSON snapshot of running and recent monitors.
    """
    monitors = MonitorStore(conn).list_recent(
        thread_id=settings.default_thread_id,
        limit=10,
    )
    if not monitors:
        return f"{MONITORS_PREFIX}(none)\n"

    payload = [
        {
            "monitor_id": monitor.id,
            "status": monitor.status,
            "description": monitor.description,
            "command": monitor.command,
            "log_dir": monitor.log_dir,
            "timeout_seconds": monitor.timeout_seconds,
            "exit_code": monitor.exit_code,
            "started_at": monitor.started_at,
            "finished_at": monitor.finished_at,
        }
        for monitor in monitors
    ]
    return f"{MONITORS_PREFIX}{json.dumps(payload, indent=2)}\n"
