"""Tests for monitored subprocess service."""

from __future__ import annotations

import sqlite3
import sys
import time

import pytest

from config import Settings
from db.connection import connect
from runtime.monitor_service import MonitorService
from store.monitors import MonitorStore
from store.tasks import TaskStore
from store.world import WorldStore


@pytest.fixture
def db_conn(tmp_path) -> sqlite3.Connection:
    """Provide an initialized SQLite connection."""
    conn = connect(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings with scratchpad under tmp_path."""
    scratchpad = tmp_path / "scratchpad"
    scratchpad.mkdir()
    return Settings(
        openai_base_url="http://localhost:8000/v1",
        openai_api_key="test",
        agent_model="test",
        langsearch_api_key="key",
        thoughtbuddy_db_path=str(tmp_path / "test.db"),
        scratchpad_directory=str(scratchpad),
        monitor_max_concurrent=5,
        monitor_default_timeout_seconds=30,
        heartbeat_interval_seconds=1,
    )


class TestMonitorService:
    """Monitor lifecycle."""

    def test_start_and_complete_enqueues_event(
        self,
        db_conn: sqlite3.Connection,
        settings: Settings,
    ) -> None:
        """A short-lived monitor should enqueue an event task and world observation."""
        notified = {"count": 0}

        def notify() -> None:
            notified["count"] += 1

        service = MonitorService(db_conn, settings, notify_wake=notify)
        command = f"{sys.executable} -c \"print('done')\""
        result = service.start(command=command, description="quick echo")
        monitor_id = result["monitor_id"]

        deadline = time.time() + 5
        while time.time() < deadline:
            monitor = MonitorStore(db_conn).get(monitor_id)
            if monitor is not None and monitor.status != "running":
                break
            time.sleep(0.05)

        monitor = MonitorStore(db_conn).get(monitor_id)
        assert monitor is not None
        assert monitor.status == "completed"
        assert notified["count"] == 1

        pending = TaskStore(db_conn).list_pending()
        assert len(pending) == 1
        assert pending[0].source == "event"
        assert pending[0].payload["kind"] == "monitor_completed"

        observations = WorldStore(db_conn).list_recent(settings.default_thread_id, limit=5)
        assert any(obs.source == "monitor" for obs in observations)

    def test_concurrent_limit(self, db_conn: sqlite3.Connection, settings: Settings) -> None:
        """Starting more than max_concurrent monitors should fail."""
        settings = settings.model_copy(update={"monitor_max_concurrent": 1})
        service = MonitorService(db_conn, settings)
        sleep_command = f"{sys.executable} -c \"import time; time.sleep(2)\""
        started = service.start(command=sleep_command, description="sleep")
        with pytest.raises(ValueError, match="concurrent"):
            service.start(command=sleep_command, description="sleep again")

        deadline = time.time() + 5
        while time.time() < deadline:
            monitor = MonitorStore(db_conn).get(started["monitor_id"])
            if monitor is not None and monitor.status != "running":
                break
            time.sleep(0.05)

    def test_reconcile_marks_interrupted(
        self,
        db_conn: sqlite3.Connection,
        settings: Settings,
    ) -> None:
        """Startup reconciliation should finalize orphaned running rows."""
        store = MonitorStore(db_conn)
        db_conn.execute("BEGIN IMMEDIATE")
        store.create(
            thread_id=settings.default_thread_id,
            command=["sleep", "999"],
            description="orphan",
            log_dir=str(settings.scratchpad_directory),
            timeout_seconds=30,
            pid=999999,
            pgid=999999,
            monitor_id="orphan-monitor",
        )
        db_conn.commit()

        service = MonitorService(db_conn, settings)
        reconciled = service.reconcile_on_startup()
        assert reconciled == 1
        monitor = store.get("orphan-monitor")
        assert monitor is not None
        assert monitor.status == "interrupted"
