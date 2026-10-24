"""Tests for event task ingress and heartbeat notify."""

from __future__ import annotations

import asyncio
import sqlite3
from unittest.mock import MagicMock

import pytest

from config import Settings
from db.connection import connect
from scheduling.heartbeat import HeartbeatService
from store.tasks import TaskStore


@pytest.fixture
def db_conn(tmp_path) -> sqlite3.Connection:
    """Provide an initialized SQLite connection."""
    conn = connect(tmp_path / "test.db")
    yield conn
    conn.close()


class TestEventTasks:
    """Event task enqueue and deduplication."""

    def test_enqueue_event_task(self, db_conn: sqlite3.Connection) -> None:
        """Event tasks should persist with source=event."""
        task = TaskStore(db_conn).enqueue_event_task(
            message="Monitor finished",
            event_key="monitor:abc",
            occurrence_at="2026-01-01T00:00:00+00:00",
            payload={"kind": "monitor_completed"},
        )
        db_conn.commit()

        assert task is not None
        assert task.source == "event"
        assert task.payload["kind"] == "monitor_completed"

    def test_duplicate_event_is_idempotent(self, db_conn: sqlite3.Connection) -> None:
        """Duplicate event_key + occurrence_at should dedupe."""
        store = TaskStore(db_conn)
        first = store.enqueue_event_task(
            message="once",
            event_key="evt:1",
            occurrence_at="2026-01-01T00:00:00+00:00",
        )
        second = store.enqueue_event_task(
            message="once",
            event_key="evt:1",
            occurrence_at="2026-01-01T00:00:00+00:00",
        )
        db_conn.commit()

        assert first is not None
        assert second is None
        assert TaskStore(db_conn).count_by_status().get("pending", 0) == 1


class TestHeartbeatNotify:
    """Immediate wake signaling."""

    @pytest.mark.asyncio
    async def test_notify_wakes_before_interval(self, db_conn: sqlite3.Connection) -> None:
        """notify() should unblock the heartbeat wait early."""
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
            heartbeat_interval_seconds=60,
        )
        heartbeat = HeartbeatService(
            db_conn,
            MagicMock(),
            model=MagicMock(),
            settings=settings,
        )
        await heartbeat.start()

        async def notify_soon() -> None:
            await asyncio.sleep(0.05)
            heartbeat.notify()

        notify_task = asyncio.create_task(notify_soon())
        try:
            await asyncio.wait_for(heartbeat._wait_for_stop_or_notify(), timeout=2)
        finally:
            await notify_task
            await heartbeat.stop()
