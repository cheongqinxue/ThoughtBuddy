"""Tests for ThoughtBuddy scheduling and persistence."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from config import Settings
from db.connection import connect
from scheduling.lock import InstanceLock, InstanceLockError
from scheduling.materialize import materialize_due_alerts
from store.alerts import AlertStore
from store.tasks import TaskStore
from utils.time import format_utc


@pytest.fixture
def db_conn(tmp_path) -> sqlite3.Connection:
    """Provide an initialized SQLite connection."""
    conn = connect(tmp_path / "test.db")
    yield conn
    conn.close()


class TestInstanceLock:
    """PID lock behavior."""

    def test_second_acquire_raises(self, tmp_path) -> None:
        """A second lock on the same path should fail."""
        path = tmp_path / "thoughtbuddy.lock"
        first = InstanceLock(path)
        second = InstanceLock(path)
        first.acquire()
        try:
            with pytest.raises(InstanceLockError):
                second.acquire()
        finally:
            first.release()


class TestMaterializeAlerts:
    """Alert to task materialization."""

    def test_once_alert_materializes_and_disables(self, db_conn: sqlite3.Connection) -> None:
        """A due one-shot alert should enqueue one task and disable itself."""
        past = format_utc(datetime.now(tz=UTC) - timedelta(minutes=5))
        alert_store = AlertStore(db_conn)
        alert = alert_store.create_once_alert(
            description="Check scrape",
            run_at=past,
            alert_id="alert-once",
        )
        db_conn.commit()

        created = materialize_due_alerts(db_conn)
        db_conn.commit()

        assert created == 1
        tasks = TaskStore(db_conn).list_pending()
        assert len(tasks) == 1
        assert tasks[0].alert_id == alert.id
        remaining = alert_store.list_enabled()
        assert remaining == []

    def test_duplicate_occurrence_is_idempotent(self, db_conn: sqlite3.Connection) -> None:
        """Materializing the same occurrence twice should not duplicate tasks."""
        past = format_utc(datetime.now(tz=UTC) - timedelta(minutes=1))
        AlertStore(db_conn).create_once_alert(
            description="Ping",
            run_at=past,
            alert_id="alert-dup",
        )
        db_conn.commit()

        assert materialize_due_alerts(db_conn) == 1
        db_conn.commit()
        assert materialize_due_alerts(db_conn) == 0
        db_conn.commit()

        assert TaskStore(db_conn).count_by_status().get("pending", 0) == 1


@pytest.mark.usefixtures("mock_hf_token_counter")
class TestHeartbeatWake:
    """Heartbeat wake gating."""

    @pytest.mark.asyncio
    async def test_tick_wakes_agent_when_pending_tasks(self, db_conn: sqlite3.Connection) -> None:
        """A pending task should trigger a wake cycle."""
        from scheduling.heartbeat import HeartbeatService

        TaskStore(db_conn).enqueue_user_task(message="hello")
        db_conn.commit()

        graph = MagicMock()
        graph.invoke.return_value = {"messages": []}
        model = MagicMock()
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
        )
        heartbeat = HeartbeatService(
            db_conn,
            graph,
            model=model,
            settings=settings,
        )

        await heartbeat.tick_once()

        graph.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_wake_failure_reverts_running_tasks(self, db_conn: sqlite3.Connection) -> None:
        """Failed wake cycles should not leave tasks stuck in running."""
        from scheduling.heartbeat import HeartbeatService

        task = TaskStore(db_conn).enqueue_user_task(message="hello")
        db_conn.commit()

        graph = MagicMock()
        graph.invoke.side_effect = RuntimeError("graph failed")
        model = MagicMock()
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
        )
        heartbeat = HeartbeatService(
            db_conn,
            graph,
            model=model,
            settings=settings,
        )

        await heartbeat.tick_once()

        updated = TaskStore(db_conn).get(task.id)
        assert updated is not None
        assert updated.status == "pending"

    @pytest.mark.asyncio
    async def test_wake_success_reverts_undischarged_tasks(
        self, db_conn: sqlite3.Connection
    ) -> None:
        """Successful wakes must not leave tasks stuck in running."""
        from scheduling.heartbeat import HeartbeatService

        task = TaskStore(db_conn).enqueue_user_task(message="hello")
        db_conn.commit()

        graph = MagicMock()
        graph.invoke.return_value = {"messages": []}
        model = MagicMock()
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
        )
        heartbeat = HeartbeatService(
            db_conn,
            graph,
            model=model,
            settings=settings,
        )

        await heartbeat.tick_once()

        updated = TaskStore(db_conn).get(task.id)
        assert updated is not None
        assert updated.status == "pending"


class TestTaskStoreRevert:
    """Task status recovery helpers."""

    def test_revert_running_to_pending(self, db_conn: sqlite3.Connection) -> None:
        """Reset running tasks back to pending."""
        task = TaskStore(db_conn).enqueue_user_task(message="hello")
        TaskStore(db_conn).mark_running([task.id])
        db_conn.commit()

        reverted = TaskStore(db_conn).revert_running_to_pending([task.id])
        db_conn.commit()

        updated = TaskStore(db_conn).get(task.id)
        assert reverted == 1
        assert updated is not None
        assert updated.status == "pending"
        assert updated.started_at is None
