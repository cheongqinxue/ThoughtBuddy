"""Tests for wake payload construction."""

from __future__ import annotations

from runtime.wake import build_wake_context, build_wake_message
from store.alerts import Alert
from store.tasks import Task


class TestWakePayload:
    """Wake message formatting."""

    def test_build_wake_message_includes_queue_context(self) -> None:
        """Wake message should include pending tasks and upcoming alerts."""
        pending = [
            Task(
                id="task-1",
                alert_id=None,
                occurrence_at=None,
                source="user",
                status="pending",
                payload={"message": "hello"},
                disposition_reason=None,
                created_at="2026-08-30 06:00:00",
                updated_at="2026-08-30 06:00:00",
                started_at=None,
                completed_at=None,
            ),
        ]
        upcoming = [
            Alert(
                id="alert-1",
                description="Daily review",
                schedule_kind="cron",
                cron_expr="0 9 * * *",
                run_at=None,
                next_fire_at="2026-08-30 09:00:00",
                payload={},
                enabled=True,
                created_at="2026-08-30 06:00:00",
                updated_at="2026-08-30 06:00:00",
            ),
        ]
        context = build_wake_context(
            pending_tasks=pending,
            upcoming_alerts=upcoming,
            queue_stats={"pending": 1},
        )
        message = build_wake_message(context)

        assert '"event": "wake"' in message
        assert "harness_phases" in message
        assert "orient" in message
        assert "task-1" in message
        assert "Daily review" in message
