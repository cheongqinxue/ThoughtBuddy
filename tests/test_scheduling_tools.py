"""Tests for scheduling tools and schedule map."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from config import Settings
from db.connection import connect
from scheduling.schedule_map import build_schedule_map
from store.alerts import AlertStore
from store.tasks import TaskStore
from tools.scheduling import create_tools as create_scheduling_tools
from utils.time import format_utc


@pytest.fixture
def scheduling_settings(tmp_path) -> Settings:
    """Settings pointing at a temporary SQLite database."""
    return Settings(
        openai_base_url="http://localhost:8000/v1",
        openai_api_key="test-key",
        agent_model="test-model",
        log_level="WARNING",
        langsearch_api_key="test-langsearch-key",
        thoughtbuddy_db_path=str(tmp_path / "test.db"),
    )


@pytest.fixture
def tool_map(scheduling_settings: Settings) -> dict[str, object]:
    """Return scheduling tools keyed by name."""
    tools = create_scheduling_tools(scheduling_settings)
    return {tool.name: tool for tool in tools}


class TestScheduleMap:
    """Schedule map builder tests."""

    def test_includes_pending_tasks_and_upcoming_once_alert(
        self,
        scheduling_settings: Settings,
    ) -> None:
        """Schedule map should surface pending tasks and one-shot fires in window."""
        conn = connect(scheduling_settings.thoughtbuddy_db_path)
        soon = format_utc(datetime.now(tz=UTC) + timedelta(minutes=10))
        AlertStore(conn).create_once_alert(description="Soon check", run_at=soon)
        TaskStore(conn).enqueue_user_task(message="hello")
        conn.commit()

        schedule = build_schedule_map(conn, window_seconds=3600)
        conn.close()

        assert len(schedule["pending_tasks"]) == 1
        assert schedule["pending_tasks"][0]["summary"] == "hello"
        assert len(schedule["upcoming_fires"]) == 1
        assert schedule["upcoming_fires"][0]["description"] == "Soon check"
        assert schedule["upcoming_fires"][0]["already_materialized"] is False


class TestSchedulingTools:
    """Agent scheduling tool tests."""

    def test_create_once_alert_tool(self, tool_map: dict[str, object]) -> None:
        """create_alert should persist a one-shot alert."""
        run_at = format_utc(datetime.now(tz=UTC) + timedelta(hours=2))
        result = json.loads(
            tool_map["create_alert"].invoke(
                {
                    "description": "Follow up",
                    "schedule_kind": "once",
                    "run_at": run_at,
                },
            ),
        )

        assert result["schedule_kind"] == "once"
        assert result["next_fire_at"] == run_at

    def test_create_cron_alert_tool(self, tool_map: dict[str, object]) -> None:
        """create_alert should persist a cron alert with computed next fire."""
        result = json.loads(
            tool_map["create_alert"].invoke(
                {
                    "description": "Daily review",
                    "schedule_kind": "cron",
                    "cron_expr": "0 9 * * *",
                },
            ),
        )

        assert result["schedule_kind"] == "cron"
        assert "next_fire_at" in result

    def test_update_alert_disable(self, tool_map: dict[str, object]) -> None:
        """update_alert should disable an existing alert."""
        created = json.loads(
            tool_map["create_alert"].invoke(
                {
                    "description": "Temp",
                    "schedule_kind": "once",
                    "run_at": format_utc(datetime.now(tz=UTC) + timedelta(hours=1)),
                },
            ),
        )
        updated = json.loads(
            tool_map["update_alert"].invoke(
                {"alert_id": created["alert_id"], "enabled": False},
            ),
        )

        assert updated["enabled"] is False

    def test_update_task_complete_and_ignore(
        self,
        tool_map: dict[str, object],
        scheduling_settings: Settings,
    ) -> None:
        """update_task should set terminal statuses."""
        conn = connect(scheduling_settings.thoughtbuddy_db_path)
        task = TaskStore(conn).enqueue_user_task(message="do thing")
        conn.commit()
        conn.close()

        completed = json.loads(
            tool_map["update_task"].invoke(
                {"task_id": task.id, "status": "completed", "disposition_reason": "done"},
            ),
        )
        assert completed["status"] == "completed"

        conn = connect(scheduling_settings.thoughtbuddy_db_path)
        task2 = TaskStore(conn).enqueue_user_task(message="skip me")
        conn.commit()
        conn.close()

        ignored = json.loads(
            tool_map["update_task"].invoke(
                {
                    "task_id": task2.id,
                    "status": "ignored",
                    "disposition_reason": "not now",
                },
            ),
        )
        assert ignored["status"] == "ignored"

    def test_list_alerts_returns_schedule_map(self, tool_map: dict[str, object]) -> None:
        """list_alerts should return the schedule map JSON."""
        tool_map["create_alert"].invoke(
            {
                "description": "Map me",
                "schedule_kind": "once",
                "run_at": format_utc(datetime.now(tz=UTC) + timedelta(minutes=30)),
            },
        )
        schedule = json.loads(tool_map["list_alerts"].invoke({"window_seconds": 3600}))

        assert schedule["window_seconds"] == 3600
        assert "standing_alerts" in schedule
        assert len(schedule["standing_alerts"]) == 1
