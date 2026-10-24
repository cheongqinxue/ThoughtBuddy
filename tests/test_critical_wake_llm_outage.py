"""Heartbeat behavior when LLM inference fails after retries."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError

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


@pytest.mark.usefixtures("mock_hf_token_counter")
class TestHeartbeatLlmOutage:
    """LLM outage after client retries should halt, not loop."""

    @pytest.mark.asyncio
    async def test_connection_error_halts_and_ignores_tasks(
        self,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Exhausted LLM retries should terminal-disposition tasks and stop heartbeat."""
        task = TaskStore(db_conn).enqueue_user_task(message="hello")
        db_conn.commit()

        request = httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions")
        outage = APIConnectionError(message="Connection refused", request=request)

        halted: list[str] = []

        def on_critical(error, *, task_ids: list[str]) -> None:
            halted.append(str(error))

        settings = Settings(
            openai_base_url="http://127.0.0.1:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
            llm_max_retries=2,
            active_experiment="chess",
        )
        heartbeat = HeartbeatService(
            db_conn,
            MagicMock(),
            model=MagicMock(),
            settings=settings,
            on_critical_failure=on_critical,
        )

        with patch(
            "scheduling.heartbeat.invoke_wake_cycle",
            side_effect=outage,
        ):
            await heartbeat.tick_once()

        updated = TaskStore(db_conn).get(task.id)
        assert updated is not None
        assert updated.status == "ignored"
        assert "LLM inference unavailable" in (updated.disposition_reason or "")
        assert halted
        assert heartbeat._stop.is_set()
