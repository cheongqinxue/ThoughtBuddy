"""Tests for critical wake failure handling."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from config import Settings
from context.budget import max_prompt_tokens, validate_wake_context_budget
from db.connection import connect
from runtime.critical import (
    CriticalWakeError,
    classify_wake_error,
    is_context_length_error,
    is_llm_outage_error,
)
from scheduling.heartbeat import HeartbeatService
from store.tasks import TaskStore


@pytest.fixture
def db_conn(tmp_path) -> sqlite3.Connection:
    """Provide an initialized SQLite connection."""
    conn = connect(tmp_path / "test.db")
    yield conn
    conn.close()


class TestCriticalClassification:
    """Map provider errors to critical wake failures."""

    def test_detects_context_length_message(self) -> None:
        """OpenAI-style context overflow should classify as critical."""
        exc = RuntimeError(
            "Error code: 400 - maximum context length is 32768 tokens. "
            "prompt contains at least 31745 input tokens",
        )
        assert is_context_length_error(exc) is True
        classified = classify_wake_error(exc)
        assert isinstance(classified, CriticalWakeError)

    def test_transient_errors_are_not_critical(self) -> None:
        """Generic failures should remain retryable."""
        assert classify_wake_error(RuntimeError("network timeout")) is None

    def test_llm_connection_error_is_critical_after_client_retries(self) -> None:
        """Connection errors from the inference client should halt the runtime."""
        import httpx
        from openai import APIConnectionError

        request = httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions")
        exc = APIConnectionError(message="Connection refused", request=request)
        settings = Settings(
            openai_base_url="http://127.0.0.1:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
            llm_max_retries=2,
        )
        assert is_llm_outage_error(exc) is True
        classified = classify_wake_error(exc, settings)
        assert isinstance(classified, CriticalWakeError)
        assert "3 client attempt" in str(classified)
        assert "8000" in str(classified)


@pytest.mark.usefixtures("mock_hf_token_counter")
class TestWakeContextBudget:
    """Pre-invoke budget validation."""

    def test_raises_when_prompt_exceeds_budget(self) -> None:
        """Oversized wake context should fail before graph invocation."""
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
            llm_context_window_tokens=1024,
            llm_max_tokens=100,
        )
        messages = [HumanMessage(content="x" * 4000)]
        with pytest.raises(CriticalWakeError, match="exceeds budget"):
            validate_wake_context_budget(messages, settings)

    def test_allows_context_within_budget(self) -> None:
        """Small contexts should pass validation."""
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
            llm_context_window_tokens=32768,
            llm_max_tokens=1024,
        )
        validate_wake_context_budget([HumanMessage(content="hello")], settings)
        assert max_prompt_tokens(settings) == 31744


class TestHeartbeatCriticalHalt:
    """Heartbeat must not retry critical wake failures."""

    @pytest.mark.asyncio
    async def test_critical_failure_ignores_tasks_and_stops(
        self,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Critical wake errors should terminal-disposition tasks and halt."""
        task = TaskStore(db_conn).enqueue_user_task(message="hello")
        db_conn.commit()

        halted: list[str] = []

        def on_critical(error: CriticalWakeError, *, task_ids: list[str]) -> None:
            halted.append(str(error))
            assert task_ids == [task.id]

        graph = MagicMock()
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
        )
        heartbeat = HeartbeatService(
            db_conn,
            graph,
            model=MagicMock(),
            settings=settings,
            on_critical_failure=on_critical,
        )

        with patch(
            "scheduling.heartbeat.invoke_wake_cycle",
            side_effect=CriticalWakeError("context overflow"),
        ):
            await heartbeat.tick_once()

        updated = TaskStore(db_conn).get(task.id)
        assert updated is not None
        assert updated.status == "ignored"
        assert updated.disposition_reason == "context overflow"
        assert halted == ["context overflow"]
        assert heartbeat._stop.is_set()
