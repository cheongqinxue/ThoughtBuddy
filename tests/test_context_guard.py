"""Tests for pre-LLM context budget guards."""

from __future__ import annotations

import sqlite3

import pytest
from langchain_core.messages import HumanMessage

from config import Settings
from context.guard import ensure_llm_prompt_budget, ensure_prompt_budget
from context.limits import compaction_enabled, compaction_token_threshold, effective_compact_ratio
from context.wake_guard import WakeGuardContext, wake_guard_context
from db.connection import connect
from llm.budget_guard import BudgetGuardChatModel, wrap_with_budget_guard
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from runtime.critical import CriticalWakeError
from store.messages import MessageLogStore


@pytest.fixture
def db_conn(tmp_path) -> sqlite3.Connection:
    """Provide an initialized SQLite connection."""
    conn = connect(tmp_path / "guard.db")
    yield conn
    conn.close()


@pytest.mark.usefixtures("mock_hf_token_counter")
class TestCompactionThreshold:
    """Compaction margin configuration."""

    def test_compaction_is_always_enabled(self) -> None:
        """Archive compaction cannot be turned off via config."""
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
            context_compact_token_threshold=0,
            context_compact_token_ratio=0,
        )
        assert compaction_enabled(settings) is True
        assert effective_compact_ratio(settings) == 0.9

    def test_explicit_threshold_overrides_ratio(self) -> None:
        """Configured token threshold takes precedence over ratio."""
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
            llm_context_window_tokens=2048,
            llm_max_tokens=100,
            context_compact_token_threshold=700,
            context_compact_token_ratio=0.5,
        )
        assert compaction_token_threshold(settings) == 700

    def test_ratio_default_is_ninety_percent_of_prompt_budget(self) -> None:
        """When threshold is unset, compaction uses ratio * max_prompt_tokens."""
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
            llm_context_window_tokens=2048,
            llm_max_tokens=100,
            context_compact_token_threshold=0,
            context_compact_token_ratio=0.9,
        )
        assert compaction_token_threshold(settings) == 1753


@pytest.mark.usefixtures("mock_hf_token_counter")
class TestEnsurePromptBudget:
    """Hard budget validation."""

    def test_raises_when_prompt_exceeds_budget(self) -> None:
        """Oversized prompts fail before reaching the inference server."""
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
            llm_context_window_tokens=1024,
            llm_max_tokens=100,
        )
        with pytest.raises(CriticalWakeError, match="exceeds budget"):
            ensure_prompt_budget([HumanMessage(content="x" * 4000)], settings)


@pytest.mark.usefixtures("mock_hf_token_counter")
class TestBudgetGuardModel:
    """Budget guard wrapper on chat models."""

    def test_wrapper_checks_budget_before_generate(self) -> None:
        """Wrapped models reject oversized prompts."""
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
            llm_context_window_tokens=1024,
            llm_max_tokens=100,
        )
        inner = FakeListChatModel(responses=["ok"])
        model = wrap_with_budget_guard(inner, settings)
        assert isinstance(model, BudgetGuardChatModel)

        with pytest.raises(CriticalWakeError, match="exceeds budget"):
            model.invoke([HumanMessage(content="x" * 4000)])

    def test_wake_guard_enables_archive_compaction_path(
        self,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Wake guard supplies archive context for pre-invoke compaction checks."""
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
            llm_context_window_tokens=32768,
            llm_max_tokens=1024,
            context_compact_token_threshold=50,
            context_retain_messages=2,
        )
        log = MessageLogStore(db_conn)
        log.append("main", HumanMessage(content="small"))
        db_conn.commit()

        guard = WakeGuardContext(
            conn=db_conn,
            settings=settings,
            thread_id="main",
            compaction_model=FakeListChatModel(responses=["summary"]),
            pending_tasks=[],
            tools=[],
        )
        with wake_guard_context(guard):
            ensure_llm_prompt_budget([HumanMessage(content="hello")], settings)
