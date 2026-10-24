"""Tests for message archive and context assembly."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from config import Settings
from context.build import (
    CHECKPOINT_PREFIX,
    GOALS_PREFIX,
    PROGRESS_PREFIX,
    SCRATCHPAD_PREFIX,
    build_agent_context,
    build_model_context,
    build_transcript_messages,
)
from context.pillars import (
    KNOWLEDGE_PREFIX,
    PLAN_PREFIX,
    SYSTEM_PREFIX,
    WORLD_PREFIX,
)
from context.compact import CompactionError, load_compaction_prompt, maybe_compact_context
from db.connection import connect
from runtime.scratchpad_layout import ensure_scratchpad_layout
from store.compaction import CompactionStore
from store.messages import MessageLogStore


@pytest.fixture
def db_conn(tmp_path) -> sqlite3.Connection:
    """Provide an initialized SQLite connection."""
    conn = connect(tmp_path / "messages.db")
    yield conn
    conn.close()


@pytest.fixture
def context_settings(tmp_path) -> Settings:
    """Settings with scratchpad directory and compaction enabled."""
    scratchpad_dir = tmp_path / "scratchpad"
    ensure_scratchpad_layout(scratchpad_dir, template_dir=Path(".data-template/scratchpad"))
    return Settings(
        openai_base_url="http://localhost:8000/v1",
        openai_api_key="test",
        agent_model="test",
        langsearch_api_key="key",
        scratchpad_directory=str(scratchpad_dir),
        context_retain_messages=5,
        context_compact_token_threshold=100,
    )


@pytest.fixture
def sample_tools() -> list[StructuredTool]:
    """Minimal tool list for context assembly tests."""

    def noop() -> str:
        """Example tool."""
        return "ok"

    return [
        StructuredTool.from_function(
            func=noop,
            name="example_tool",
            description="An example tool.",
        ),
    ]


@pytest.mark.usefixtures("mock_hf_token_counter")
class TestMessageArchive:
    """Append-only archive tests."""

    def test_build_transcript_without_compaction(self, db_conn: sqlite3.Connection) -> None:
        """All archived messages appear in transcript when uncompacted."""
        log = MessageLogStore(db_conn)
        log.append("main", HumanMessage(content="first"))
        log.append("main", AIMessage(content="second"))
        db_conn.commit()

        context = build_transcript_messages(db_conn, "main")

        assert len(context) == 2
        assert context[0].content == "first"
        assert context[1].content == "second"

    def test_compaction_keeps_checkpoint_and_post_boundary_only(
        self,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Transcript shows checkpoint plus only messages after the boundary."""
        log = MessageLogStore(db_conn)
        for index in range(1, 6):
            log.append("main", HumanMessage(content=f"msg-{index}"))
        CompactionStore(db_conn).create(
            "main",
            boundary_seq=3,
            retain_count=2,
            summary="Earlier discussion about goals.",
        )
        db_conn.commit()

        context = build_transcript_messages(db_conn, "main")

        assert context[0].content.startswith(CHECKPOINT_PREFIX)
        assert "Earlier discussion" in str(context[0].content)
        assert [message.content for message in context[1:]] == ["msg-4", "msg-5"]
        assert log.count("main") == 5

    def test_maybe_compact_creates_compaction_point(self, db_conn: sqlite3.Connection) -> None:
        """Compaction runs when token threshold is exceeded."""
        log = MessageLogStore(db_conn)
        for index in range(30):
            log.append("main", HumanMessage(content=f"message number {index} " * 20))
        db_conn.commit()

        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test",
            langsearch_api_key="key",
            context_retain_messages=5,
            context_compact_token_threshold=100,
        )
        model = FakeListChatModel(responses=["Summary of prior work."])

        compacted = maybe_compact_context(db_conn, settings, model, "main")
        db_conn.commit()

        assert compacted is True
        point = CompactionStore(db_conn).get_latest("main")
        assert point is not None
        assert point.summary == "Summary of prior work."
        assert log.count("main") == 30

        context = build_model_context(db_conn, "main")
        assert context[0].content.startswith(CHECKPOINT_PREFIX)

    def test_maybe_compact_uses_compact_md_prompt(
        self,
        db_conn: sqlite3.Connection,
        context_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Compaction summarization uses the owner prompt from _compact.md."""
        log = MessageLogStore(db_conn)
        for index in range(30):
            log.append("main", HumanMessage(content=f"message number {index} " * 20))
        db_conn.commit()

        scratchpad_dir = Path(context_settings.scratchpad_directory)
        (scratchpad_dir / "_compact.md").write_text(
            "Custom compaction instructions from owner.",
            encoding="utf-8",
        )

        captured: dict[str, str] = {}

        class CapturingModel(FakeListChatModel):
            def invoke(self, input, config=None, **kwargs):  # noqa: ANN001
                messages = input
                captured["system"] = messages[0].content
                return super().invoke(input, config=config, **kwargs)

        model = CapturingModel(responses=["Custom summary."])

        compacted = maybe_compact_context(db_conn, context_settings, model, "main")
        db_conn.commit()

        assert compacted is True
        assert captured["system"] == "Custom compaction instructions from owner."


class TestLoadCompactionPrompt:
    """Compaction prompt loading tests."""

    def test_load_compaction_prompt_reads_compact_md(self, context_settings: Settings) -> None:
        """Return trimmed contents of _compact.md."""
        scratchpad_dir = Path(context_settings.scratchpad_directory)
        (scratchpad_dir / "_compact.md").write_text("Owner compaction rules.", encoding="utf-8")

        assert load_compaction_prompt(context_settings) == "Owner compaction rules."

    def test_load_compaction_prompt_raises_when_empty(self, context_settings: Settings) -> None:
        """Reject blank _compact.md files."""
        scratchpad_dir = Path(context_settings.scratchpad_directory)
        (scratchpad_dir / "_compact.md").write_text("", encoding="utf-8")

        with pytest.raises(CompactionError, match="missing or empty"):
            load_compaction_prompt(context_settings)

    def test_load_compaction_prompt_raises_when_missing(self, context_settings: Settings) -> None:
        """Reject when _compact.md does not exist."""
        scratchpad_dir = Path(context_settings.scratchpad_directory)
        (scratchpad_dir / "_compact.md").unlink()

        with pytest.raises(CompactionError, match="missing or empty"):
            load_compaction_prompt(context_settings)


class TestBuildAgentContext:
    """Full wake context assembly tests."""

    def test_build_agent_context_orders_components(
        self,
        db_conn: sqlite3.Connection,
        context_settings: Settings,
        sample_tools: list[StructuredTool],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Assemble system, pillars, transcript, and open tasks in order."""
        scratchpad_dir = Path(context_settings.scratchpad_directory)
        (scratchpad_dir / "_goals.md").write_text("Ship ThoughtBuddy v1.", encoding="utf-8")
        (scratchpad_dir / "_system.md").write_text("Play to win.", encoding="utf-8")
        (scratchpad_dir / "PROGRESS.md").write_text("Built context assembly.", encoding="utf-8")
        (scratchpad_dir / "SCRATCHPAD.md").write_text("Check queue next.", encoding="utf-8")

        monkeypatch.setattr(
            "context.build.build_knowledge_recall",
            lambda _settings, query, top_k=5: "[]",
        )

        log = MessageLogStore(db_conn)
        log.append("main", HumanMessage(content="prior turn"))
        CompactionStore(db_conn).create(
            "main",
            boundary_seq=1,
            retain_count=5,
            summary="Earlier wake.",
        )
        log.append("main", HumanMessage(content="recent turn"))
        db_conn.commit()

        pending_tasks = [{"id": "task-1", "status": "pending", "payload": {"message": "hello"}}]
        context = build_agent_context(
            db_conn,
            context_settings,
            thread_id="main",
            pending_tasks=pending_tasks,
            tools=sample_tools,
        )

        assert isinstance(context[0], SystemMessage)
        assert "ThoughtBuddy" in str(context[0].content)
        assert str(context[1].content).startswith(GOALS_PREFIX)
        assert str(context[2].content).startswith(SYSTEM_PREFIX)
        assert str(context[3].content).startswith(PLAN_PREFIX)
        assert str(context[4].content).startswith(WORLD_PREFIX)
        assert str(context[5].content).startswith(KNOWLEDGE_PREFIX)
        assert str(context[6].content).startswith(PROGRESS_PREFIX)
        assert str(context[7].content).startswith(SCRATCHPAD_PREFIX)
        assert str(context[8].content).startswith("Monitored processes:")
        assert str(context[9].content).startswith(CHECKPOINT_PREFIX)
        assert context[10].content == "recent turn"
        assert str(context[11].content).startswith("Open tasks:")
        assert "task-1" in str(context[11].content)

    def test_build_agent_context_uses_empty_placeholder_for_missing_files(
        self,
        db_conn: sqlite3.Connection,
        context_settings: Settings,
        sample_tools: list[StructuredTool],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Blank scratchpad files render as an empty placeholder."""
        monkeypatch.setattr(
            "context.build.build_knowledge_recall",
            lambda _settings, query, top_k=5: "[]",
        )
        scratchpad_dir = Path(context_settings.scratchpad_directory)
        for filename in ("_goals.md", "_system.md", "PROGRESS.md", "SCRATCHPAD.md"):
            (scratchpad_dir / filename).write_text("", encoding="utf-8")

        context = build_agent_context(
            db_conn,
            context_settings,
            thread_id="main",
            pending_tasks=[],
            tools=sample_tools,
        )

        assert "(empty)" in str(context[1].content)
        assert "(empty)" in str(context[2].content)
        assert "(empty)" in str(context[6].content)
        assert "(empty)" in str(context[7].content)
