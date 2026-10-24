"""Tests for message archive pagination, FTS search, and history tools."""

from __future__ import annotations

import json
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from config import Settings
from db.connection import connect
from history.tools import create_tools
from store.messages import HistoryCursor, MessageLogStore


@pytest.fixture
def message_conn(tmp_path):
    """SQLite connection with schema initialized."""
    conn = connect(tmp_path / "test.db")
    yield conn
    conn.close()


class TestMessageLogPagination:
    """Datetime pagination over message_log."""

    def test_list_by_time_orders_ascending(self, message_conn) -> None:
        """Messages return oldest-first with timestamp ordering."""
        store = MessageLogStore(message_conn)
        store.append("t1", HumanMessage(content="first"))
        time.sleep(1.1)
        store.append("t1", HumanMessage(content="second"))
        message_conn.commit()

        page = store.list_by_time("t1", limit=10, order="asc")

        assert [message["content"] for message in page.messages] == ["first", "second"]
        assert page.total_count == 2
        assert page.next_cursor is not None

    def test_cursor_continues_chronological_page(self, message_conn) -> None:
        """Cursor pagination returns the next time-ordered slice."""
        store = MessageLogStore(message_conn)
        for index in range(3):
            store.append("t1", HumanMessage(content=f"msg-{index}"))
            time.sleep(1.1)
        message_conn.commit()

        first = store.list_by_time("t1", limit=2, order="asc")
        second = store.list_by_time(
            "t1",
            limit=2,
            order="asc",
            cursor=HistoryCursor.decode(first.next_cursor),
        )

        assert [message["content"] for message in first.messages] == ["msg-0", "msg-1"]
        assert [message["content"] for message in second.messages] == ["msg-2"]

    def test_start_at_bounds_window(self, message_conn) -> None:
        """Inclusive timestamp lower bound filters pages."""
        store = MessageLogStore(message_conn)
        store.append("t1", HumanMessage(content="early"))
        time.sleep(1.1)
        store.append("t1", HumanMessage(content="late"))
        message_conn.commit()

        rows = message_conn.execute(
            "SELECT created_at FROM message_log WHERE thread_id = 't1' ORDER BY seq ASC",
        ).fetchall()
        start_at = rows[1]["created_at"]

        page = store.list_by_time("t1", start_at=start_at, order="asc")

        assert len(page.messages) == 1
        assert page.messages[0]["content"] == "late"


class TestMessageLogSearch:
    """BM25 search over archived messages."""

    def test_search_finds_tool_name_and_content(self, message_conn) -> None:
        """FTS indexes tool names and message bodies."""
        store = MessageLogStore(message_conn)
        store.append(
            "t1",
            AIMessage(
                content="playing chess",
                tool_calls=[{"name": "world_chess_move", "args": {"uci": "e2e4"}, "id": "1"}],
            ),
        )
        store.append(
            "t1",
            ToolMessage(
                content='{"ok": true, "move": "e2e4"}',
                tool_call_id="1",
                name="world_chess_move",
            ),
        )
        message_conn.commit()

        hits = store.search("t1", query="world_chess_move e2e4", limit=5)

        assert hits
        assert any("e2e4" in hit.snippet or "world_chess_move" in hit.snippet for hit in hits)


class TestHistoryTools:
    """Agent-facing history tools."""

    def test_history_read_via_tool(self, tmp_path, settings: Settings) -> None:
        """history_read returns paginated archive JSON."""
        db_path = tmp_path / "test.db"
        settings = settings.model_copy(update={"thoughtbuddy_db_path": str(db_path)})
        conn = connect(db_path)
        store = MessageLogStore(conn)
        store.append(settings.default_thread_id, HumanMessage(content="logged action"))
        conn.commit()
        conn.close()

        tools = {tool.name: tool for tool in create_tools(settings)}
        payload = json.loads(tools["history_read"].invoke({"limit": 10, "order": "asc"}))

        assert payload["count"] == 1
        assert payload["messages"][0]["content"] == "logged action"

    def test_history_search_via_tool(self, tmp_path, settings: Settings) -> None:
        """history_search returns ranked hits."""
        db_path = tmp_path / "test.db"
        settings = settings.model_copy(update={"thoughtbuddy_db_path": str(db_path)})
        conn = connect(db_path)
        store = MessageLogStore(conn)
        store.append(settings.default_thread_id, HumanMessage(content="AAPL limit order filled"))
        conn.commit()
        conn.close()

        tools = {tool.name: tool for tool in create_tools(settings)}
        payload = json.loads(tools["history_search"].invoke({"query": "AAPL", "limit": 5}))

        assert payload["count"] >= 1
        assert any("AAPL" in hit["snippet"] for hit in payload["hits"])
