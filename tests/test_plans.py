"""Tests for plan DAG store and tools."""

from __future__ import annotations

import json

import pytest

from config import Settings
from db.connection import connect
from store.plans import PlanStore


@pytest.fixture
def plan_conn(tmp_path):
    """SQLite connection with schema initialized."""
    conn = connect(tmp_path / "test.db")
    yield conn
    conn.close()


class TestPlanStore:
    """Plan graph persistence and dependency semantics."""

    def test_add_nodes_and_edge(self, plan_conn) -> None:
        """Create nodes and dependency edge."""
        store = PlanStore(plan_conn)
        a = store.add_node("t1", payload={"description": "A"})
        b = store.add_node("t1", payload={"description": "B"})
        store.add_edge("t1", a.id, b.id)
        plan_conn.commit()
        assert len(store.list_edges("t1")) == 1

    def test_reject_cycle(self, plan_conn) -> None:
        """Cycle edges are rejected."""
        store = PlanStore(plan_conn)
        a = store.add_node("t1")
        b = store.add_node("t1")
        c = store.add_node("t1")
        store.add_edge("t1", a.id, b.id)
        store.add_edge("t1", b.id, c.id)
        with pytest.raises(ValueError, match="cycle"):
            store.add_edge("t1", c.id, a.id)

    def test_ready_and_blocked(self, plan_conn) -> None:
        """Ready/blocked sets follow dependency completion."""
        store = PlanStore(plan_conn)
        a = store.add_node("t1", payload={"description": "root"})
        b = store.add_node("t1", payload={"description": "child"})
        store.add_edge("t1", a.id, b.id)
        assert [n.id for n in store.ready_nodes("t1")] == [a.id]
        store.update_node("t1", a.id, status="done")
        assert [n.id for n in store.ready_nodes("t1")] == [b.id]
        assert store.blocked_nodes("t1") == []


class TestPlanTools:
    """Plan agent tools."""

    def test_plan_read_via_tool(self, tmp_path, settings: Settings) -> None:
        """plan_add_node and plan_read work through tools."""
        from plans.tools import create_tools

        settings = settings.model_copy(update={"thoughtbuddy_db_path": str(tmp_path / "test.db")})
        tools = {tool.name: tool for tool in create_tools(settings)}
        add_result = json.loads(
            tools["plan_add_node"].invoke({"description": "Win game", "status": "active"}),
        )
        assert add_result["ok"] is True
        read_result = json.loads(tools["plan_read"].invoke({}))
        assert len(read_result["nodes"]) == 1
