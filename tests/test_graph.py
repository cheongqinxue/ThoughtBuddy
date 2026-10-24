"""Tests for the ThoughtBuddy agent workflow graph."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool

from agent import run_agent
from graph.builder import build_graph
from graph.phases import PHASE_ORDER, tools_for_phase


class TestGraphPhases:
    """Phase configuration."""

    def test_phase_order_is_fixed(self) -> None:
        """Graph runs six phases in the documented order."""
        assert PHASE_ORDER == ("orient", "observe", "measure", "reason", "plan", "act")

    def test_observe_includes_world_tools(self) -> None:
        """Observe phase includes world and web tools."""
        tools = [
            StructuredTool.from_function(func=lambda: "", name="websearch", description=""),
            StructuredTool.from_function(func=lambda: "", name="world_chess_status", description=""),
            StructuredTool.from_function(func=lambda: "", name="plan_read", description=""),
        ]
        names = {tool.name for tool in tools_for_phase("observe", tools)}
        assert names == {"websearch", "world_chess_status"}

    def test_measure_includes_plan_world_and_history_tools(self) -> None:
        """Measure phase includes plan, world, and history read tools."""
        tools = [
            StructuredTool.from_function(func=lambda: "", name="plan_read", description=""),
            StructuredTool.from_function(func=lambda: "", name="world_observe", description=""),
            StructuredTool.from_function(func=lambda: "", name="history_read", description=""),
            StructuredTool.from_function(func=lambda: "", name="history_search", description=""),
            StructuredTool.from_function(func=lambda: "", name="websearch", description=""),
        ]
        names = {tool.name for tool in tools_for_phase("measure", tools)}
        assert names == {"plan_read", "world_observe", "history_read", "history_search"}


class TestBuildGraph:
    """Graph compilation and invocation tests."""

    def test_graph_compiles(self) -> None:
        """Graph compiles with a fake model."""
        model = FakeListChatModel(responses=["phase complete"] * 12)
        graph = build_graph(model, tools=[])
        assert graph is not None

    def test_graph_returns_assistant_reply(self) -> None:
        """Invoke the graph and return the fake model response."""
        model = FakeListChatModel(responses=["Test assistant reply."] * 12)
        graph = build_graph(model)

        reply = run_agent(graph, "Help me think through a decision.")

        assert reply == "Test assistant reply."

    def test_graph_appends_messages(self) -> None:
        """Graph invoke returns assistant messages beyond the initial input."""
        model = FakeListChatModel(responses=["Test assistant reply."] * 12)
        graph = build_graph(model)

        result = graph.invoke({"messages": [HumanMessage(content="Hello")]})

        assert len(result["messages"]) > 1
        assert result["messages"][0].content == "Hello"
        assert result["messages"][-1].content == "Test assistant reply."
