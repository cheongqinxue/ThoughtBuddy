"""Tests for the ThoughtBuddy agent graph export."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

import agent
from agent import create_agent, make_graph
from graph.builder import build_graph


class TestAgentGraphExport:
    """LangGraph server export tests."""

    def test_make_graph_returns_compiled_graph(self, monkeypatch) -> None:
        """Expose a graph factory for LangGraph dev server loading."""
        compiled = build_graph(FakeListChatModel(responses=["Hello from ThoughtBuddy."]))
        wrapped = build_graph(FakeListChatModel(responses=["Wrapped."]))
        monkeypatch.setattr(agent, "create_agent", lambda _settings=None: compiled)
        monkeypatch.setattr(
            agent,
            "apply_studio_langfuse_config",
            lambda graph, _settings: wrapped,
        )

        result = make_graph(RunnableConfig(configurable={}))

        assert result is wrapped
        assert isinstance(result, CompiledStateGraph)

    def test_create_agent_returns_compiled_graph(self, settings) -> None:
        """Build a fresh compiled graph from settings."""
        compiled = create_agent(settings)

        assert isinstance(compiled, CompiledStateGraph)
