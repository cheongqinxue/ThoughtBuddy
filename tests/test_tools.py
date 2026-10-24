"""Tests for agent tool registry."""

from __future__ import annotations

import pytest

from config import Settings
from tools import get_tools
from tools.tool_boundary import is_tool_boundary_wrapped


class TestGetTools:
    """Tool registry tests."""

    def test_returns_core_agent_tools(self, settings: Settings) -> None:
        """Return scheduling, pillar, and web tools when configured."""
        tools = get_tools(settings)
        names = {tool.name for tool in tools}

        assert {
            "create_alert",
            "update_alert",
            "update_task",
            "list_alerts",
            "edit_scratchpad",
            "edit_progress",
            "websearch",
            "webfetch",
            "world_observe",
            "history_read",
            "history_search",
            "knowledge_add",
            "knowledge_edit",
            "knowledge_search",
            "plan_read",
            "plan_add_node",
            "plan_update_node",
            "plan_link",
        }.issubset(names)

    def test_omits_websearch_without_langsearch_api_key(self) -> None:
        """Omit websearch when LangSearch credentials are missing."""
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test-key",
            agent_model="test-model",
            langsearch_api_key="",
        )

        tools = get_tools(settings)
        names = {tool.name for tool in tools}
        assert "websearch" not in names
        assert "create_alert" in names

    def test_scheduling_value_error_returns_json(self, settings: Settings) -> None:
        """Agent mistakes in scheduling tools should not crash the wake."""
        tools = {tool.name: tool for tool in get_tools(settings)}
        result = tools["create_alert"].invoke(
            {
                "description": "test",
                "schedule_kind": "once",
            },
        )
        import json

        payload = json.loads(result)
        assert payload["ok"] is False
        assert payload["tool"] == "create_alert"
        assert "run_at" in payload["message"]

    def test_all_tools_are_boundary_wrapped(self, settings: Settings) -> None:
        """Every aggregated tool should use the shared error boundary."""
        import experiments  # noqa: F401 — register built-ins

        configured = settings.model_copy(update={"active_experiment": "chess"})
        tools = get_tools(configured)
        assert tools
        assert all(is_tool_boundary_wrapped(tool) for tool in tools)

    def test_includes_experiment_tools_when_active(self, settings: Settings) -> None:
        """Append experiment tools when an experiment is active."""
        import experiments  # noqa: F401 — register built-ins

        settings = settings.model_copy(update={"active_experiment": "chess"})
        tools = get_tools(settings)
        names = {tool.name for tool in tools}
        assert {"world_chess_status", "world_chess_move"}.issubset(names)
