"""Tests for the experiment registry."""

from __future__ import annotations

import pytest

import experiments  # noqa: F401 — registers built-in experiments
from config import Settings
from experiments.registry import get_experiment_tools, list_experiments, register_experiment
from langchain_core.tools import StructuredTool


class TestExperimentRegistry:
    """Core experiment registration capabilities."""

    def test_chess_is_registered(self) -> None:
        """Built-in chess experiment is available after import."""
        assert "chess" in list_experiments()

    def test_no_active_experiment_returns_empty_tools(self, settings: Settings) -> None:
        """Return no experiment tools when active experiment is unset."""
        settings = settings.model_copy(update={"active_experiment": ""})
        assert get_experiment_tools(settings) == []

    def test_active_chess_returns_world_tools(self, settings: Settings) -> None:
        """Return chess virtual-world tools when chess is active."""
        settings = settings.model_copy(update={"active_experiment": "chess"})
        tools = get_experiment_tools(settings)
        names = {tool.name for tool in tools}
        assert names == {"world_chess_status", "world_chess_move", "world_chess_reset"}

    def test_unknown_experiment_raises(self, settings: Settings) -> None:
        """Raise when config points at an unregistered experiment."""
        settings = settings.model_copy(update={"active_experiment": "unknown-game"})
        with pytest.raises(ValueError, match="Unknown experiment: 'unknown-game'"):
            get_experiment_tools(settings)

    def test_register_experiment_adds_factory(self, settings: Settings) -> None:
        """Allow registering additional experiments at runtime."""

        def stub_factory(_settings: Settings) -> list[StructuredTool]:
            def ping() -> str:
                return "pong"

            return [
                StructuredTool.from_function(
                    func=ping,
                    name="stub_ping",
                    description="Stub experiment tool.",
                ),
            ]

        register_experiment("stub", stub_factory)
        settings = settings.model_copy(update={"active_experiment": "stub"})
        tools = get_experiment_tools(settings)
        assert [tool.name for tool in tools] == ["stub_ping"]
