"""Self-registration and aggregation for agent tools."""

from __future__ import annotations

import importlib
import logging

from langchain_core.tools import BaseTool

from config import Settings
from experiments.registry import get_experiment_tools
from tools.registry import get_registered_tools, register_tools
from tools.tool_boundary import wrap_tool_with_boundary

logger = logging.getLogger("thoughtbuddy.tools")

_OPTIONAL_MODULES = (
    "tools.scheduling",
    "tools.monitors",
    "tools.scratchpad",
    "tools.progress",
    "tools.webfetch",
    "tools.websearch",
    "history.tools",
    "world.tools",
    "knowledge.tools",
    "plans.tools",
)


def _ensure_registered() -> None:
    """Import tool modules so they register with the registry."""
    for module_name in _OPTIONAL_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 — optional tool modules may fail at import
            logger.warning("Skipping tool module %s: %s", module_name, exc)


def get_tools(settings: Settings) -> list[BaseTool]:
    """Return the tools available to the ThoughtBuddy agent.

    Args:
        settings: Application settings.

    Returns:
        Configured LangChain tools for the agent graph.
    """
    _ensure_registered()
    tools = get_registered_tools(settings)
    tools.extend(get_experiment_tools(settings))
    return [wrap_tool_with_boundary(tool) for tool in tools]


__all__ = ["get_tools", "register_tools"]
