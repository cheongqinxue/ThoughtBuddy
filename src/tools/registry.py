"""Self-registration for agent tool factories."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import BaseTool

from config import Settings

ToolFactory = Callable[[Settings], list[BaseTool]]

_FACTORIES: list[ToolFactory] = []


def register_tools(factory: ToolFactory) -> None:
    """Register a tool factory.

    Args:
        factory: Callable that takes ``Settings`` and returns LangChain tools.
    """
    _FACTORIES.append(factory)


def get_registered_tools(settings: Settings) -> list[BaseTool]:
    """Collect tools from all registered factories.

    Args:
        settings: Application settings.

    Returns:
        Flat list of configured tools.
    """
    tools: list[BaseTool] = []
    for factory in _FACTORIES:
        tools.extend(factory(settings))
    return tools
