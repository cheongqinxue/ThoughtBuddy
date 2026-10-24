"""Web search agent tool — provider-agnostic ``websearch`` with swappable backends."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from config import Settings
from tools.websearch.backend import Freshness, create_websearch_backend
from tools.websearch.format import format_search_results
from tools.websearch.schemas import WebSearchInput


def create_tools(settings: Settings) -> list[StructuredTool]:
    """Create websearch tools from application settings.

    Args:
        settings: Application settings.

    Returns:
        LangChain tools that perform web queries, or empty when no backend
        is configured.
    """
    backend = create_websearch_backend(settings)
    if backend is None:
        return []

    def websearch(
        query: str,
        count: int = 5,
        freshness: Freshness = "noLimit",
    ) -> str:
        """Search the web for current information."""
        search_data = backend.search(query, count=count, freshness=freshness)
        return format_search_results(search_data)

    return [
        StructuredTool.from_function(
            func=websearch,
            name="websearch",
            description=(
                "Search the web for up-to-date information. Use this when the user asks "
                "about recent events, reports, facts, or anything that requires current "
                "information from the internet."
            ),
            args_schema=WebSearchInput,
        ),
    ]


from tools.registry import register_tools

register_tools(create_tools)

__all__ = ["Freshness", "WebSearchInput", "create_tools", "format_search_results"]
