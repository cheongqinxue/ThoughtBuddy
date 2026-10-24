"""Web search backend protocol — swappable provider implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from config import Settings

Freshness = Literal["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"]


class WebSearchBackend(Protocol):
    """Provider-agnostic web search interface."""

    def search(
        self,
        query: str,
        *,
        count: int = 5,
        freshness: Freshness = "noLimit",
    ) -> dict[str, Any]:
        """Run a web search and return a normalized SearchResponse-shaped dict.

        Args:
            query: User search query.
            count: Number of results to return.
            freshness: Time range filter for results.

        Returns:
            Dict with ``webPages`` (and optional ``queryContext``), same shape
            consumed by ``format_search_results``.
        """


def create_websearch_backend(settings: Settings) -> WebSearchBackend | None:
    """Instantiate the configured web search backend, if credentials allow.

    Args:
        settings: Application settings.

    Returns:
        Backend instance, or ``None`` when search is unavailable.
    """
    provider = settings.websearch_provider or "langsearch"
    if provider == "langsearch":
        if not settings.langsearch_api_key:
            return None
        from tools.websearch.providers.langsearch import LangSearchBackend

        return LangSearchBackend(settings)
    msg = f"Unknown websearch provider: {provider!r}"
    raise ValueError(msg)
