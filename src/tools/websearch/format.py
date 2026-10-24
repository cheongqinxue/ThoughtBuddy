"""Format normalized search payloads for LLM consumption."""

from __future__ import annotations

from typing import Any


def format_search_results(search_data: dict[str, Any]) -> str:
    """Format search results into numbered citations.

    Args:
        search_data: Normalized SearchResponse payload (``webPages``, optional
            ``queryContext``).

    Returns:
        Human-readable search results with numbered citations.
    """
    web_pages = search_data.get("webPages") or {}
    results = web_pages.get("value") or []
    if not results:
        return "No relevant web results found."

    original_query = (search_data.get("queryContext") or {}).get("originalQuery")
    sections: list[str] = []
    if original_query:
        sections.append(f"Query: {original_query}")

    for index, page in enumerate(results, start=1):
        title = page.get("name") or "Untitled"
        url = page.get("url") or page.get("displayUrl") or "Unknown URL"
        content = page.get("summary") or page.get("snippet") or ""
        published = page.get("datePublished")
        citation = (
            f"Citation {index}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Published: {published or 'Unknown'}\n"
            f"Content: {content}"
        )
        sections.append(citation)

    return "\n\n".join(sections)
