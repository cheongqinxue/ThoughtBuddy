"""Live websearch integration tests (LangSearch provider).

These tests call the real LangSearch API and are excluded from the default
``pytest`` run. Execute them explicitly with:

    uv run pytest -m integration
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from config import Settings
from tools.websearch import create_tools, format_search_results
from tools.websearch.providers.langsearch import LangSearchClient


@pytest.fixture
def langsearch_api_key() -> str:
    """Load a LangSearch API key from the environment.

    Returns:
        LangSearch API key.

    Raises:
        pytest.skip: If ``LANGSEARCH_API_KEY`` is not configured.
    """
    load_dotenv()
    api_key = os.getenv("LANGSEARCH_API_KEY", "").strip()
    if not api_key:
        pytest.skip("LANGSEARCH_API_KEY is not set")
    return api_key


@pytest.mark.integration
class TestWebsearchLive:
    """Live websearch tests via the LangSearch provider."""

    def test_client_search_returns_results(self, langsearch_api_key: str) -> None:
        """Perform a live web search and return parsed result pages."""
        client = LangSearchClient(langsearch_api_key)
        data = client.search("ThoughtBuddy agent framework", count=3, summary=True)

        results = data.get("webPages", {}).get("value") or []
        assert results, "Expected LangSearch to return at least one web result"
        assert results[0].get("name")
        assert results[0].get("url")

    def test_websearch_tool_returns_formatted_citations(
        self,
        langsearch_api_key: str,
    ) -> None:
        """Run the websearch tool end-to-end against LangSearch."""
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test-key",
            agent_model="test-model",
            langsearch_api_key=langsearch_api_key,
        )
        tool = create_tools(settings)[0]
        result = tool.invoke({"query": "LangGraph documentation", "count": 2})

        assert "Citation 1" in result
        assert "http" in result.lower()
        assert result != "No relevant web results found."

    def test_format_search_results_with_live_payload(self, langsearch_api_key: str) -> None:
        """Format a live LangSearch payload into agent-readable citations."""
        client = LangSearchClient(langsearch_api_key)
        data = client.search("Python pytest integration tests", count=2, summary=False)

        formatted = format_search_results(data)

        assert "Citation 1" in formatted
        assert "Title:" in formatted
        assert "URL:" in formatted
