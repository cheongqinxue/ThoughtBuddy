"""Tests for the websearch tool and LangSearch provider."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from config import Settings
from tools.websearch import create_tools, format_search_results
from tools.websearch.providers.langsearch import LangSearchClient, extract_search_data
from utils.api_retry import ApiClientError, ApiRetry, RetryableApiError


class TestExtractSearchData:
    """LangSearch payload parsing tests."""

    def test_extracts_wrapped_response(self) -> None:
        """Parse the standard LangSearch API envelope."""
        payload = {
            "code": 200,
            "data": {
                "_type": "SearchResponse",
                "webPages": {"value": [{"name": "Example", "url": "https://example.com"}]},
            },
        }

        data = extract_search_data(payload)

        assert data["_type"] == "SearchResponse"
        assert len(data["webPages"]["value"]) == 1

    def test_extracts_direct_search_response(self) -> None:
        """Parse a bare SearchResponse payload."""
        payload = {
            "_type": "SearchResponse",
            "webPages": {"value": []},
        }

        data = extract_search_data(payload)

        assert data["_type"] == "SearchResponse"

    def test_raises_for_api_error(self) -> None:
        """Raise when the API envelope reports a non-200 code."""
        payload = {"code": 401, "msg": "Unauthorized", "data": None}

        with pytest.raises(ApiClientError, match="401"):
            extract_search_data(payload)

    def test_maps_transient_api_code_to_retryable_error(self) -> None:
        """Retry JSON envelope failures that look like transient HTTP statuses."""
        payload = {"code": 503, "msg": "Service unavailable", "data": None}

        with pytest.raises(RetryableApiError, match="503"):
            extract_search_data(payload)


class TestFormatSearchResults:
    """Search result formatting tests."""

    def test_formats_citations(self) -> None:
        """Format web pages into numbered citations."""
        search_data = {
            "queryContext": {"originalQuery": "apple esg 2024"},
            "webPages": {
                "value": [
                    {
                        "name": "Apple cuts greenhouse gas emissions in half",
                        "url": "https://www.apple.com/newsroom/2024/04/",
                        "summary": "Apple reduced emissions by more than 55 percent.",
                        "datePublished": "2024-04-18T00:00:00Z",
                    }
                ]
            },
        }

        formatted = format_search_results(search_data)

        assert "Query: apple esg 2024" in formatted
        assert "Citation 1" in formatted
        assert "Apple cuts greenhouse gas emissions in half" in formatted
        assert "https://www.apple.com/newsroom/2024/04/" in formatted

    def test_returns_message_for_empty_results(self) -> None:
        """Return a helpful message when no pages are found."""
        formatted = format_search_results({"webPages": {"value": []}})

        assert formatted == "No relevant web results found."


class TestLangSearchClient:
    """LangSearch HTTP client tests."""

    def test_search_returns_parsed_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return parsed search data from a successful API response."""
        response_payload: dict[str, Any] = {
            "code": 200,
            "data": {
                "_type": "SearchResponse",
                "webPages": {"value": [{"name": "Result", "url": "https://example.com"}]},
            },
        }

        class MockResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return response_payload

        class MockClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> MockClient:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def post(self, *args: Any, **kwargs: Any) -> MockResponse:
                assert kwargs["json"]["query"] == "apple esg 2024"
                assert kwargs["headers"]["Authorization"] == "Bearer test-key"
                return MockResponse()

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = LangSearchClient("test-key")
        data = client.search("apple esg 2024")

        assert data["_type"] == "SearchResponse"

    def test_search_retries_rate_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retry HTTP 429 responses before succeeding."""
        attempts = {"count": 0}
        response_payload: dict[str, Any] = {
            "code": 200,
            "data": {
                "_type": "SearchResponse",
                "webPages": {"value": [{"name": "Result", "url": "https://example.com"}]},
            },
        }

        class MockResponse:
            def __init__(self, status_code: int) -> None:
                self.status_code = status_code

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    request = httpx.Request("POST", "https://api.langsearch.com/v1/web-search")
                    response = httpx.Response(self.status_code, request=request)
                    raise httpx.HTTPStatusError("error", request=request, response=response)

            def json(self) -> dict[str, Any]:
                return response_payload

        class MockClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> MockClient:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def post(self, *args: Any, **kwargs: Any) -> MockResponse:
                attempts["count"] += 1
                if attempts["count"] == 1:
                    return MockResponse(429)
                return MockResponse(200)

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = LangSearchClient(
            "test-key",
            retry=ApiRetry(max_attempts=3, min_wait_seconds=0.01, max_wait_seconds=0.05),
        )
        data = client.search("apple esg 2024")

        assert attempts["count"] == 2
        assert data["_type"] == "SearchResponse"


class TestWebsearchTool:
    """LangChain websearch tool tests."""

    def test_tool_returns_formatted_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return formatted search output from the websearch tool."""
        search_data = {
            "_type": "SearchResponse",
            "webPages": {
                "value": [
                    {
                        "name": "Apple ESG",
                        "url": "https://example.com/esg",
                        "summary": "Highlights from the report.",
                    }
                ]
            },
        }

        def mock_search(
            self: LangSearchClient,
            query: str,
            *,
            count: int = 5,
            summary: bool = True,
            freshness: str = "noLimit",
        ) -> dict[str, Any]:
            assert query == "apple esg report"
            return search_data

        monkeypatch.setattr(LangSearchClient, "search", mock_search)

        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test-key",
            agent_model="test-model",
            langsearch_api_key="test-key",
        )
        tool = create_tools(settings)[0]
        result = tool.invoke({"query": "apple esg report", "count": 3})

        assert "Citation 1" in result
        assert "Apple ESG" in result

    def test_tool_returns_json_error_for_recoverable_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Return structured JSON and rely on boundary logging for recoverable failures."""

        def mock_search(
            self: LangSearchClient,
            query: str,
            *,
            count: int = 5,
            summary: bool = True,
            freshness: str = "noLimit",
        ) -> dict[str, Any]:
            raise RetryableApiError("network down")

        monkeypatch.setattr(LangSearchClient, "search", mock_search)

        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test-key",
            agent_model="test-model",
            langsearch_api_key="test-key",
        )
        from tools.tool_boundary import wrap_tool_with_boundary

        tool = wrap_tool_with_boundary(create_tools(settings)[0])
        result = tool.invoke({"query": "apple esg report"})

        import json

        payload = json.loads(result)
        assert payload["ok"] is False
        assert payload["tool"] == "websearch"
        assert payload["retryable"] is True
