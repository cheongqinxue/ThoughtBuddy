"""Tests for the webfetch tool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config import Settings
from utils.api_retry import ApiRetry, RetryableApiError
from tools.webfetch import (
    PageFetcher,
    RenderedPage,
    WebFetchError,
    create_tools as create_webfetch_tools,
    extract_page_text,
    format_page_content,
)
from tools.webfetch.fetcher import is_retryable_playwright_error
from playwright.sync_api import Error as PlaywrightError


class TestExtractPageText:
    """HTML-to-text extraction tests."""

    def test_extracts_main_content(self) -> None:
        """Extract readable article text from rendered HTML."""
        page = RenderedPage(
            url="https://example.com/post",
            title="Example Post",
            html=(
                "<html><head><title>Example Post</title></head>"
                "<body><article><h1>Example Post</h1>"
                "<p>This is the article body with enough words to extract.</p>"
                "</article></body></html>"
            ),
        )

        text = extract_page_text(page)

        assert "article body" in text


class TestFormatPageContent:
    """Output formatting tests."""

    def test_truncates_long_content(self) -> None:
        """Include a truncation notice when content exceeds max_chars."""
        page = RenderedPage(
            url="https://example.com/long",
            title="Long Page",
            html="<html><body><p>" + ("word " * 200) + "</p></body></html>",
        )

        output = format_page_content(page, max_chars=50)

        assert "Truncated to 50 characters" in output
        assert output.count("word") <= 50


class TestPageFetcher:
    """Headless browser fetch tests."""

    @patch("tools.webfetch.fetcher.sync_playwright")
    def test_fetch_waits_for_load_and_network_idle(self, mock_sync_playwright) -> None:
        """Wait for load, network idle, and post-load delay before reading HTML."""
        page = MagicMock()
        page.url = "https://example.com/article"
        page.title.return_value = "Example"
        page.content.return_value = "<html><body><p>Rendered</p></body></html>"

        response = MagicMock()
        response.status = 200
        page.goto.return_value = response

        browser = MagicMock()
        browser.new_page.return_value = page

        playwright = MagicMock()
        playwright.chromium.launch.return_value = browser
        mock_sync_playwright.return_value.__enter__.return_value = playwright

        fetcher = PageFetcher(
            timeout_ms=30_000,
            network_idle_timeout_ms=5_000,
            post_load_wait_ms=1_000,
        )
        result = fetcher.fetch("https://example.com/article")

        page.goto.assert_called_once_with(
            "https://example.com/article",
            wait_until="domcontentloaded",
            timeout=30_000,
        )
        page.wait_for_load_state.assert_any_call("load", timeout=30_000)
        page.wait_for_load_state.assert_any_call("networkidle", timeout=5_000)
        page.wait_for_timeout.assert_called_once_with(1_000)
        assert result.html == "<html><body><p>Rendered</p></body></html>"

    @patch("tools.webfetch.fetcher.sync_playwright")
    def test_fetch_raises_for_http_error(self, mock_sync_playwright) -> None:
        """Surface HTTP error statuses as fetch failures."""
        page = MagicMock()
        response = MagicMock()
        response.status = 404
        page.goto.return_value = response

        browser = MagicMock()
        browser.new_page.return_value = page

        playwright = MagicMock()
        playwright.chromium.launch.return_value = browser
        mock_sync_playwright.return_value.__enter__.return_value = playwright

        fetcher = PageFetcher(post_load_wait_ms=0, network_idle_timeout_ms=0)

        with pytest.raises(WebFetchError, match="HTTP 404"):
            fetcher.fetch("https://example.com/missing")

    def test_fetch_retries_transient_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retry transient browser failures before surfacing an error."""
        fetcher = PageFetcher(
            post_load_wait_ms=0,
            network_idle_timeout_ms=0,
            retry=ApiRetry(max_attempts=3, min_wait_seconds=0.01, max_wait_seconds=0.05),
        )
        attempts = {"count": 0}
        page = RenderedPage(
            url="https://example.com/post",
            title="Example",
            html="<html><body><p>Rendered</p></body></html>",
        )

        def flaky_fetch_once(_validated_url: str) -> RenderedPage:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RetryableApiError("temporary browser failure")
            return page

        monkeypatch.setattr(fetcher, "_fetch_once", flaky_fetch_once)
        monkeypatch.setattr("tools.webfetch.fetcher.validate_fetch_url", lambda url: url)

        result = fetcher.fetch("https://example.com/post")

        assert result == page
        assert attempts["count"] == 3


class TestPlaywrightErrorClassification:
    """Playwright retry classification tests."""

    def test_network_errors_are_retryable(self) -> None:
        """Treat common browser network failures as transient."""
        exc = PlaywrightError("net::ERR_CONNECTION_RESET")

        assert is_retryable_playwright_error(exc) is True


class TestWebfetchTool:
    """LangChain tool wrapper tests."""

    @patch("tools.webfetch.PageFetcher.fetch")
    def test_tool_returns_formatted_page(self, mock_fetch) -> None:
        """Return formatted page text from the tool."""
        mock_fetch.return_value = RenderedPage(
            url="https://example.com/post",
            title="Example Post",
            html=(
                "<html><body><article><h1>Example Post</h1>"
                "<p>Detailed article content for the agent.</p></article></body></html>"
            ),
        )
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test-key",
            agent_model="test-model",
            langsearch_api_key="test-langsearch-key",
        )

        tool = create_webfetch_tools(settings)[0]
        output = tool.invoke({"url": "https://example.com/post"})

        assert "URL: https://example.com/post" in output
        assert "Detailed article content" in output

    @patch("tools.webfetch.PageFetcher.fetch")
    def test_tool_returns_validation_errors(self, mock_fetch) -> None:
        """Return user-facing errors instead of raising from the tool."""
        mock_fetch.side_effect = WebFetchError("Timed out loading page")
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test-key",
            agent_model="test-model",
            langsearch_api_key="test-langsearch-key",
        )

        from tools.tool_boundary import wrap_tool_with_boundary

        tool = wrap_tool_with_boundary(create_webfetch_tools(settings)[0])
        output = tool.invoke({"url": "https://example.com/slow"})

        import json

        payload = json.loads(output)
        assert payload["ok"] is False
        assert payload["tool"] == "webfetch"
        assert payload["message"] == "Timed out loading page"
        assert payload["retryable"] is False
