"""Web fetch agent tool — Playwright rendering with SSRF-safe URL validation."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from config import Settings
from tools.webfetch.fetcher import PageFetcher, RenderedPage, WebFetchError
from tools.webfetch.format import extract_page_text, format_page_content
from tools.webfetch.schemas import WebFetchInput
from tools.webfetch.url_safety import UnsafeUrlError
from utils.api_retry import create_api_retry


def create_tools(settings: Settings) -> list[StructuredTool]:
    """Create webfetch tools from application settings.

    Args:
        settings: Application settings controlling fetch timeouts and limits.

    Returns:
        LangChain tools that fetch fully rendered page content.
    """
    fetcher = PageFetcher(
        timeout_ms=settings.webfetch_timeout_seconds * 1_000,
        network_idle_timeout_ms=settings.webfetch_network_idle_timeout_ms,
        post_load_wait_ms=settings.webfetch_post_load_wait_ms,
        retry=create_api_retry(settings, operation_name="Webfetch page load"),
    )
    default_max_chars = settings.webfetch_max_chars

    def webfetch(url: str, max_chars: int = default_max_chars) -> str:
        """Fetch a web page after JavaScript rendering."""

        try:
            page = fetcher.fetch(str(url))
        except UnsafeUrlError as exc:
            raise WebFetchError(str(exc)) from exc
        return format_page_content(page, max_chars=max_chars)

    return [
        StructuredTool.from_function(
            func=webfetch,
            name="webfetch",
            description=(
                "Fetch the full contents of a specific public web page URL. Use this "
                "after websearch when snippets are insufficient, or when the user "
                "provides a URL that must be read in depth. Runs a headless browser "
                "so JavaScript-rendered pages are loaded before text is extracted."
            ),
            args_schema=WebFetchInput,
        ),
    ]


from tools.registry import register_tools

register_tools(create_tools)

__all__ = [
    "PageFetcher",
    "RenderedPage",
    "WebFetchError",
    "WebFetchInput",
    "create_tools",
    "extract_page_text",
    "format_page_content",
]
