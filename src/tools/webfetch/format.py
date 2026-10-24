"""Format rendered web pages for LLM consumption."""

from __future__ import annotations

import trafilatura

from tools.webfetch.fetcher import RenderedPage


def extract_page_text(page: RenderedPage) -> str:
    """Extract readable text from rendered HTML.

    Args:
        page: Rendered page from the headless browser.

    Returns:
        Main article/page text, or a fallback body extract.
    """
    extracted = trafilatura.extract(
        page.html,
        url=page.url,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if extracted:
        return extracted.strip()

    fallback = trafilatura.extract(
        page.html,
        url=page.url,
        include_comments=False,
        include_tables=True,
        favor_precision=False,
    )
    return (fallback or "").strip()


def format_page_content(page: RenderedPage, *, max_chars: int) -> str:
    """Format extracted page content for LLM consumption.

    Args:
        page: Rendered page from the headless browser.
        max_chars: Maximum characters of body text to include.

    Returns:
        Human-readable page summary with optional truncation notice.
    """
    body = extract_page_text(page)
    if not body:
        body = "[No readable text could be extracted from this page.]"

    truncated = len(body) > max_chars
    if truncated:
        body = body[:max_chars].rstrip()

    sections = [
        f"URL: {page.url}",
        f"Title: {page.title or 'Untitled'}",
        "Content:",
        body,
    ]
    if truncated:
        sections.append(f"[Truncated to {max_chars} characters.]")

    return "\n".join(sections)
