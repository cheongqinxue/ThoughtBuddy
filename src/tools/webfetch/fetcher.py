"""Headless browser page fetcher for webfetch."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from tools.webfetch.url_safety import UnsafeUrlError, validate_fetch_url
from utils.api_retry import (
    ApiRetry,
    RetryableApiError,
    classify_http_status,
    is_retryable_http_status,
)

logger = logging.getLogger("thoughtbuddy.tools.webfetch")

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; ThoughtBuddy/1.0; +https://github.com/thoughtbuddy)"
)
_TRANSIENT_PLAYWRIGHT_HINTS = (
    "net::",
    "NS_ERROR",
    "connection refused",
    "connection reset",
    "target page, context or browser has been closed",
    "browser has been closed",
)


class WebFetchError(Exception):
    """Raised when a page cannot be fetched or rendered."""


@dataclass(frozen=True)
class RenderedPage:
    """Fully rendered page content captured from a headless browser."""

    url: str
    title: str
    html: str


class PageFetcher:
    """Fetch web pages with JavaScript rendering via Playwright."""

    def __init__(
        self,
        *,
        timeout_ms: int = 30_000,
        network_idle_timeout_ms: int = 5_000,
        post_load_wait_ms: int = 1_000,
        user_agent: str = _DEFAULT_USER_AGENT,
        retry: ApiRetry | None = None,
    ) -> None:
        """Initialize the fetcher.

        Args:
            timeout_ms: Default Playwright operation timeout in milliseconds.
            network_idle_timeout_ms: Extra wait for network idle after load.
                Set to 0 to skip. Times out gracefully and continues.
            post_load_wait_ms: Fixed delay after load for lazy-rendered content.
                Set to 0 to skip.
            user_agent: Browser user agent string.
            retry: Optional retry helper for transient browser/network failures.
        """
        self._timeout_ms = timeout_ms
        self._network_idle_timeout_ms = network_idle_timeout_ms
        self._post_load_wait_ms = post_load_wait_ms
        self._user_agent = user_agent
        self._retry = retry or ApiRetry(operation_name="Webfetch page load")

    def fetch(self, url: str) -> RenderedPage:
        """Load a URL in headless Chromium and return rendered HTML.

        Waits for DOMContentLoaded, then the load event, then optionally
        network idle and a short post-load delay so JavaScript-heavy pages
        can finish rendering.

        Args:
            url: Public HTTP(S) URL.

        Returns:
            Rendered page metadata and HTML.

        Raises:
            UnsafeUrlError: If the URL fails SSRF validation.
            WebFetchError: If navigation or rendering fails with a non-retryable error.
            RetryableApiError: If retries are exhausted for a transient failure.
        """
        validated_url = validate_fetch_url(url)
        return self._retry.call(lambda: self._fetch_once(validated_url))

    def _fetch_once(self, validated_url: str) -> RenderedPage:
        """Perform one browser fetch attempt.

        Args:
            validated_url: SSRF-validated URL.

        Returns:
            Rendered page metadata and HTML.

        Raises:
            WebFetchError: For client-side or permanent fetch failures.
            RetryableApiError: For transient browser or HTTP failures.
        """
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    page = browser.new_page(user_agent=self._user_agent)
                    page.set_default_timeout(self._timeout_ms)
                    response = page.goto(
                        validated_url,
                        wait_until="domcontentloaded",
                        timeout=self._timeout_ms,
                    )
                    if response is None:
                        msg = f"Navigation did not return a response for {validated_url}"
                        raise RetryableApiError(msg)

                    status = response.status
                    if status >= 400:
                        message = f"HTTP {status} for {validated_url}"
                        if is_retryable_http_status(status):
                            raise classify_http_status(status, message=message)
                        raise WebFetchError(message)

                    self._wait_for_rendered_page(page)
                    return RenderedPage(
                        url=page.url,
                        title=page.title(),
                        html=page.content(),
                    )
                finally:
                    browser.close()
        except (WebFetchError, RetryableApiError):
            raise
        except PlaywrightTimeoutError as exc:
            msg = f"Timed out loading {validated_url}"
            raise RetryableApiError(msg) from exc
        except PlaywrightError as exc:
            msg = f"Browser failed to load {validated_url}: {exc}"
            if is_retryable_playwright_error(exc):
                raise RetryableApiError(msg) from exc
            raise WebFetchError(msg) from exc

    def _wait_for_rendered_page(self, page) -> None:
        """Wait for load, network idle, and optional lazy-render delay.

        Args:
            page: Active Playwright page.
        """
        try:
            page.wait_for_load_state("load", timeout=self._timeout_ms)
        except PlaywrightTimeoutError:
            logger.warning("Load event timed out; continuing with rendered DOM")

        if self._network_idle_timeout_ms > 0:
            try:
                page.wait_for_load_state(
                    "networkidle",
                    timeout=self._network_idle_timeout_ms,
                )
            except PlaywrightTimeoutError:
                logger.debug("Network idle wait timed out; continuing")

        if self._post_load_wait_ms > 0:
            page.wait_for_timeout(self._post_load_wait_ms)


def is_retryable_playwright_error(exc: PlaywrightError) -> bool:
    """Return whether a Playwright failure is likely transient.

    Args:
        exc: Playwright error raised during navigation or rendering.

    Returns:
        True when a retry may succeed.
    """
    message = str(exc).lower()
    return any(hint in message for hint in _TRANSIENT_PLAYWRIGHT_HINTS)
