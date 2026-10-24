"""LangSearch provider for the websearch tool."""

from __future__ import annotations

from typing import Any

import httpx

from config import Settings
from tools.websearch.backend import Freshness
from utils.api_retry import (
    ApiClientError,
    ApiRetry,
    RetryableApiError,
    classify_http_status,
    create_api_retry,
    http_status_error,
    is_retryable_http_status,
)

LANGSEARCH_WEB_SEARCH_URL = "https://api.langsearch.com/v1/web-search"
NON_RETRYABLE_API_CODES = frozenset({400, 401, 403, 404, 422})


class LangSearchError(Exception):
    """Raised when the LangSearch API returns a non-retryable error response."""


class LangSearchClient:
    """HTTP client for the LangSearch Web Search API."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 30.0,
        retry: ApiRetry | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: LangSearch API key.
            timeout: Request timeout in seconds.
            retry: Optional retry helper. Defaults to ``ApiRetry()``.
        """
        self._api_key = api_key
        self._timeout = timeout
        self._retry = retry or ApiRetry(operation_name="LangSearch web search")

    def search(
        self,
        query: str,
        *,
        count: int = 5,
        summary: bool = True,
        freshness: Freshness = "noLimit",
    ) -> dict[str, Any]:
        """Execute a web search request.

        Args:
            query: User search query.
            count: Number of results to return.
            summary: Whether to request long-text summaries.
            freshness: Time range filter for results.

        Returns:
            Normalized SearchResponse-shaped dict.

        Raises:
            LangSearchError: If the API request fails with a non-retryable error.
            RetryableApiError: If retries are exhausted for a transient failure.
        """
        payload = {
            "query": query,
            "count": count,
            "summary": summary,
            "freshness": freshness,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        def perform_request() -> dict[str, Any]:
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        LANGSEARCH_WEB_SEARCH_URL,
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                mapped = http_status_error(exc)
                raise mapped from exc
            except httpx.HTTPError as exc:
                msg = f"LangSearch request failed: {exc}"
                raise RetryableApiError(msg) from exc

            body = response.json()
            return extract_search_data(body)

        try:
            return self._retry.call(perform_request)
        except ApiClientError as exc:
            raise LangSearchError(str(exc)) from exc


def extract_search_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the SearchResponse payload from a LangSearch API body.

    Args:
        payload: Raw JSON response from LangSearch.

    Returns:
        Search response object containing ``webPages``.

    Raises:
        LangSearchError: If the payload does not contain usable search data.
    """
    if "data" in payload:
        code = payload.get("code")
        if code != 200:
            message = payload.get("msg") or "Unknown LangSearch API error"
            error_message = f"LangSearch API error ({code}): {message}"
            if code in NON_RETRYABLE_API_CODES:
                raise ApiClientError(error_message)
            if is_retryable_http_status(code):
                raise classify_http_status(code, message=error_message)
            raise LangSearchError(error_message)
        data = payload["data"]
        if not isinstance(data, dict):
            msg = "LangSearch API returned an invalid data payload."
            raise ApiClientError(msg)
        return data

    if payload.get("_type") == "SearchResponse":
        return payload

    msg = "LangSearch API returned an unexpected response shape."
    raise ApiClientError(msg)


class LangSearchBackend:
    """WebSearchBackend implementation backed by LangSearch."""

    def __init__(self, settings: Settings) -> None:
        """Create a LangSearch backend from application settings.

        Args:
            settings: Application settings (``langsearch_api_key`` required).
        """
        self._client = LangSearchClient(
            settings.langsearch_api_key,
            retry=create_api_retry(settings, operation_name="LangSearch web search"),
        )

    def search(
        self,
        query: str,
        *,
        count: int = 5,
        freshness: Freshness = "noLimit",
    ) -> dict[str, Any]:
        """Run a web search via LangSearch.

        Args:
            query: User search query.
            count: Number of results to return.
            freshness: Time range filter for results.

        Returns:
            Normalized SearchResponse-shaped dict.
        """
        return self._client.search(query, count=count, summary=True, freshness=freshness)
