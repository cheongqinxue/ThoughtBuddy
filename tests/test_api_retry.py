"""Tests for shared API retry utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx
import pytest

from config import Settings

from utils.api_retry import (
    RETRYABLE_HTTP_STATUS_CODES,
    ApiClientError,
    ApiRetry,
    RetryableApiError,
    create_api_retry,
    http_status_error,
    is_retryable_exception,
)

RETRYABLE_STATUS_CODES = sorted(RETRYABLE_HTTP_STATUS_CODES)
NON_RETRYABLE_STATUS_CODES = [400, 401, 403, 404, 405, 409, 422]
MAX_ATTEMPTS = 4


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    """Build an httpx status error for a given HTTP status code.

    Args:
        status_code: HTTP status code to simulate.

    Returns:
        Configured ``HTTPStatusError`` instance.
    """
    request = httpx.Request("POST", "https://api.example.com/search")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"HTTP {status_code}", request=request, response=response)


def _retry_helper(*, max_attempts: int = MAX_ATTEMPTS) -> ApiRetry:
    """Return an ``ApiRetry`` configured for fast tests.

    Args:
        max_attempts: Maximum number of attempts including the first call.

    Returns:
        Fast-running retry helper for unit tests.
    """
    return ApiRetry(
        max_attempts=max_attempts,
        min_wait_seconds=0.01,
        max_wait_seconds=0.05,
    )


class TestIsRetryableException:
    """Retry classification tests."""

    def test_retryable_api_error_is_retryable(self) -> None:
        """Treat RetryableApiError as retryable."""
        assert is_retryable_exception(RetryableApiError("temporary"))

    def test_api_client_error_is_not_retryable(self) -> None:
        """Treat ApiClientError as non-retryable."""
        assert not is_retryable_exception(ApiClientError("bad request"))

    @pytest.mark.parametrize("status_code", RETRYABLE_STATUS_CODES)
    def test_retryable_http_status_codes(self, status_code: int) -> None:
        """Mark common transient HTTP status codes as retryable."""
        assert is_retryable_exception(_http_status_error(status_code))

    @pytest.mark.parametrize("status_code", NON_RETRYABLE_STATUS_CODES)
    def test_non_retryable_http_status_codes(self, status_code: int) -> None:
        """Mark common client HTTP status codes as non-retryable."""
        assert not is_retryable_exception(_http_status_error(status_code))

    def test_connect_error_is_retryable(self) -> None:
        """Retry connection failures."""
        assert is_retryable_exception(httpx.ConnectError("connection reset"))

    def test_timeout_error_is_retryable(self) -> None:
        """Retry request timeouts."""
        assert is_retryable_exception(httpx.ReadTimeout("timed out"))

    def test_value_error_is_not_retryable(self) -> None:
        """Fail immediately for internal logic errors."""
        assert not is_retryable_exception(ValueError("bad payload"))


class TestHttpStatusErrorMapping:
    """HTTP status to exception mapping tests."""

    @pytest.mark.parametrize("status_code", RETRYABLE_STATUS_CODES)
    def test_maps_retryable_status_to_retryable_api_error(self, status_code: int) -> None:
        """Map transient HTTP statuses to RetryableApiError."""
        mapped = http_status_error(_http_status_error(status_code))

        assert isinstance(mapped, RetryableApiError)
        assert mapped.status_code == status_code

    @pytest.mark.parametrize("status_code", NON_RETRYABLE_STATUS_CODES)
    def test_maps_client_status_to_api_client_error(self, status_code: int) -> None:
        """Map client HTTP statuses to ApiClientError."""
        mapped = http_status_error(_http_status_error(status_code))

        assert isinstance(mapped, ApiClientError)


class TestApiRetryHttpStatusCodes:
    """ApiRetry behavior against simulated HTTP status codes."""

    @pytest.mark.parametrize("status_code", RETRYABLE_STATUS_CODES)
    def test_retries_until_attempts_are_exhausted(self, status_code: int) -> None:
        """Retry transient HTTP errors until max attempts are exhausted."""
        attempts = {"count": 0}

        def always_fails() -> None:
            attempts["count"] += 1
            raise _http_status_error(status_code)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            _retry_helper().call(always_fails)

        assert exc_info.value.response.status_code == status_code
        assert attempts["count"] == MAX_ATTEMPTS

    @pytest.mark.parametrize("status_code", NON_RETRYABLE_STATUS_CODES)
    def test_does_not_retry_client_http_errors(self, status_code: int) -> None:
        """Fail immediately for non-retryable HTTP client errors."""
        attempts = {"count": 0}

        def client_error() -> None:
            attempts["count"] += 1
            raise _http_status_error(status_code)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            _retry_helper().call(client_error)

        assert exc_info.value.response.status_code == status_code
        assert attempts["count"] == 1

    @pytest.mark.parametrize("status_code", RETRYABLE_STATUS_CODES)
    def test_recovers_after_transient_http_error(self, status_code: int) -> None:
        """Return successfully once a transient HTTP error clears."""
        attempts = {"count": 0}

        def fails_once() -> str:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise _http_status_error(status_code)
            return "recovered"

        result = _retry_helper().call(fails_once)

        assert result == "recovered"
        assert attempts["count"] == 2


class TestApiRetry:
    """General ApiRetry execution tests."""

    def test_retries_transient_failure_then_succeeds(self) -> None:
        """Retry a transient error and return the successful result."""
        attempts = {"count": 0}

        def flaky_call() -> str:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RetryableApiError("temporary outage", status_code=503)
            return "ok"

        result = _retry_helper(max_attempts=4).call(flaky_call)

        assert result == "ok"
        assert attempts["count"] == 3

    def test_does_not_retry_client_error(self) -> None:
        """Fail immediately for non-retryable client errors."""
        attempts = {"count": 0}

        def bad_call() -> str:
            attempts["count"] += 1
            raise ApiClientError("invalid payload")

        with pytest.raises(ApiClientError, match="invalid payload"):
            _retry_helper().call(bad_call)

        assert attempts["count"] == 1

    def test_raises_after_exhausting_retries(self) -> None:
        """Re-raise the last retryable error when attempts are exhausted."""

        def always_fails() -> str:
            raise RetryableApiError("still down", status_code=503)

        with pytest.raises(RetryableApiError, match="still down"):
            _retry_helper(max_attempts=3).call(always_fails)

    @pytest.mark.parametrize(
        ("exception_factory", "expected_attempts"),
        [
            (lambda: httpx.ConnectError("connection reset"), MAX_ATTEMPTS),
            (lambda: httpx.ReadTimeout("timed out"), MAX_ATTEMPTS),
        ],
    )
    def test_retries_transport_errors(
        self,
        exception_factory: Callable[[], Exception],
        expected_attempts: int,
    ) -> None:
        """Retry common network and timeout transport failures."""
        attempts = {"count": 0}

        def always_fails() -> None:
            attempts["count"] += 1
            raise exception_factory()

        with pytest.raises(type(exception_factory())):
            _retry_helper().call(always_fails)

        assert attempts["count"] == expected_attempts


class TestApiRetryLogging:
    """Retry logging behavior tests."""

    def test_logs_once_when_retries_are_exhausted(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Emit one ERROR log when all retry attempts fail."""
        attempts = {"count": 0}

        def always_fails() -> None:
            attempts["count"] += 1
            raise RetryableApiError("still down", status_code=503)

        with caplog.at_level(logging.ERROR, logger="thoughtbuddy.utils.api_retry"):
            with pytest.raises(RetryableApiError):
                _retry_helper(max_attempts=3).call(always_fails)

        error_records = [record for record in caplog.records if record.levelname == "ERROR"]
        assert len(error_records) == 1
        assert attempts["count"] == 3

    def test_does_not_log_error_when_retry_recovers(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Stay silent when a transient failure clears on retry."""
        attempts = {"count": 0}

        def flaky() -> str:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RetryableApiError("temporary", status_code=503)
            return "ok"

        with caplog.at_level(logging.ERROR, logger="thoughtbuddy.utils.api_retry"):
            result = _retry_helper(max_attempts=3).call(flaky)

        assert result == "ok"
        assert not [record for record in caplog.records if record.levelname == "ERROR"]

    def test_create_api_retry_reads_settings(self) -> None:
        """Build retry helpers from application settings."""
        settings = Settings(
            retry_log_each_attempt=True,
            retry_max_attempts=5,
            retry_min_wait_seconds=0.5,
            retry_max_wait_seconds=10.0,
        )

        retry = create_api_retry(settings, operation_name="demo")

        assert retry._log_each_attempt is True
        assert retry._max_attempts == 5
        assert retry._operation_name == "demo"
