"""Shared retry utilities for external API calls."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import httpx
from tenacity import (
    Retrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger("thoughtbuddy.utils.api_retry")

T = TypeVar("T")

RETRYABLE_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
NON_RETRYABLE_HTTP_STATUS_CODES = frozenset({400, 401, 403, 404, 405, 409, 422})

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_MIN_WAIT_SECONDS = 1.0
DEFAULT_MAX_WAIT_SECONDS = 30.0


class ApiClientError(Exception):
    """Non-retryable API error caused by invalid client usage or auth."""


class RetryableApiError(Exception):
    """Transient API error that may succeed on retry."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        """Initialize a retryable API error.

        Args:
            message: Human-readable error description.
            status_code: Optional HTTP status code associated with the failure.
        """
        super().__init__(message)
        self.status_code = status_code


def is_retryable_http_status(status_code: int) -> bool:
    """Return whether an HTTP status code represents a transient failure.

    Args:
        status_code: HTTP response status code.

    Returns:
        True when the status may clear on retry.
    """
    return status_code in RETRYABLE_HTTP_STATUS_CODES


def classify_http_status(status_code: int, *, message: str) -> ApiClientError | RetryableApiError:
    """Map an HTTP status code to a retryable or client-side API error.

    Args:
        status_code: HTTP response status code.
        message: Human-readable error description.

    Returns:
        Retryable or non-retryable API exception.
    """
    if is_retryable_http_status(status_code):
        return RetryableApiError(message, status_code=status_code)
    return ApiClientError(message)


def is_retryable_exception(exc: BaseException) -> bool:
    """Return whether an exception represents a transient API failure.

    Logic errors, validation failures, and explicit client errors fail immediately.
    Only network blips and known transient HTTP statuses are retried.

    Args:
        exc: Exception raised by an API call.

    Returns:
        True when the error may succeed if retried.
    """
    if isinstance(exc, (ApiClientError, ValueError, TypeError, KeyError)):
        return False
    if isinstance(exc, RetryableApiError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_HTTP_STATUS_CODES
    return isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.CloseError,
            httpx.RemoteProtocolError,
            httpx.ProtocolError,
        ),
    )


def http_status_error(exc: httpx.HTTPStatusError) -> ApiClientError | RetryableApiError:
    """Convert an HTTP status error into a retry classification.

    Args:
        exc: HTTP status error from httpx.

    Returns:
        Retryable or non-retryable API exception.
    """
    status_code = exc.response.status_code
    message = f"HTTP {status_code} from {exc.request.url}"
    return classify_http_status(status_code, message=message)


def create_api_retry(
    settings: Settings | None = None,
    *,
    max_attempts: int | None = None,
    min_wait_seconds: float | None = None,
    max_wait_seconds: float | None = None,
    log_each_attempt: bool | None = None,
    operation_name: str = "API call",
) -> ApiRetry:
    """Build a shared ``ApiRetry`` instance from settings or explicit overrides.

    Args:
        settings: Application settings. When provided, unset retry kwargs are read
            from ``logging.retry`` fields on settings.
        max_attempts: Maximum number of attempts including the first call.
        min_wait_seconds: Minimum wait between retries.
        max_wait_seconds: Maximum wait between retries.
        log_each_attempt: When True, log each retry at DEBUG. Defaults to settings
            or False.
        operation_name: Label included in exhaustion logs.

    Returns:
        Configured retry helper.
    """
    resolved_max_attempts = max_attempts
    resolved_min_wait = min_wait_seconds
    resolved_max_wait = max_wait_seconds
    resolved_log_each = log_each_attempt
    if settings is not None:
        if resolved_max_attempts is None:
            resolved_max_attempts = settings.retry_max_attempts
        if resolved_min_wait is None:
            resolved_min_wait = settings.retry_min_wait_seconds
        if resolved_max_wait is None:
            resolved_max_wait = settings.retry_max_wait_seconds
        if resolved_log_each is None:
            resolved_log_each = settings.retry_log_each_attempt

    return ApiRetry(
        max_attempts=resolved_max_attempts or DEFAULT_MAX_ATTEMPTS,
        min_wait_seconds=resolved_min_wait or DEFAULT_MIN_WAIT_SECONDS,
        max_wait_seconds=resolved_max_wait or DEFAULT_MAX_WAIT_SECONDS,
        log_each_attempt=resolved_log_each if resolved_log_each is not None else False,
        operation_name=operation_name,
    )


class ApiRetry:
    """Tenacity-backed retry helper for external API calls."""

    def __init__(
        self,
        *,
        max_attempts: int = 4,
        min_wait_seconds: float = 1.0,
        max_wait_seconds: float = 30.0,
        log_each_attempt: bool = False,
        operation_name: str = "API call",
    ) -> None:
        """Initialize retry settings.

        Args:
            max_attempts: Maximum number of attempts including the first call.
            min_wait_seconds: Minimum wait between retries.
            max_wait_seconds: Maximum wait between retries.
            log_each_attempt: Log each retry attempt at DEBUG when True.
            operation_name: Label included in exhaustion logs.
        """
        self._max_attempts = max_attempts
        self._log_each_attempt = log_each_attempt
        self._operation_name = operation_name
        self._retrying = Retrying(
            retry=retry_if_exception(is_retryable_exception),
            wait=wait_exponential_jitter(
                initial=min_wait_seconds,
                max=max_wait_seconds,
            ),
            stop=stop_after_attempt(max_attempts),
            before_sleep=self._before_sleep if log_each_attempt else None,
            reraise=True,
        )

    def _before_sleep(self, retry_state: RetryCallState) -> None:
        """Log an upcoming retry attempt at DEBUG."""
        attempt = retry_state.attempt_number
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.debug(
            "%s retry %s/%s after %s: %s",
            self._operation_name,
            attempt,
            self._max_attempts,
            retry_state.next_action.sleep if retry_state.next_action else 0,
            exc,
        )

    def call(self, func: Callable[[], T]) -> T:
        """Execute a callable with retry for transient API failures.

        Retries are silent by default. When all attempts fail, logs once at ERROR
        with the full traceback, then re-raises the underlying exception.

        Args:
            func: Zero-argument callable that performs the API request.

        Returns:
            Callable result from the first successful attempt.

        Raises:
            ApiClientError: For non-retryable client-side API failures.
            RetryableApiError: If retries are exhausted for a transient failure.
            httpx.HTTPError: If retries are exhausted for a transport failure.
        """
        try:
            return self._retrying(func)
        except Exception as exc:
            if is_retryable_exception(exc):
                logger.error(
                    "%s failed after %s attempts",
                    self._operation_name,
                    self._max_attempts,
                    exc_info=exc,
                )
            raise
