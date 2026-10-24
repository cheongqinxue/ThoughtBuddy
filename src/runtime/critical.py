"""Critical wake failures that must halt the runtime instead of retrying."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Settings

_CONTEXT_LENGTH_RE = re.compile(
    r"maximum context length|context length is \d+|input_tokens",
    re.IGNORECASE,
)

_LLM_OUTAGE_TYPE_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "OpenAIConnectionError",
        "OpenAITimeoutError",
        "OpenAIRateLimitError",
        "ModelConnectionError",
        "ModelTimeoutError",
        "ModelRateLimitError",
    },
)

_LLM_OUTAGE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class CriticalWakeError(Exception):
    """Unrecoverable wake failure — retrying will not help until operator action."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        """Initialize a critical wake error.

        Args:
            message: Operator-facing explanation.
            cause: Optional underlying exception.
        """
        super().__init__(message)
        self.cause = cause


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    """Walk an exception and its causes/contexts without cycles.

    Args:
        exc: Root exception.

    Returns:
        Ordered chain from root outward.
    """
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def is_context_length_error(exc: BaseException) -> bool:
    """Return whether an exception indicates a model context window overflow.

    Args:
        exc: Exception raised during wake or LLM invocation.

    Returns:
        True when the error message matches known context-length failures.
    """
    for current in _iter_exception_chain(exc):
        if _CONTEXT_LENGTH_RE.search(str(current)):
            return True
    return False


def is_llm_outage_error(exc: BaseException) -> bool:
    """Return whether an exception indicates inference is unavailable.

    LangChain ``ChatOpenAI`` retries transient errors up to ``llm.max_retries``
    before raising. Errors matching this check have already exhausted retries.

    Args:
        exc: Exception raised during wake or LLM invocation.

    Returns:
        True for connection failures, timeouts, and retryable HTTP statuses.
    """
    for current in _iter_exception_chain(exc):
        type_name = type(current).__name__
        if type_name in _LLM_OUTAGE_TYPE_NAMES:
            if type_name == "APIStatusError":
                status_code = getattr(current, "status_code", None)
                if status_code not in _LLM_OUTAGE_STATUS_CODES:
                    continue
            return True

        # httpx errors from direct API calls (e.g. token counting)
        if type_name in {"ConnectError", "ReadTimeout", "ConnectTimeout", "NetworkError"}:
            return True

    return False


def classify_wake_error(
    exc: BaseException,
    settings: Settings | None = None,
) -> CriticalWakeError | None:
    """Map a wake exception to a critical failure when retry is pointless.

    Args:
        exc: Exception raised during wake.
        settings: Optional settings for operator-facing LLM outage messages.

    Returns:
        CriticalWakeError when the failure is unrecoverable without operator action.
    """
    if isinstance(exc, CriticalWakeError):
        return exc
    if is_context_length_error(exc):
        return CriticalWakeError(
            "Model context window exceeded. Reduce archive size, lower "
            "llm.max_tokens, or tune context.compact_token_threshold / "
            "context.compact_token_ratio so compaction runs sooner.",
            cause=exc,
        )
    if is_llm_outage_error(exc):
        attempts = (settings.llm_max_retries + 1) if settings is not None else None
        base_url = settings.openai_base_url if settings is not None else "the inference server"
        attempt_text = (
            f"after {attempts} client attempt(s) including retries"
            if attempts is not None
            else "after client retries were exhausted"
        )
        return CriticalWakeError(
            f"LLM inference unavailable {attempt_text}. Check vLLM at {base_url}.",
            cause=exc,
        )
    return None
