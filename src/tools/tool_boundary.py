"""Tool execution boundary — logging and recoverable error payloads for the agent."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from langchain_core.tools import BaseTool, StructuredTool

from utils.api_retry import RetryableApiError

T = TypeVar("T")

ToolFunc = Callable[..., T]
AsyncToolFunc = Callable[..., Awaitable[T]]

BOUNDARY_WRAPPED_ATTR = "_thoughtbuddy_boundary_wrapped"

_RECOVERABLE_MODULE_PREFIXES = (
    "tools.",
    "experiments.",
    "utils.",
    "store.",
    "knowledge.",
    "plans.",
    "world.",
    "history.",
    "scheduling.",
)


def tool_error_payload(
    tool_name: str,
    exc: Exception,
    *,
    retryable: bool = False,
) -> str:
    """Format a recoverable tool failure for the LLM.

    Args:
        tool_name: Agent-visible tool name.
        exc: Failure that was handled at the tool boundary.
        retryable: Whether the agent may succeed by retrying later.

    Returns:
        JSON string describing the failure.
    """
    return json.dumps(
        {
            "ok": False,
            "tool": tool_name,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "retryable": retryable,
        },
        indent=2,
    )


def is_retryable_tool_error(exc: Exception) -> bool:
    """Return whether a tool error is likely ephemeral.

    Args:
        exc: Exception raised while executing a tool.

    Returns:
        True for network-style failures the agent may retry.
    """
    return isinstance(exc, RetryableApiError)


def is_recoverable_tool_error(exc: Exception) -> bool:
    """Return whether a tool failure should be returned to the agent.

    Args:
        exc: Exception raised while executing a tool.

    Returns:
        True when the agent can plausibly recover by adjusting its next action.
    """
    return is_retryable_tool_error(exc) or _is_recoverable_domain_error(exc)


def is_tool_boundary_wrapped(tool: BaseTool) -> bool:
    """Return whether a tool already has the boundary wrapper applied.

    Args:
        tool: LangChain tool instance.

    Returns:
        True when ``wrap_tool_with_boundary`` has already wrapped the tool.
    """
    return getattr(tool, BOUNDARY_WRAPPED_ATTR, False)


def run_tool_boundary(tool_name: str, func: ToolFunc, /, *args: Any, **kwargs: Any) -> Any:
    """Execute a tool with consistent logging and recoverable error returns.

    INFO logs summarize the call; ERROR logs include full tracebacks for
    recoverable external failures before returning a structured payload to the LLM.

    Args:
        tool_name: Agent-visible tool name.
        func: Tool implementation callable.
        *args: Positional arguments for ``func``.
        **kwargs: Keyword arguments for ``func``.

    Returns:
        Tool result, or a JSON error payload for recoverable failures.

    Raises:
        Exception: Unexpected failures propagate after logging.
    """
    log = logging.getLogger(f"thoughtbuddy.tools.{tool_name}")
    arg_summary = _summarize_args(args, kwargs)
    log.info("tool_start %s", arg_summary)
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        if is_recoverable_tool_error(exc):
            log.error("tool_failed_recoverable", exc_info=exc, extra={"tool": tool_name})
            return tool_error_payload(
                tool_name,
                exc,
                retryable=is_retryable_tool_error(exc),
            )
        log.exception("tool_failed_unexpected", extra={"tool": tool_name})
        raise

    log.info("tool_ok result_chars=%s", len(str(result)))
    return result


async def run_tool_boundary_async(
    tool_name: str,
    func: AsyncToolFunc,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Async variant of :func:`run_tool_boundary`.

    Args:
        tool_name: Agent-visible tool name.
        func: Async tool implementation callable.
        *args: Positional arguments for ``func``.
        **kwargs: Keyword arguments for ``func``.

    Returns:
        Tool result, or a JSON error payload for recoverable failures.

    Raises:
        Exception: Unexpected failures propagate after logging.
    """
    log = logging.getLogger(f"thoughtbuddy.tools.{tool_name}")
    arg_summary = _summarize_args(args, kwargs)
    log.info("tool_start %s", arg_summary)
    try:
        result = await func(*args, **kwargs)
    except Exception as exc:
        if is_recoverable_tool_error(exc):
            log.error("tool_failed_recoverable", exc_info=exc, extra={"tool": tool_name})
            return tool_error_payload(
                tool_name,
                exc,
                retryable=is_retryable_tool_error(exc),
            )
        log.exception("tool_failed_unexpected", extra={"tool": tool_name})
        raise

    log.info("tool_ok result_chars=%s", len(str(result)))
    return result


def wrap_tool_with_boundary(tool: BaseTool) -> BaseTool:
    """Wrap a LangChain tool so execution uses :func:`run_tool_boundary`.

    Args:
        tool: Tool returned from a domain factory.

    Returns:
        The same tool instance when already wrapped, otherwise a wrapped copy.
    """
    if is_tool_boundary_wrapped(tool):
        return tool

    tool_name = tool.name

    if isinstance(tool, StructuredTool) and tool.func is not None:
        original = tool.func

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return run_tool_boundary(tool_name, original, *args, **kwargs)

        wrapped.__name__ = getattr(original, "__name__", f"{tool_name}_wrapped")
        wrapped.__doc__ = getattr(original, "__doc__", None)
        new_tool = tool.model_copy(update={"func": wrapped})
        setattr(new_tool, BOUNDARY_WRAPPED_ATTR, True)
        return new_tool

    if isinstance(tool, StructuredTool) and tool.coroutine is not None:
        original = tool.coroutine

        async def wrapped_async(*args: Any, **kwargs: Any) -> Any:
            return await run_tool_boundary_async(tool_name, original, *args, **kwargs)

        wrapped_async.__name__ = getattr(original, "__name__", f"{tool_name}_wrapped_async")
        wrapped_async.__doc__ = getattr(original, "__doc__", None)
        new_tool = tool.model_copy(update={"coroutine": wrapped_async})
        setattr(new_tool, BOUNDARY_WRAPPED_ATTR, True)
        return new_tool

    msg = f"Cannot apply tool boundary to {tool_name!r}: unsupported tool type {type(tool)!r}"
    raise TypeError(msg)


def _is_recoverable_domain_error(exc: Exception) -> bool:
    """Return whether a domain-specific tool error should return to the LLM."""
    if isinstance(exc, ValueError):
        return True

    name = type(exc).__name__
    module = type(exc).__module__
    return name.endswith("Error") and module.startswith(_RECOVERABLE_MODULE_PREFIXES)


def _summarize_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Build a concise, log-safe summary of tool arguments.

    Args:
        args: Positional tool arguments.
        kwargs: Keyword tool arguments.

    Returns:
        Short summary string for INFO logs.
    """
    parts: list[str] = []
    for value in args:
        parts.append(_summarize_value(value))
    for key, value in kwargs.items():
        parts.append(f"{key}={_summarize_value(value)}")
    return ", ".join(parts) if parts else "(no args)"


def _summarize_value(value: Any) -> str:
    """Summarize one argument value without logging large payloads."""
    text = str(value)
    if len(text) > 120:
        return f"{text[:117]}..."
    return text
