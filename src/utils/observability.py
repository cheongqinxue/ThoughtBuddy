"""Observability helpers for ThoughtBuddy graph invocations (v1: Langfuse)."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgraph.graph.state import CompiledStateGraph

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger("thoughtbuddy.utils.observability")

_LANGFUSE_LEVELS = {
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "ERROR",
}


@dataclass(frozen=True)
class LangfuseInvokeTrace:
    """Active Langfuse instrumentation for one graph invocation."""

    handler: Any
    callbacks: list[Any]
    run_name: str
    tags: tuple[str, ...]

    def invoke_config(
        self,
        thread_id: str,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build RunnableConfig fields for a traced graph invocation.

        Args:
            thread_id: LangGraph checkpoint thread identifier.
            extra_metadata: Optional metadata merged into Langfuse fields.

        Returns:
            Config mapping with callbacks and Langfuse metadata.
        """
        metadata: dict[str, Any] = {
            "langfuse_session_id": thread_id,
            "langfuse_tags": list(self.tags),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return {
            "configurable": {"thread_id": thread_id},
            "callbacks": self.callbacks,
            "run_name": self.run_name,
            "metadata": metadata,
        }


LangfuseWakeTrace = LangfuseInvokeTrace

_STUDIO_CALLBACK_HANDLER: Any | None = None


def is_langfuse_configured() -> bool:
    """Return whether Langfuse credentials are present in the environment."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def is_langfuse_enabled(settings: Settings) -> bool:
    """Return whether Langfuse tracing should run for this process.

    Args:
        settings: Application settings.

    Returns:
        True when tracing is enabled and credentials are configured.
    """
    return settings.langfuse_enabled and is_langfuse_configured()


def configure_langfuse_environment(settings: Settings) -> None:
    """Apply non-secret Langfuse settings to the process environment.

    Args:
        settings: Application settings.
    """
    if settings.langfuse_base_url:
        os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_base_url)


def create_callback_handler(*, trace_id: str | None = None) -> Any:
    """Create a Langfuse LangChain callback handler.

    Args:
        trace_id: Optional explicit trace id. When omitted, Langfuse assigns one.

    Returns:
        Langfuse ``CallbackHandler`` instance.
    """
    from langfuse.langchain import CallbackHandler

    if trace_id is None:
        return CallbackHandler()
    return CallbackHandler(trace_context={"trace_id": normalize_trace_id(trace_id)})


def _studio_callback_handler() -> Any:
    """Return a process-wide callback handler for LangGraph Studio invocations."""
    global _STUDIO_CALLBACK_HANDLER
    if _STUDIO_CALLBACK_HANDLER is None:
        _STUDIO_CALLBACK_HANDLER = create_callback_handler()
    return _STUDIO_CALLBACK_HANDLER


def apply_studio_langfuse_config(
    graph: CompiledStateGraph,
    settings: Settings,
) -> CompiledStateGraph:
    """Attach default Langfuse callbacks for LangGraph Studio invocations.

    The runtime wake path uses ``create_agent()`` without this wrapper so queue
    wakes can supply their own trace ids and metadata per invocation.

    Args:
        graph: Compiled ThoughtBuddy graph.
        settings: Application settings.

    Returns:
        The same graph instance, optionally wrapped with Langfuse defaults.
    """
    if not is_langfuse_enabled(settings):
        return graph

    return graph.with_config(
        {
            "callbacks": [_studio_callback_handler()],
            "run_name": "thoughtbuddy-studio",
            "metadata": {
                "langfuse_tags": ["thoughtbuddy", "studio"],
            },
        },
    )


def normalize_trace_id(trace_id: str) -> str:
    """Normalize an application trace id for Langfuse/OpenTelemetry.

    Args:
        trace_id: Wake or request identifier.

    Returns:
        Lowercase hex trace id without separators when possible.
    """
    cleaned = trace_id.replace("-", "").lower()
    if len(cleaned) == 32 and all(char in "0123456789abcdef" for char in cleaned):
        return cleaned
    return trace_id


@contextmanager
def _langfuse_invoke_trace(
    settings: Settings,
    *,
    trace_id: str | None,
    thread_id: str,
    trace_name: str,
    run_name: str,
    tags: tuple[str, ...],
    metadata: dict[str, Any],
) -> Iterator[LangfuseInvokeTrace | None]:
    """Open a Langfuse trace scope for one graph invocation.

    Args:
        settings: Application settings.
        trace_id: Optional explicit Langfuse trace id.
        thread_id: Conversation thread identifier.
        trace_name: Top-level Langfuse trace name.
        run_name: LangChain run name for the graph invocation.
        tags: Langfuse tags applied to the trace.
        metadata: Additional trace metadata.

    Yields:
        Active trace helpers, or ``None`` when Langfuse is disabled.
    """
    if not is_langfuse_enabled(settings):
        yield None
        return

    from langfuse import propagate_attributes

    handler = create_callback_handler(trace_id=trace_id)
    trace = LangfuseInvokeTrace(
        handler=handler,
        callbacks=[handler],
        run_name=run_name,
        tags=tags,
    )
    with propagate_attributes(
        trace_name=trace_name,
        session_id=thread_id,
        tags=list(tags),
        metadata=metadata,
    ):
        yield trace


@contextmanager
def langfuse_wake_trace(
    settings: Settings,
    *,
    wake_id: str,
    thread_id: str,
    task_ids: list[str],
) -> Iterator[LangfuseWakeTrace | None]:
    """Open a Langfuse trace scope for one wake cycle.

    Args:
        settings: Application settings.
        wake_id: Unique wake identifier used as the Langfuse trace id.
        thread_id: Conversation thread identifier.
        task_ids: Queue task ids processed in this wake.

    Yields:
        Active trace helpers, or ``None`` when Langfuse is disabled.
    """
    metadata = {
        "wake_id": wake_id,
        "thread_id": thread_id,
        "task_ids": task_ids,
        "pending_task_count": len(task_ids),
    }
    with _langfuse_invoke_trace(
        settings,
        trace_id=wake_id,
        thread_id=thread_id,
        trace_name="thoughtbuddy-wake",
        run_name="thoughtbuddy-wake",
        tags=("thoughtbuddy", "wake"),
        metadata=metadata,
    ) as trace:
        yield trace


@contextmanager
def langfuse_direct_trace(
    settings: Settings,
    *,
    thread_id: str,
    invoke_id: str | None = None,
) -> Iterator[LangfuseInvokeTrace | None]:
    """Open a Langfuse trace scope for a direct graph invocation.

    Used by helper entrypoints such as ``run_agent`` that bypass the task queue.

    Args:
        settings: Application settings.
        thread_id: Conversation thread identifier.
        invoke_id: Optional explicit invoke identifier used as the trace id.

    Yields:
        Active trace helpers, or ``None`` when Langfuse is disabled.
    """
    resolved_invoke_id = invoke_id or str(uuid.uuid4())
    metadata = {
        "invoke_id": resolved_invoke_id,
        "thread_id": thread_id,
        "source": "direct",
    }
    with _langfuse_invoke_trace(
        settings,
        trace_id=resolved_invoke_id,
        thread_id=thread_id,
        trace_name="thoughtbuddy-direct",
        run_name="thoughtbuddy-direct",
        tags=("thoughtbuddy", "direct"),
        metadata=metadata,
    ) as trace:
        yield trace


def flush_langfuse(settings: Settings) -> None:
    """Flush buffered Langfuse observations.

    Args:
        settings: Application settings.
    """
    if not is_langfuse_enabled(settings):
        return

    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        logger.exception("langfuse_flush_failed")


def delete_langfuse_trace(settings: Settings, trace_id: str) -> bool:
    """Queue deletion of one Langfuse trace by id.

    Langfuse removes tracing data asynchronously after the API accepts the
    request. Integration and smoke tests call this so temporary traces are not
    kept indefinitely.

    Args:
        settings: Application settings.
        trace_id: Langfuse trace identifier (UUID with or without dashes).

    Returns:
        True when the delete API call succeeds.
    """
    if not is_langfuse_enabled(settings):
        return False

    configure_langfuse_environment(settings)
    normalized_trace_id = normalize_trace_id(trace_id)

    try:
        from langfuse import get_client

        get_client().api.trace.delete(normalized_trace_id)
        logger.debug("langfuse_trace_deleted trace_id=%s", normalized_trace_id)
        return True
    except Exception:
        logger.exception("langfuse_trace_delete_failed trace_id=%s", normalized_trace_id)
        return False


def shutdown_langfuse(settings: Settings) -> None:
    """Flush and shut down the Langfuse client.

    Args:
        settings: Application settings.
    """
    if not is_langfuse_enabled(settings):
        return

    try:
        from langfuse import get_client

        get_client().shutdown()
    except Exception:
        logger.exception("langfuse_shutdown_failed")


class LangfuseLogHandler(logging.Handler):
    """Forward ``thoughtbuddy`` WARNING+ logs to the active Langfuse span."""

    def __init__(self, *, min_level: int = logging.WARNING) -> None:
        """Initialize the handler.

        Args:
            min_level: Minimum Python log level forwarded to Langfuse.
        """
        super().__init__(level=min_level)
        self._min_level = min_level

    def emit(self, record: logging.LogRecord) -> None:
        """Attach a log line to the current Langfuse observation when available.

        Args:
            record: Log record to forward.
        """
        if record.levelno < self._min_level:
            return

        try:
            from langfuse import get_client

            message = self.format(record)
            level = _LANGFUSE_LEVELS.get(record.levelno, "DEFAULT")
            metadata = {
                "logger": record.name,
                "levelname": record.levelname,
            }
            wake_id = getattr(record, "wake_id", None)
            thread_id = getattr(record, "thread_id", None)
            if wake_id:
                metadata["wake_id"] = wake_id
            if thread_id:
                metadata["thread_id"] = thread_id

            get_client().update_current_span(
                level=level,
                status_message=message[:1000],
                metadata=metadata,
            )
        except Exception:
            return


def install_langfuse_log_handler(settings: Settings) -> LangfuseLogHandler | None:
    """Attach Langfuse log forwarding to the ``thoughtbuddy`` logger.

    Args:
        settings: Application settings.

    Returns:
        Installed handler, or ``None`` when Langfuse is disabled.
    """
    if not is_langfuse_enabled(settings):
        return None

    min_level = getattr(logging, settings.langfuse_log_min_level.upper(), logging.WARNING)
    handler = LangfuseLogHandler(min_level=min_level)
    thoughtbuddy_logger = logging.getLogger("thoughtbuddy")
    thoughtbuddy_logger.addHandler(handler)
    return handler


def log_langfuse_startup_status(settings: Settings) -> None:
    """Log whether Langfuse tracing is active for this process.

    Args:
        settings: Application settings.
    """
    if not settings.langfuse_enabled:
        logger.debug("Langfuse tracing disabled via config")
        return

    if not is_langfuse_configured():
        logger.warning(
            "Langfuse tracing enabled in config but LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY are missing; tracing will stay off",
        )
        return

    logger.info("Langfuse tracing enabled")
