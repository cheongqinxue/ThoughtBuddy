"""Logging bootstrap for ThoughtBuddy."""

from __future__ import annotations

import logging
import sys

from config import Settings
from utils.logging.context import WakeContextFilter
from utils.logging.filters import RedactionFilter
from utils.observability import (
    configure_langfuse_environment,
    install_langfuse_log_handler,
    log_langfuse_startup_status,
)

_THIRD_PARTY_LOGGERS = (
    "httpx",
    "httpcore",
    "chromadb",
    "playwright",
    "urllib3",
    "asyncio",
    # LangGraph dev hot-reload spams INFO on every filesystem touch (DB, logs, …).
    "watchfiles",
    "langgraph_api",
    "langgraph_runtime",
    "langgraph_runtime_inmem",
)


class _ContextFormatter(logging.Formatter):
    """Formatter that injects optional wake/thread suffixes."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a record with optional wake context fields.

        Args:
            record: Log record to format.

        Returns:
            Formatted log line.
        """
        wake_id = getattr(record, "wake_id", None)
        thread_id = getattr(record, "thread_id", None)
        record.wake_id = f" wake={wake_id}" if wake_id else ""
        record.thread_id = f" thread={thread_id}" if thread_id else ""
        return super().format(record)


def configure_observability(settings: Settings) -> None:
    """Configure application logging from settings.

    Single entry point for console logging, filters, and third-party verbosity.

    Args:
        settings: Application settings.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.addFilter(RedactionFilter())
    handler.addFilter(WakeContextFilter())
    handler.setFormatter(
        _ContextFormatter(
            "%(asctime)s %(levelname)s %(name)s%(wake_id)s%(thread_id)s: %(message)s",
        ),
    )
    root.addHandler(handler)

    logging.getLogger("thoughtbuddy").setLevel(level)

    third_party_level = logging.DEBUG if level <= logging.DEBUG else logging.WARNING
    for logger_name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(third_party_level)

    configure_langfuse_environment(settings)
    install_langfuse_log_handler(settings)
    log_langfuse_startup_status(settings)


def configure_logging(settings: Settings) -> None:
    """Backward-compatible alias for :func:`configure_observability`.

    Args:
        settings: Application settings.
    """
    configure_observability(settings)
