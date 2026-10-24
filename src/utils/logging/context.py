"""Wake-scoped logging context for ThoughtBuddy."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_wake_id: ContextVar[str | None] = ContextVar("wake_id", default=None)
_thread_id: ContextVar[str | None] = ContextVar("thread_id", default=None)


class WakeContextFilter(logging.Filter):
    """Inject wake and thread identifiers into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach context fields when a wake scope is active.

        Args:
            record: Log record about to be emitted.

        Returns:
            Always True so the record is not dropped.
        """
        record.wake_id = _wake_id.get()
        record.thread_id = _thread_id.get()
        return True


@contextmanager
def wake_logging_context(*, wake_id: str, thread_id: str) -> Iterator[None]:
    """Bind wake identifiers for downstream log records.

    Args:
        wake_id: Unique identifier for the current wake cycle.
        thread_id: Conversation thread identifier.

    Yields:
        Nothing; restores prior context on exit.
    """
    wake_token = _wake_id.set(wake_id)
    thread_token = _thread_id.set(thread_id)
    try:
        yield
    finally:
        _wake_id.reset(wake_token)
        _thread_id.reset(thread_token)
