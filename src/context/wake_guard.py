"""Per-wake context for archive-backed graph execution."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from config import Settings

_wake_guard: ContextVar["WakeGuardContext | None"] = ContextVar("wake_guard", default=None)


@dataclass(frozen=True)
class WakeGuardContext:
    """Archive and queue context shared across one wake-cycle graph run."""

    conn: sqlite3.Connection
    settings: Settings
    thread_id: str
    compaction_model: BaseChatModel
    pending_tasks: list[dict[str, Any]]
    tools: list[BaseTool]
    callbacks: list | None = None


def get_wake_guard() -> WakeGuardContext | None:
    """Return the active wake guard context, if any.

    Returns:
        Wake guard for the current wake cycle, or ``None`` outside queue wakes.
    """
    return _wake_guard.get()


@contextmanager
def wake_guard_context(context: WakeGuardContext) -> Iterator[WakeGuardContext]:
    """Install wake guard context for the duration of a graph invocation.

    Args:
        context: Guard metadata for archive compaction and phase rebuilds.

    Yields:
        The same context instance.
    """
    token = _wake_guard.set(context)
    try:
        yield context
    finally:
        _wake_guard.reset(token)
