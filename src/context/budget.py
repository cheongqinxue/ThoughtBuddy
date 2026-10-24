"""Wake context budget validation before LLM invocation."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from config import Settings
from context.guard import ensure_prompt_budget
from context.limits import max_prompt_tokens


def reserved_completion_tokens(settings: Settings) -> int:
    """Return completion tokens reserved against the model context window.

    Args:
        settings: Application settings.

    Returns:
        Tokens reserved for model output on each call.
    """
    return settings.llm_max_tokens


def validate_wake_context_budget(
    messages: Sequence[BaseMessage],
    settings: Settings,
    *,
    conn: sqlite3.Connection | None = None,
    thread_id: str | None = None,
    model: BaseChatModel | None = None,
    callbacks: list | None = None,
) -> None:
    """Fail fast when assembled wake context exceeds the model window.

    Runs archive compaction when ``conn``, ``thread_id``, and ``model`` are
    provided, then validates the prompt against the hard token budget.

    Args:
        messages: Full wake context about to be sent to the model.
        settings: Application settings.
        conn: Optional SQLite connection for archive compaction.
        thread_id: Optional thread id for archive compaction.
        model: Optional chat model for compaction summarization.
        callbacks: Optional LangChain callbacks for compaction.

    Raises:
        CriticalWakeError: When estimated prompt tokens exceed budget.
    """
    ensure_prompt_budget(
        messages,
        settings,
        conn=conn,
        thread_id=thread_id,
        model=model,
        callbacks=callbacks,
    )


__all__ = [
    "max_prompt_tokens",
    "reserved_completion_tokens",
    "validate_wake_context_budget",
]
