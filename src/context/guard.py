"""Pre-LLM context budget checks and archive compaction."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from config import Settings
from context.build import build_agent_context, build_model_context
from context.compact import maybe_compact_context
from context.limits import (
    compaction_token_threshold,
    effective_compact_ratio,
    max_prompt_tokens,
)
from context.tokens import estimate_messages_tokens
from llm.tokens import count_messages_tokens
from runtime.critical import CriticalWakeError

logger = logging.getLogger("thoughtbuddy.context.guard")

_MAX_COMPACTION_ROUNDS = 10


def compact_archive_until_under_threshold(
    conn: sqlite3.Connection,
    settings: Settings,
    model: BaseChatModel,
    thread_id: str,
    *,
    callbacks: list | None = None,
) -> bool:
    """Compact the message archive until it is below the compaction threshold.

    Args:
        conn: SQLite connection.
        settings: Application settings.
        model: Chat model used for summarization.
        thread_id: Conversation thread identifier.
        callbacks: Optional LangChain callbacks for compaction LLM calls.

    Returns:
        True when at least one compaction point was written.
    """
    target = compaction_token_threshold(settings)
    compacted_any = False
    for round_index in range(_MAX_COMPACTION_ROUNDS):
        assembled = build_model_context(conn, thread_id)
        estimated = estimate_messages_tokens(assembled, settings, add_generation_prompt=False)
        if estimated <= target:
            break
        if not maybe_compact_context(
            conn,
            settings,
            model,
            thread_id,
            callbacks=callbacks,
        ):
            break
        compacted_any = True
        logger.info(
            "archive_compacted round=%s estimated_tokens=%s target=%s",
            round_index + 1,
            estimated,
            target,
        )
    return compacted_any


def ensure_prompt_budget(
    messages: Sequence[BaseMessage],
    settings: Settings,
    *,
    conn: sqlite3.Connection | None = None,
    thread_id: str | None = None,
    model: BaseChatModel | None = None,
    callbacks: list | None = None,
) -> None:
    """Fail fast when a prompt exceeds the model context window.

    When ``conn``, ``thread_id``, and ``model`` are supplied, runs archive
    compaction first if the archived context is over the compaction threshold.

    Args:
        messages: Messages about to be sent to the model.
        settings: Application settings.
        conn: Optional SQLite connection for archive compaction.
        thread_id: Optional thread id for archive compaction.
        model: Optional chat model for compaction summarization.
        callbacks: Optional LangChain callbacks for compaction.

    Raises:
        CriticalWakeError: When estimated prompt tokens exceed the hard budget.
    """
    if conn is not None and thread_id is not None and model is not None:
        compact_archive_until_under_threshold(
            conn,
            settings,
            model,
            thread_id,
            callbacks=callbacks,
        )

    estimated = count_messages_tokens(list(messages), settings)
    budget = max_prompt_tokens(settings)
    if estimated <= budget:
        return

    compaction_hint = _compaction_hint(settings)
    msg = (
        f"Prompt estimated at {estimated} tokens exceeds budget {budget} "
        f"(context_window={settings.llm_context_window_tokens}, "
        f"max_completion={settings.llm_max_tokens}, "
        f"compaction_threshold={compaction_token_threshold(settings)}). "
        f"Compaction is {compaction_hint}."
    )
    raise CriticalWakeError(msg)


def ensure_llm_prompt_budget(
    messages: Sequence[BaseMessage],
    settings: Settings,
) -> list[BaseMessage]:
    """Prepare messages immediately before an LLM invocation.

    Uses the active :class:`context.wake_guard.WakeGuardContext` when present
    so archive compaction can run before graph and ReAct subgraph calls.

    Args:
        messages: Messages about to be sent to the model.
        settings: Application settings.

    Returns:
        The same message list (unchanged content).

    Raises:
        CriticalWakeError: When the prompt exceeds the hard token budget.
    """
    from context.wake_guard import get_wake_guard

    guard = get_wake_guard()
    if guard is not None:
        ensure_prompt_budget(
            messages,
            guard.settings,
            conn=guard.conn,
            thread_id=guard.thread_id,
            model=guard.compaction_model,
            callbacks=guard.callbacks,
        )
        return list(messages)

    ensure_prompt_budget(messages, settings)
    return list(messages)


def build_phase_input_messages(
    settings: Settings,
    conn: sqlite3.Connection,
    thread_id: str,
    phase: str,
    *,
    pending_tasks: list[dict],
    tools: Sequence[BaseTool],
    compaction_model: BaseChatModel,
    callbacks: list | None = None,
) -> list[BaseMessage]:
    """Rebuild archived wake context plus one harness phase instruction.

    Compacts the archive when needed, then validates the phase prompt budget.

    Args:
        settings: Application settings.
        conn: SQLite connection.
        thread_id: Conversation thread identifier.
        phase: Harness phase name.
        pending_tasks: Open queue tasks for context assembly.
        tools: Registered tools for context assembly.
        compaction_model: Chat model used for compaction summarization.
        callbacks: Optional LangChain callbacks for compaction.

    Returns:
        Messages to pass into a phase ReAct subgraph.

    Raises:
        CriticalWakeError: When the assembled phase prompt exceeds budget.
    """
    from graph.phases import PHASE_INSTRUCTIONS

    compact_archive_until_under_threshold(
        conn,
        settings,
        compaction_model,
        thread_id,
        callbacks=callbacks,
    )
    messages = list(
        build_agent_context(
            conn,
            settings,
            thread_id=thread_id,
            pending_tasks=pending_tasks,
            tools=tools,
        ),
    )
    messages.append(HumanMessage(content=PHASE_INSTRUCTIONS[phase]))
    ensure_prompt_budget(
        messages,
        settings,
        conn=conn,
        thread_id=thread_id,
        model=compaction_model,
        callbacks=callbacks,
    )
    return messages


def _compaction_hint(settings: Settings) -> str:
    threshold = compaction_token_threshold(settings)
    if settings.context_compact_token_threshold > 0:
        return f"enabled (threshold={threshold})"
    ratio = effective_compact_ratio(settings)
    return f"enabled (threshold={threshold}, ratio={ratio})"
