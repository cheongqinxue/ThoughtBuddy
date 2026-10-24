"""Context compaction when the archive exceeds budget."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config import Settings
from context.build import build_model_context
from context.limits import compaction_token_threshold
from context.tokens import estimate_messages_tokens
from runtime.scratchpad_layout import COMPACT_FILENAME
from store.compaction import CompactionStore
from store.messages import MessageLogStore
from utils.anchored_edit import read_file


class CompactionError(Exception):
    """Raised when compaction cannot run due to missing configuration."""


def load_compaction_prompt(settings: Settings) -> str:
    """Load the compaction system prompt from owner-controlled ``_compact.md``.

    Args:
        settings: Application settings.

    Returns:
        Compaction prompt text.

    Raises:
        CompactionError: When ``_compact.md`` is missing or empty.
    """
    path = Path(settings.scratchpad_directory) / COMPACT_FILENAME
    content = read_file(path).strip()
    if not content:
        msg = f"Compaction prompt file is missing or empty: {path}"
        raise CompactionError(msg)
    return content


def maybe_compact_context(
    conn: sqlite3.Connection,
    settings: Settings,
    model: BaseChatModel,
    thread_id: str,
    *,
    callbacks: list | None = None,
) -> bool:
    """Compact archived messages when the assembled context exceeds budget.

    Args:
        conn: SQLite connection.
        settings: Application settings with retention and threshold values.
        model: Chat model used to generate the summary.
        thread_id: Conversation thread identifier.
        callbacks: Optional LangChain callbacks (e.g. Langfuse handler).

    Returns:
        True if a new compaction point was written.
    """
    threshold = compaction_token_threshold(settings)
    assembled = build_model_context(conn, thread_id)
    if estimate_messages_tokens(assembled, settings, add_generation_prompt=False) <= threshold:
        return False

    log = MessageLogStore(conn)
    max_seq = log.max_seq(thread_id)
    retain = settings.context_retain_messages
    if max_seq <= retain:
        return False

    boundary_seq = max_seq - retain
    compaction = CompactionStore(conn).get_latest(thread_id)
    summarize_start = (compaction.boundary_seq + 1) if compaction else 1
    if boundary_seq < summarize_start:
        return False

    to_summarize = log.list_range(thread_id, start_seq=summarize_start, end_seq=boundary_seq)
    if not to_summarize and not compaction:
        return False

    compaction_prompt = load_compaction_prompt(settings)
    summary = _summarize_messages(
        model,
        compaction_prompt,
        compaction.summary if compaction else None,
        to_summarize,
        callbacks=callbacks,
    )
    CompactionStore(conn).create(
        thread_id,
        boundary_seq=boundary_seq,
        retain_count=retain,
        summary=summary,
    )
    return True


def _summarize_messages(
    model: BaseChatModel,
    compaction_prompt: str,
    previous_summary: str | None,
    messages: list,
    *,
    callbacks: list | None = None,
) -> str:
    """Ask the model to summarize message history.

    Args:
        model: Chat model for summarization.
        compaction_prompt: System prompt loaded from ``_compact.md``.
        previous_summary: Prior compaction summary, if rolling forward.
        messages: Messages to include in this compaction window.
        callbacks: Optional LangChain callbacks (e.g. Langfuse handler).

    Returns:
        Summary text without the display prefix.
    """
    parts: list[str] = []
    if previous_summary:
        parts.append(f"Previous summary:\n{previous_summary}\n")
    parts.append("Messages to compress:\n")
    for index, message in enumerate(messages, start=1):
        parts.append(f"{index}. [{message.type}] {message.content}")
    prompt = "\n".join(parts)

    invoke_config = {"callbacks": callbacks} if callbacks else None
    response = model.invoke(
        [
            SystemMessage(content=compaction_prompt),
            HumanMessage(content=prompt),
        ],
        config=invoke_config,
    )
    content = response.content
    if isinstance(content, list):
        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(text_parts).strip()
    return str(content).strip()
