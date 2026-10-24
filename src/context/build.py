"""Assemble model context from persistent files, compaction, and message archive."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from config import Settings
from context.pillars import (
    KNOWLEDGE_PREFIX,
    PLAN_PREFIX,
    SYSTEM_PREFIX,
    WORLD_PREFIX,
    build_knowledge_recall,
    build_plan_summary,
    build_world_snapshot,
)
from context.monitors import build_monitors_snapshot
from context.system import build_system_prompt
from runtime.scratchpad_layout import PROGRESS_FILENAME, SCRATCHPAD_FILENAME
from store.compaction import CompactionStore
from store.messages import MessageLogStore
from utils.anchored_edit import read_file

GOALS_FILENAME = "_goals.md"
SYSTEM_FILENAME = "_system.md"
EMPTY_PLACEHOLDER = "(empty)"
CHECKPOINT_PREFIX = "Memory checkpoint (compressed earlier conversation):\n"
GOALS_PREFIX = "Goals:\n"
PROGRESS_PREFIX = "Progress summary:\n"
SCRATCHPAD_PREFIX = "Scratchpad:\n"
OPEN_TASKS_PREFIX = "Open tasks:\n"

# Backward-compatible alias used in docs and compaction token checks.
SUMMARY_PREFIX = CHECKPOINT_PREFIX


def build_agent_context(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    thread_id: str,
    pending_tasks: list[dict[str, Any]],
    tools: Sequence[BaseTool],
) -> list[BaseMessage]:
    """Build the full message list sent to the LLM for a wake cycle.

    Order:
        1. System prompt
        2. Central goal (`_goals.md`)
        3. Owner instructions (`_system.md`)
        4. Plan summary
        5. World snapshot
        6. Knowledge recall
        7. Progress summary (`PROGRESS.md`)
        8. Scratchpad
        9. Monitored processes snapshot
        10. Memory checkpoint + verbatim transcript
        11. Open tasks

    Args:
        conn: SQLite connection.
        settings: Application settings.
        thread_id: Conversation thread identifier.
        pending_tasks: Pending task snapshots for the open-tasks block.
        tools: Tools available to the agent this wake.

    Returns:
        Messages to pass to the model.
    """
    scratchpad_dir = Path(settings.scratchpad_directory)
    goals_text = _read_data_file(scratchpad_dir, GOALS_FILENAME)
    messages: list[BaseMessage] = [
        SystemMessage(content=build_system_prompt(settings, tools)),
        HumanMessage(content=f"{GOALS_PREFIX}{goals_text}"),
        HumanMessage(content=f"{SYSTEM_PREFIX}{_read_data_file(scratchpad_dir, SYSTEM_FILENAME)}"),
        HumanMessage(content=f"{PLAN_PREFIX}{build_plan_summary(conn, thread_id)}"),
        HumanMessage(content=f"{WORLD_PREFIX}{build_world_snapshot(conn, thread_id)}"),
        HumanMessage(
            content=f"{KNOWLEDGE_PREFIX}{build_knowledge_recall(settings, query=goals_text, top_k=5)}",
        ),
        HumanMessage(
            content=f"{PROGRESS_PREFIX}{_read_data_file(scratchpad_dir, PROGRESS_FILENAME)}",
        ),
        HumanMessage(
            content=f"{SCRATCHPAD_PREFIX}{_read_data_file(scratchpad_dir, SCRATCHPAD_FILENAME)}",
        ),
        HumanMessage(content=build_monitors_snapshot(conn, settings)),
        *build_transcript_messages(conn, thread_id),
        HumanMessage(content=f"{OPEN_TASKS_PREFIX}{json.dumps(pending_tasks, indent=2)}"),
    ]
    return messages


def build_transcript_messages(conn: sqlite3.Connection, thread_id: str) -> list[BaseMessage]:
    """Build checkpoint plus verbatim messages after the compaction boundary.

    When no compaction exists, returns the full archived transcript.

    Args:
        conn: SQLite connection.
        thread_id: Conversation thread identifier.

    Returns:
        Checkpoint and/or archived conversation messages.
    """
    log = MessageLogStore(conn)
    compaction = CompactionStore(conn).get_latest(thread_id)
    max_seq = log.max_seq(thread_id)
    if max_seq == 0:
        return []

    if compaction is None:
        return log.list_range(thread_id, start_seq=1, end_seq=max_seq)

    messages: list[BaseMessage] = [
        HumanMessage(content=f"{CHECKPOINT_PREFIX}{compaction.summary}"),
    ]
    if compaction.boundary_seq < max_seq:
        messages.extend(
            log.list_range(
                thread_id,
                start_seq=compaction.boundary_seq + 1,
                end_seq=max_seq,
            ),
        )
    return messages


def build_model_context(conn: sqlite3.Connection, thread_id: str) -> list[BaseMessage]:
    """Build transcript-only context for token estimation and legacy callers.

    Args:
        conn: SQLite connection.
        thread_id: Conversation thread identifier.

    Returns:
        Checkpoint and verbatim transcript messages.
    """
    return build_transcript_messages(conn, thread_id)


def _read_data_file(directory: Path, filename: str) -> str:
    """Read a scratchpad-directory markdown file for context assembly.

    Args:
        directory: Scratchpad root directory.
        filename: Basename within the directory.

    Returns:
        File contents, or ``EMPTY_PLACEHOLDER`` when missing or blank.
    """
    content = read_file(directory / filename)
    if not content:
        return EMPTY_PLACEHOLDER
    return content
