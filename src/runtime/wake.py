"""Build wake-cycle payloads and invoke the agent graph."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage
from langgraph.graph.state import CompiledStateGraph

from config import Settings
from context.budget import validate_wake_context_budget
from context.build import build_agent_context
from context.compact import maybe_compact_context
from context.wake_guard import WakeGuardContext, wake_guard_context
from runtime.critical import CriticalWakeError, classify_wake_error
from graph.phases import PHASE_ORDER
from store.alerts import Alert
from store.messages import MessageLogStore
from store.tasks import Task
from tools import get_tools
from utils.observability import flush_langfuse, langfuse_wake_trace
from utils.logging import wake_logging_context

logger = logging.getLogger("thoughtbuddy.runtime.wake")


@dataclass(frozen=True)
class WakeContext:
    """Snapshot passed to the agent at the start of a wake cycle."""

    pending_tasks: list[dict[str, Any]]
    upcoming_alerts: list[dict[str, Any]]
    queue_stats: dict[str, int]


def build_wake_context(
    *,
    pending_tasks: list[Task],
    upcoming_alerts: list[Alert],
    queue_stats: dict[str, int],
) -> WakeContext:
    """Build structured wake context for the agent.

    Args:
        pending_tasks: Tasks awaiting action.
        upcoming_alerts: Enabled alerts scheduled soon.
        queue_stats: Task counts grouped by status.

    Returns:
        Wake context snapshot.
    """
    return WakeContext(
        pending_tasks=[_task_to_dict(task) for task in pending_tasks],
        upcoming_alerts=[_alert_to_dict(alert) for alert in upcoming_alerts],
        queue_stats=queue_stats,
    )


def build_wake_message(context: WakeContext) -> str:
    """Render wake context as a prompt for the agent orient step.

    Args:
        context: Wake context snapshot.

    Returns:
        Human-readable wake instruction for the graph.
    """
    body = {
        "event": "wake",
        "harness_phases": list(PHASE_ORDER),
        "instructions": [
            "Run the fixed harness phases in order.",
            "Orient → Observe → Measure → Reason → Plan → Act/rest.",
            "Complete or ignore queue tasks during Act.",
            "Refresh PROGRESS.md before resting.",
        ],
        "context": asdict(context),
    }
    return json.dumps(body, indent=2)


def invoke_wake_cycle(
    graph: CompiledStateGraph,
    context: WakeContext,
    conn: sqlite3.Connection,
    settings: Settings,
    model: BaseChatModel,
    *,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Invoke one agent wake cycle with archive-backed context.

    Appends the wake message to ``message_log``, compacts if over budget, builds
    full agent context, invokes the graph, then archives new messages from the turn.

    Args:
        graph: Compiled ThoughtBuddy graph.
        context: Wake context snapshot.
        conn: SQLite connection for message archive and compaction.
        settings: Application settings.
        model: Chat model used for compaction summarization.
        thread_id: LangGraph thread identifier. Defaults to ``settings.default_thread_id``.

    Returns:
        Graph invocation result.
    """
    resolved_thread_id = thread_id or settings.default_thread_id
    wake_id = str(uuid.uuid4())
    log = MessageLogStore(conn)

    task_ids = [task["id"] for task in context.pending_tasks]

    with wake_logging_context(wake_id=wake_id, thread_id=resolved_thread_id):
        with langfuse_wake_trace(
            settings,
            wake_id=wake_id,
            thread_id=resolved_thread_id,
            task_ids=task_ids,
        ) as langfuse_trace:
            logger.info(
                "wake_start pending_tasks=%s",
                len(context.pending_tasks),
                extra={"task_ids": task_ids},
            )
            try:
                wake_message = HumanMessage(content=build_wake_message(context))
                log.append(resolved_thread_id, wake_message)
                conn.commit()

                langfuse_callbacks = langfuse_trace.callbacks if langfuse_trace else None
                maybe_compact_context(
                    conn,
                    settings,
                    model,
                    resolved_thread_id,
                    callbacks=langfuse_callbacks,
                )
                conn.commit()

                tools = get_tools(settings)
                context_messages = build_agent_context(
                    conn,
                    settings,
                    thread_id=resolved_thread_id,
                    pending_tasks=context.pending_tasks,
                    tools=tools,
                )
                validate_wake_context_budget(
                    context_messages,
                    settings,
                    conn=conn,
                    thread_id=resolved_thread_id,
                    model=model,
                    callbacks=langfuse_callbacks,
                )
                input_messages: list[BaseMessage] = [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *context_messages,
                ]
                invoke_config = (
                    langfuse_trace.invoke_config(
                        resolved_thread_id,
                        extra_metadata={"source": "wake"},
                    )
                    if langfuse_trace
                    else {"configurable": {"thread_id": resolved_thread_id}}
                )
                guard = WakeGuardContext(
                    conn=conn,
                    settings=settings,
                    thread_id=resolved_thread_id,
                    compaction_model=model,
                    pending_tasks=list(context.pending_tasks),
                    tools=tools,
                    callbacks=langfuse_callbacks,
                )
                with wake_guard_context(guard):
                    result = graph.invoke({"messages": input_messages}, config=invoke_config)

                logger.info("wake_end")
                flush_langfuse(settings)
                return result
            except Exception as exc:
                critical = classify_wake_error(exc, settings)
                if critical is not None:
                    raise critical from exc
                raise


def _task_to_dict(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "source": task.source,
        "status": task.status,
        "payload": task.payload,
        "created_at": task.created_at,
        "alert_id": task.alert_id,
        "occurrence_at": task.occurrence_at,
    }


def _alert_to_dict(alert: Alert) -> dict[str, Any]:
    return {
        "id": alert.id,
        "description": alert.description,
        "schedule_kind": alert.schedule_kind,
        "cron_expr": alert.cron_expr,
        "run_at": alert.run_at,
        "next_fire_at": alert.next_fire_at,
    }
