"""ThoughtBuddy runtime orchestration."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from agent import create_agent
from config import Settings
from db.connection import connect
from experiments.chess.plugin.validate import validate_chess_experiment
from llm import create_chat_model
from runtime.critical import CriticalWakeError
from runtime.failure_report import write_experiment_failure_report
from runtime.monitor_service import MonitorService
from runtime.scratchpad_layout import ensure_scratchpad_layout
from scheduling.heartbeat import HeartbeatService
from scheduling.lock import InstanceLock
from utils.observability import shutdown_langfuse

logger = logging.getLogger(__name__)

_active_runtime: ThoughtBuddyRuntime | None = None


def get_active_runtime() -> ThoughtBuddyRuntime | None:
    """Return the live runtime for the current process, if started.

    Returns:
        Active runtime bundle, or None before lifespan start.
    """
    return _active_runtime


def _set_active_runtime(runtime: ThoughtBuddyRuntime | None) -> None:
    global _active_runtime
    _active_runtime = runtime


@dataclass
class ThoughtBuddyRuntime:
    """Owns SQLite, the PID lock, monitors, and the heartbeat service."""

    settings: Settings
    conn: sqlite3.Connection
    graph: CompiledStateGraph
    model: BaseChatModel
    lock: InstanceLock
    heartbeat: HeartbeatService
    monitors: MonitorService
    halt_reason: str | None = None
    halt_task_ids: list[str] | None = None

    def halt(
        self,
        reason: str,
        *,
        task_ids: list[str] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        """Record a critical failure and stop scheduling further wakes.

        Args:
            reason: Operator-facing explanation persisted in health checks.
            task_ids: Queue tasks involved in the failed wake.
            cause: Optional underlying exception for the failure report.
        """
        if self.halt_reason is not None:
            return
        self.halt_reason = reason
        self.halt_task_ids = list(task_ids or [])
        cause_text = f"{type(cause).__name__}: {cause}" if cause is not None else None
        write_experiment_failure_report(
            self.settings,
            reason,
            task_ids=task_ids,
            cause=cause_text,
        )
        logger.critical("ThoughtBuddy runtime halted: %s", reason)

    def is_halted(self) -> bool:
        """Return whether the runtime stopped due to a critical failure."""
        return self.halt_reason is not None

    async def start(self) -> None:
        """Acquire the instance lock and start background services."""
        import asyncio

        self.lock.acquire()
        await asyncio.to_thread(self.monitors.reconcile_on_startup)
        logger.info("ThoughtBuddy runtime started (db=%s)", self.settings.thoughtbuddy_db_path)
        await self.heartbeat.start()

    async def stop(self) -> None:
        """Stop background services and release resources."""
        import asyncio

        await self.heartbeat.stop()
        await asyncio.to_thread(self.monitors.shutdown_all)
        shutdown_langfuse(self.settings)
        self.conn.close()
        self.lock.release()
        logger.info("ThoughtBuddy runtime stopped")


def create_runtime(settings: Settings | None = None) -> ThoughtBuddyRuntime:
    """Build a runtime from settings.

    Args:
        settings: Optional settings override.

    Returns:
        Unstarted runtime bundle.
    """
    resolved = settings or Settings.load()
    validate_chess_experiment(resolved)
    ensure_scratchpad_layout(resolved.scratchpad_directory)
    db_path = Path(resolved.thoughtbuddy_db_path)
    conn = connect(db_path)
    graph = create_agent(resolved)
    model = create_chat_model(resolved)
    lock = InstanceLock(Path(resolved.thoughtbuddy_lock_path))
    runtime_ref: list[ThoughtBuddyRuntime] = []

    def _on_critical_failure(
        error: CriticalWakeError,
        *,
        task_ids: list[str],
    ) -> None:
        if runtime_ref:
            runtime_ref[0].halt(
                str(error),
                task_ids=task_ids,
                cause=error.cause,
            )

    heartbeat = HeartbeatService(
        conn,
        graph,
        model=model,
        settings=resolved,
        on_critical_failure=_on_critical_failure,
        is_halted=lambda: runtime_ref[0].is_halted() if runtime_ref else False,
    )
    monitors = MonitorService(conn, resolved, notify_wake=heartbeat.notify)
    runtime = ThoughtBuddyRuntime(
        settings=resolved,
        conn=conn,
        graph=graph,
        model=model,
        lock=lock,
        heartbeat=heartbeat,
        monitors=monitors,
    )
    runtime_ref.append(runtime)
    return runtime
