"""Hidden heartbeat loop."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from config import Settings
from runtime.critical import CriticalWakeError, classify_wake_error
from runtime.wake import build_wake_context, invoke_wake_cycle
from runtime.experiment_ingress import materialize_experiment_tasks
from scheduling.materialize import materialize_due_alerts
from store.alerts import AlertStore
from store.tasks import TaskStore

logger = logging.getLogger("thoughtbuddy.scheduling.heartbeat")

WakeHandler = Callable[[], Awaitable[None]]
CriticalWakeHandler = Callable[..., None]
IsHaltedCallback = Callable[[], bool]


class HeartbeatService:
    """Poll SQLite, materialize alerts, and wake the agent when work exists."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        graph: CompiledStateGraph,
        *,
        model: BaseChatModel,
        settings: Settings,
        upcoming_window_seconds: int | None = None,
        thread_id: str | None = None,
        on_critical_failure: CriticalWakeHandler | None = None,
        is_halted: IsHaltedCallback | None = None,
    ) -> None:
        """Initialize the heartbeat service.

        Args:
            conn: SQLite connection.
            graph: Compiled agent graph used for wake cycles.
            model: Chat model used for context compaction summaries.
            settings: Application settings.
            upcoming_window_seconds: Lookahead window for upcoming alerts in wake context.
            thread_id: LangGraph thread id for wake invocations.
            on_critical_failure: Optional callback when a wake fails unrecoverably.
            is_halted: Optional callback to skip wakes after a critical halt.
        """
        self._conn = conn
        self._graph = graph
        self._model = model
        self._settings = settings
        self._interval_seconds = settings.heartbeat_interval_seconds
        self._stale_task_seconds = settings.stale_task_seconds
        self._upcoming_window_seconds = (
            upcoming_window_seconds or settings.upcoming_alerts_window_seconds
        )
        self._thread_id = thread_id or settings.default_thread_id
        self._on_critical_failure = on_critical_failure
        self._is_halted = is_halted
        self._stop = asyncio.Event()
        self._wake_pending = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._agent_busy = False
        self._task: asyncio.Task[None] | None = None

    @property
    def interval_seconds(self) -> int:
        """Return the configured heartbeat interval."""
        return self._interval_seconds

    def notify(self) -> None:
        """Signal the heartbeat to wake soon after an event task is enqueued."""
        self._wake_pending.set()
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._wake_pending.set)

    async def start(self) -> None:
        """Start the heartbeat loop as a background asyncio task."""
        if self._task is not None and not self._task.done():
            return
        self._loop = asyncio.get_running_loop()
        self._stop.clear()
        self._wake_pending.clear()
        self._task = asyncio.create_task(self._run(), name="thoughtbuddy-heartbeat")

    async def stop(self) -> None:
        """Stop the heartbeat loop and wait for cancellation."""
        self._stop.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        self._loop = None

    async def tick_once(self) -> None:
        """Run a single heartbeat tick (used in tests)."""
        await asyncio.to_thread(self._maintenance_tick)
        await self._maybe_wake()

    async def _run(self) -> None:
        logger.info("ThoughtBuddy heartbeat started (interval=%ss)", self._interval_seconds)
        try:
            while not self._stop.is_set():
                await self.tick_once()
                self._wake_pending.clear()
                try:
                    await asyncio.wait_for(
                        self._wait_for_stop_or_notify(),
                        timeout=self._interval_seconds,
                    )
                except TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info("ThoughtBuddy heartbeat cancelled")
            raise
        finally:
            logger.info("ThoughtBuddy heartbeat stopped")

    async def _wait_for_stop_or_notify(self) -> None:
        """Wait until stop is requested or an immediate wake is signaled."""
        stop_task = asyncio.create_task(self._stop.wait())
        wake_task = asyncio.create_task(self._wake_pending.wait())
        done, pending = await asyncio.wait(
            {stop_task, wake_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    def _maintenance_tick(self) -> None:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            TaskStore(self._conn).reap_stale_running(
                stale_after_seconds=self._stale_task_seconds,
            )
            created = materialize_due_alerts(self._conn)
            created += materialize_experiment_tasks(self._conn, self._settings)
            self._conn.commit()
            if created:
                logger.info("Materialized %s wake task(s)", created)
        except Exception:
            self._conn.rollback()
            logger.exception("Heartbeat maintenance tick failed")
            raise

    async def _maybe_wake(self) -> None:
        if self._agent_busy:
            return
        if self._is_halted is not None and self._is_halted():
            return

        pending = await asyncio.to_thread(self._load_pending)
        if not pending:
            return

        self._agent_busy = True
        try:
            upcoming, stats = await asyncio.to_thread(self._load_wake_metadata)
            context = build_wake_context(
                pending_tasks=pending,
                upcoming_alerts=upcoming,
                queue_stats=stats,
            )
            task_ids = [task.id for task in pending]
            await asyncio.to_thread(self._mark_running, task_ids)
            await asyncio.to_thread(
                invoke_wake_cycle,
                self._graph,
                context,
                self._conn,
                self._settings,
                self._model,
                thread_id=self._thread_id,
            )
            await asyncio.to_thread(self._release_undischarged_tasks, task_ids)
        except Exception as exc:
            critical = (
                exc
                if isinstance(exc, CriticalWakeError)
                else classify_wake_error(exc, self._settings)
            )
            if critical is not None:
                logger.critical(
                    "Critical wake failure — halting heartbeat: %s",
                    critical,
                    extra={"task_ids": task_ids},
                )
                await asyncio.to_thread(self._handle_critical_failure, task_ids, critical)
                self._stop.set()
            else:
                logger.exception(
                    "Agent wake cycle failed",
                    extra={"task_ids": task_ids},
                )
                await asyncio.to_thread(self._revert_running_tasks, task_ids)
        finally:
            self._agent_busy = False

    def _load_pending(self):
        return TaskStore(self._conn).list_pending()

    def _load_wake_metadata(self):
        alert_store = AlertStore(self._conn)
        task_store = TaskStore(self._conn)
        return (
            alert_store.list_upcoming(within_seconds=self._upcoming_window_seconds),
            task_store.count_by_status(),
        )

    def _mark_running(self, task_ids: list[str]) -> None:
        self._conn.execute("BEGIN IMMEDIATE")
        TaskStore(self._conn).mark_running(task_ids)
        self._conn.commit()

    def _revert_running_tasks(self, task_ids: list[str]) -> None:
        self._conn.execute("BEGIN IMMEDIATE")
        TaskStore(self._conn).revert_running_to_pending(task_ids)
        self._conn.commit()

    def _handle_critical_failure(
        self,
        task_ids: list[str],
        error: CriticalWakeError,
    ) -> None:
        """Terminal-disposition wake tasks and notify the runtime to halt.

        Args:
            task_ids: Task identifiers included in the failed wake.
            error: Critical failure that must not be retried.
        """
        reason = str(error)[:500]
        self._conn.execute("BEGIN IMMEDIATE")
        task_store = TaskStore(self._conn)
        for task_id in task_ids:
            try:
                task_store.update_status(
                    task_id,
                    status="ignored",
                    disposition_reason=reason,
                )
            except ValueError:
                logger.warning("Could not ignore task after critical failure: %s", task_id)
        self._conn.commit()
        if self._on_critical_failure is not None:
            self._on_critical_failure(error, task_ids=task_ids)

    def _release_undischarged_tasks(self, task_ids: list[str]) -> None:
        """Revert wake tasks still running after a successful wake cycle.

        The agent must call ``update_task`` during Act. If the wake graph returns
        without a terminal disposition, leave the queue unblocked for retry.

        Args:
            task_ids: Task identifiers included in the wake.
        """
        self._conn.execute("BEGIN IMMEDIATE")
        reverted = TaskStore(self._conn).revert_running_to_pending(task_ids)
        self._conn.commit()
        if reverted:
            logger.warning(
                "Wake finished without task disposition; reverted %s task(s) to pending",
                reverted,
                extra={"task_ids": task_ids},
            )
