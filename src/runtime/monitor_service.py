"""Background monitored subprocess service."""

from __future__ import annotations

import logging
import os
import shlex
import signal
import sqlite3
import subprocess
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import Settings
from store.monitors import MonitoredProcess, MonitorStore, MonitorStatus
from store.tasks import TaskStore
from utils.time import utc_now_str
from world.tools import log_observation

if TYPE_CHECKING:
    from scheduling.heartbeat import HeartbeatService

logger = logging.getLogger("thoughtbuddy.runtime.monitors")

NotifyCallback = Callable[[], None]


class MonitorService:
    """Start, watch, and finalize monitored subprocesses."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        settings: Settings,
        *,
        notify_wake: NotifyCallback | None = None,
    ) -> None:
        """Initialize the monitor service.

        Args:
            conn: Shared SQLite connection for the runtime process.
            settings: Application settings.
            notify_wake: Optional callback to signal the heartbeat immediately.
        """
        self._conn = conn
        self._settings = settings
        self._notify_wake = notify_wake
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def set_notify_wake(self, notify_wake: NotifyCallback | None) -> None:
        """Attach or replace the heartbeat notify callback.

        Args:
            notify_wake: Callable invoked after an event task is enqueued.
        """
        self._notify_wake = notify_wake

    @property
    def min_expected_seconds(self) -> int:
        """Return the configured minimum expected monitor duration."""
        heartbeat_floor = (
            self._settings.monitor_min_expected_heartbeats
            * self._settings.heartbeat_interval_seconds
        )
        return max(self._settings.monitor_min_expected_seconds, heartbeat_floor)

    def reconcile_on_startup(self) -> int:
        """Mark orphaned running monitors interrupted after an outage.

        Returns:
            Number of monitors reconciled.
        """
        store = MonitorStore(self._conn)
        reconciled = 0
        for monitor in store.list_running():
            alive = monitor.pid is not None and _pid_alive(monitor.pid)
            if alive:
                _kill_process_group(monitor.pgid or monitor.pid)
            self._finalize_locked(
                monitor.id,
                status="interrupted",
                exit_code=None,
                signal_name="SIGTERM" if alive else None,
                process=None,
            )
            reconciled += 1
        if reconciled:
            self._conn.commit()
            logger.warning("Reconciled %s interrupted monitor(s) on startup", reconciled)
        return reconciled

    def shutdown_all(self) -> None:
        """Terminate all running monitors during runtime shutdown."""
        store = MonitorStore(self._conn)
        running = store.list_running()
        for monitor in running:
            proc = self._processes.get(monitor.id)
            if proc is not None and proc.poll() is None:
                _kill_process_group(monitor.pgid or proc.pid)
            elif monitor.pid is not None and _pid_alive(monitor.pid):
                _kill_process_group(monitor.pgid or monitor.pid)
        self._conn.execute("BEGIN IMMEDIATE")
        for monitor in running:
            try:
                self._finalize_locked(
                    monitor.id,
                    status="interrupted",
                    exit_code=None,
                    signal_name="SIGTERM",
                    process=self._processes.get(monitor.id),
                )
            except ValueError:
                continue
        self._conn.commit()
        self._processes.clear()

    def start(
        self,
        *,
        command: str,
        description: str,
        timeout_seconds: int | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Start a monitored subprocess.

        Args:
            command: Shell command split with ``shlex`` (no shell invocation).
            description: Short human-readable summary for context and events.
            timeout_seconds: Optional timeout override.
            thread_id: Conversation thread id override.

        Returns:
            JSON-serializable monitor metadata.

        Raises:
            ValueError: When limits are exceeded or command is empty.
        """
        argv = _parse_command(command)
        if not argv:
            msg = "command must not be empty"
            raise ValueError(msg)

        store = MonitorStore(self._conn)
        if store.count_running() >= self._settings.monitor_max_concurrent:
            msg = (
                f"At most {self._settings.monitor_max_concurrent} concurrent monitors allowed"
            )
            raise ValueError(msg)

        resolved_timeout = timeout_seconds or self._settings.monitor_default_timeout_seconds
        resolved_thread = thread_id or self._settings.default_thread_id
        monitor_id = str(uuid.uuid4())
        log_dir = Path(self._settings.scratchpad_directory) / "monitors" / monitor_id
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"

        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                argv,
                stdout=stdout_handle,
                stderr=stderr_handle,
                cwd=str(Path.cwd()),
                start_new_session=True,
            )
        except OSError as exc:
            stdout_handle.close()
            stderr_handle.close()
            msg = f"Failed to start monitor subprocess: {exc}"
            raise ValueError(msg) from exc

        pgid = os.getpgid(process.pid)
        self._conn.execute("BEGIN IMMEDIATE")
        monitor = store.create(
            thread_id=resolved_thread,
            command=argv,
            description=description,
            log_dir=str(log_dir),
            timeout_seconds=resolved_timeout,
            pid=process.pid,
            pgid=pgid,
            monitor_id=monitor_id,
        )
        self._conn.commit()

        with self._lock:
            self._processes[monitor_id] = process

        watcher = threading.Thread(
            target=self._watch_process,
            args=(monitor_id, process, stdout_handle, stderr_handle, resolved_timeout),
            name=f"monitor-{monitor_id[:8]}",
            daemon=True,
        )
        watcher.start()

        return {
            "monitor_id": monitor.id,
            "status": monitor.status,
            "description": monitor.description,
            "command": monitor.command,
            "log_dir": monitor.log_dir,
            "timeout_seconds": monitor.timeout_seconds,
            "min_expected_seconds": self.min_expected_seconds,
        }

    def _watch_process(
        self,
        monitor_id: str,
        process: subprocess.Popen[bytes],
        stdout_handle,
        stderr_handle,
        timeout_seconds: int,
    ) -> None:
        status: MonitorStatus = "failed"
        exit_code: int | None = None
        signal_name: str | None = None
        try:
            try:
                exit_code = process.wait(timeout=timeout_seconds)
                status = "completed" if exit_code == 0 else "failed"
            except subprocess.TimeoutExpired:
                _kill_process_group(os.getpgid(process.pid))
                try:
                    exit_code = process.wait(timeout=5)
                    status = "timed_out"
                except subprocess.TimeoutExpired:
                    process.kill()
                    exit_code = process.wait(timeout=5)
                    status = "timed_out"
                    signal_name = "SIGKILL"
        finally:
            stdout_handle.close()
            stderr_handle.close()
            with self._lock:
                self._processes.pop(monitor_id, None)
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._finalize_locked(
                    monitor_id,
                    status=status,
                    exit_code=exit_code,
                    signal_name=signal_name,
                    process=process,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                logger.exception("Failed to finalize monitor %s", monitor_id)

    def _finalize_locked(
        self,
        monitor_id: str,
        *,
        status: MonitorStatus,
        exit_code: int | None,
        signal_name: str | None,
        process: subprocess.Popen[bytes] | None,
    ) -> None:
        store = MonitorStore(self._conn)
        monitor = store.get(monitor_id)
        if monitor is None:
            msg = f"Monitor not found: {monitor_id}"
            raise ValueError(msg)
        if monitor.status != "running":
            return

        if process is not None and process.poll() is None:
            _kill_process_group(monitor.pgid or process.pid)

        finalized = store.finalize(
            monitor_id,
            status=status,
            exit_code=exit_code,
            signal_name=signal_name,
        )
        stdout_tail, stderr_tail = _read_log_tails(
            Path(finalized.log_dir),
            max_bytes=self._settings.monitor_log_tail_bytes,
        )
        event_payload = _build_monitor_event_payload(
            finalized,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )
        message = event_payload["message"]
        occurrence_at = utc_now_str()
        event_key = f"monitor:{monitor_id}"
        TaskStore(self._conn).enqueue_event_task(
            message=message,
            event_key=event_key,
            occurrence_at=occurrence_at,
            payload=event_payload,
        )
        log_observation(
            self._settings,
            source="monitor",
            summary=message,
            payload=event_payload,
            thread_id=finalized.thread_id,
            conn=self._conn,
        )
        if self._notify_wake is not None:
            self._notify_wake()

    def enqueue_external_event(
        self,
        *,
        message: str,
        event_key: str,
        payload: dict[str, Any] | None = None,
        occurrence_at: str | None = None,
    ) -> dict[str, Any] | None:
        """Enqueue an external callback event and signal the heartbeat.

        Args:
            message: Wake instruction for the agent.
            event_key: Dedup key for the event source.
            payload: Optional structured payload.
            occurrence_at: Optional occurrence timestamp override.

        Returns:
            Created task metadata, or None when deduped.
        """
        resolved_occurrence = occurrence_at or utc_now_str()
        body = {"message": message, **(payload or {})}
        self._conn.execute("BEGIN IMMEDIATE")
        task = TaskStore(self._conn).enqueue_event_task(
            message=message,
            event_key=event_key,
            occurrence_at=resolved_occurrence,
            payload=body,
        )
        self._conn.commit()
        if task is None:
            return None
        if self._notify_wake is not None:
            self._notify_wake()
        return {"task_id": task.id, "status": task.status, "source": task.source}


def _parse_command(command: str) -> list[str]:
    """Split a command string into argv without shell interpolation.

    Args:
        command: User/agent command string.

    Returns:
        argv list for ``subprocess.Popen``.
    """
    stripped = command.strip()
    if not stripped:
        return []
    return shlex.split(stripped)


def _pid_alive(pid: int) -> bool:
    """Return whether a pid is still running.

    Args:
        pid: Process id.

    Returns:
        True when ``kill(pid, 0)`` succeeds.
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _kill_process_group(pgid: int | None) -> None:
    """Send SIGTERM to a process group.

    Args:
        pgid: Process group id.
    """
    if pgid is None:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _read_log_tails(log_dir: Path, *, max_bytes: int) -> tuple[str, str]:
    """Read trailing stdout/stderr log bytes.

    Args:
        log_dir: Monitor log directory.
        max_bytes: Maximum bytes per stream.

    Returns:
        Tuple of stdout and stderr tail text.
    """
    return (
        _read_tail(log_dir / "stdout.log", max_bytes=max_bytes),
        _read_tail(log_dir / "stderr.log", max_bytes=max_bytes),
    )


def _read_tail(path: Path, *, max_bytes: int) -> str:
    """Read the tail of a text file.

    Args:
        path: File path.
        max_bytes: Maximum bytes to read from the end.

    Returns:
        Tail text or empty string when missing.
    """
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


def _build_monitor_event_payload(
    monitor: MonitoredProcess,
    *,
    stdout_tail: str,
    stderr_tail: str,
) -> dict[str, Any]:
    """Build the event/world payload for a finalized monitor.

    Args:
        monitor: Finalized monitor row.
        stdout_tail: Truncated stdout text.
        stderr_tail: Truncated stderr text.

    Returns:
        Structured payload stored on the event task.
    """
    kind = f"monitor_{monitor.status}"
    message = (
        f"Monitor '{monitor.description}' finished with status={monitor.status}"
        f" exit_code={monitor.exit_code}."
    )
    return {
        "kind": kind,
        "message": message,
        "monitor_id": monitor.id,
        "description": monitor.description,
        "command": monitor.command,
        "status": monitor.status,
        "exit_code": monitor.exit_code,
        "signal": monitor.signal_name,
        "log_dir": monitor.log_dir,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "started_at": monitor.started_at,
        "finished_at": monitor.finished_at,
    }
