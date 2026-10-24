"""Agent-facing monitored subprocess tools."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from config import Settings
from runtime.service import get_active_runtime
from tools.registry import register_tools


class RunMonitoredCommandInput(BaseModel):
    """Input schema for ``run_monitored_command``."""

    command: str = Field(
        description=(
            "Command to run (split with shlex, no shell). Use for work expected to "
            "exceed the configured minimum duration (~60s or several heartbeats)."
        ),
    )
    description: str = Field(
        description="Short summary of what this subprocess is doing.",
    )
    timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        le=86400,
        description="Optional timeout override in seconds (default 600).",
    )


class ListMonitorsInput(BaseModel):
    """Input schema for ``list_monitors``."""

    limit: int = Field(default=10, ge=1, le=50, description="Maximum rows to return.")


def create_tools(settings: Settings) -> list[BaseTool]:
    """Create monitor tools bound to the active runtime.

    Args:
        settings: Application settings.

    Returns:
        Monitor subprocess tools.
    """
    _ = settings

    def run_monitored_command(
        command: str,
        description: str,
        timeout_seconds: int | None = None,
    ) -> str:
        """Start a long-running subprocess monitored by the harness."""
        runtime = get_active_runtime()
        if runtime is None:
            return json.dumps({"error": "Runtime not started; monitor unavailable."})
        try:
            result = runtime.monitors.start(
                command=command,
                description=description,
                timeout_seconds=timeout_seconds,
            )
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(result, indent=2)

    def list_monitors(limit: int = 10) -> str:
        """List recent and running monitored subprocesses."""
        runtime = get_active_runtime()
        if runtime is None:
            return json.dumps({"error": "Runtime not started; monitor unavailable."})
        from store.monitors import MonitorStore

        monitors = MonitorStore(runtime.conn).list_recent(
            thread_id=runtime.settings.default_thread_id,
            limit=limit,
        )
        payload = [
            {
                "monitor_id": monitor.id,
                "status": monitor.status,
                "description": monitor.description,
                "command": monitor.command,
                "log_dir": monitor.log_dir,
                "timeout_seconds": monitor.timeout_seconds,
                "exit_code": monitor.exit_code,
                "started_at": monitor.started_at,
                "finished_at": monitor.finished_at,
            }
            for monitor in monitors
        ]
        return json.dumps({"monitors": payload}, indent=2)

    return [
        StructuredTool.from_function(
            func=run_monitored_command,
            name="run_monitored_command",
            description=(
                "Start a background subprocess for long-running work (default timeout 10 "
                "minutes). Use only when the job is expected to exceed ~60 seconds or "
                "several heartbeat intervals — not for chess moves, websearch, or other "
                "fast tools. Logs land under .data/scratchpad/monitors/<id>/; completion "
                "enqueues an event task and wakes the agent."
            ),
            args_schema=RunMonitoredCommandInput,
        ),
        StructuredTool.from_function(
            func=list_monitors,
            name="list_monitors",
            description="List running and recently finished monitored subprocesses.",
            args_schema=ListMonitorsInput,
        ),
    ]


register_tools(create_tools)
