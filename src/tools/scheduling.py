"""Agent-facing scheduling tools."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from config import Settings
from db.connection import connect
from scheduling.schedule_map import build_schedule_map, compute_initial_next_fire
from store.alerts import AlertStore
from store.tasks import TaskStore

T = TypeVar("T")


class CreateAlertInput(BaseModel):
    """Input schema for create_alert."""

    description: str = Field(description="Human-readable summary of what this alert is for.")
    schedule_kind: Literal["cron", "once"] = Field(
        description="Use 'cron' for recurring alerts or 'once' for a single fire time.",
    )
    cron_expr: str | None = Field(
        default=None,
        description="Cron expression for recurring alerts (required when schedule_kind='cron').",
    )
    run_at: str | None = Field(
        default=None,
        description="UTC fire time 'YYYY-MM-DD HH:MM:SS' for one-shot alerts.",
    )
    payload: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON payload copied into materialized tasks.",
    )


class UpdateAlertInput(BaseModel):
    """Input schema for update_alert."""

    alert_id: str = Field(description="Identifier of the alert to update.")
    enabled: bool | None = Field(default=None, description="Set false to disable an alert.")
    description: str | None = Field(default=None, description="Updated description.")
    cron_expr: str | None = Field(default=None, description="Updated cron expression.")
    run_at: str | None = Field(default=None, description="Updated one-shot run time.")
    next_fire_at: str | None = Field(
        default=None,
        description="Override the next fire time (UTC 'YYYY-MM-DD HH:MM:SS').",
    )
    payload: dict[str, Any] | None = Field(default=None, description="Replacement payload.")


class UpdateTaskInput(BaseModel):
    """Input schema for update_task."""

    task_id: str = Field(description="Identifier of the task to update.")
    status: Literal["completed", "ignored"] = Field(
        description="Terminal status: 'completed' if handled, 'ignored' to skip.",
    )
    disposition_reason: str | None = Field(
        default=None,
        description="Why the task was completed or ignored.",
    )


class ListAlertsInput(BaseModel):
    """Input schema for list_alerts."""

    window_seconds: int = Field(
        default=3600,
        description="Lookahead window in seconds for upcoming alert fires.",
        ge=60,
        le=86400,
    )


def create_tools(settings: Settings) -> list[BaseTool]:
    """Create ThoughtBuddy scheduling tools bound to the configured SQLite database.

    Args:
        settings: Application settings containing the database path.

    Returns:
        LangChain tools for alert and task management.
    """
    db_path = Path(settings.thoughtbuddy_db_path)

    def with_db(action: Callable[[Any], T]) -> T:
        conn = connect(db_path)
        try:
            result = action(conn)
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_alert(
        description: str,
        schedule_kind: Literal["cron", "once"],
        cron_expr: str | None = None,
        run_at: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        def work(conn) -> dict[str, Any]:
            store = AlertStore(conn)
            body = payload or {}
            if schedule_kind == "once":
                if not run_at:
                    msg = "run_at is required when schedule_kind='once'"
                    raise ValueError(msg)
                alert = store.create_once_alert(
                    description=description,
                    run_at=run_at,
                    payload=body,
                )
            else:
                if not cron_expr:
                    msg = "cron_expr is required when schedule_kind='cron'"
                    raise ValueError(msg)
                next_fire = compute_initial_next_fire(cron_expr)
                alert = store.create_cron_alert(
                    description=description,
                    cron_expr=cron_expr,
                    next_fire_at=next_fire,
                    payload=body,
                )
            return {
                "alert_id": alert.id,
                "schedule_kind": alert.schedule_kind,
                "next_fire_at": alert.next_fire_at,
                "description": alert.description,
            }

        return json.dumps(with_db(work), indent=2)

    def update_alert(
        alert_id: str,
        enabled: bool | None = None,
        description: str | None = None,
        cron_expr: str | None = None,
        run_at: str | None = None,
        next_fire_at: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        def work(conn) -> dict[str, Any]:
            store = AlertStore(conn)
            resolved_next_fire = next_fire_at
            if cron_expr is not None and next_fire_at is None:
                resolved_next_fire = compute_initial_next_fire(cron_expr)
            alert = store.update_alert(
                alert_id,
                enabled=enabled,
                description=description,
                cron_expr=cron_expr,
                run_at=run_at,
                next_fire_at=resolved_next_fire,
                payload=payload,
            )
            return {
                "alert_id": alert.id,
                "enabled": alert.enabled,
                "schedule_kind": alert.schedule_kind,
                "cron_expr": alert.cron_expr,
                "run_at": alert.run_at,
                "next_fire_at": alert.next_fire_at,
                "description": alert.description,
            }

        return json.dumps(with_db(work), indent=2)

    def update_task(
        task_id: str,
        status: Literal["completed", "ignored"],
        disposition_reason: str | None = None,
    ) -> str:
        def work(conn) -> dict[str, Any]:
            task = TaskStore(conn).update_status(
                task_id,
                status=status,
                disposition_reason=disposition_reason,
            )
            return {
                "task_id": task.id,
                "status": task.status,
                "disposition_reason": task.disposition_reason,
                "completed_at": task.completed_at,
            }

        return json.dumps(with_db(work), indent=2)

    def list_alerts(window_seconds: int = 3600) -> str:
        return json.dumps(
            with_db(lambda conn: build_schedule_map(conn, window_seconds=window_seconds)),
            indent=2,
        )

    return [
        StructuredTool.from_function(
            func=create_alert,
            name="create_alert",
            description=(
                "Create a new cron or one-shot alert. Use this for new schedules and to "
                "reschedule work (after ignoring the current task)."
            ),
            args_schema=CreateAlertInput,
        ),
        StructuredTool.from_function(
            func=update_alert,
            name="update_alert",
            description=(
                "Update or disable an existing alert. Use list_alerts first to find alert_id."
            ),
            args_schema=UpdateAlertInput,
        ),
        StructuredTool.from_function(
            func=update_task,
            name="update_task",
            description=(
                "Mark a task completed or ignored. Always provide disposition_reason when "
                "ignoring."
            ),
            args_schema=UpdateTaskInput,
        ),
        StructuredTool.from_function(
            func=list_alerts,
            name="list_alerts",
            description=(
                "Return a schedule map: pending tasks, upcoming fires in the lookahead "
                "window, and all standing enabled alerts."
            ),
            args_schema=ListAlertsInput,
        ),
    ]


from tools.registry import register_tools

register_tools(create_tools)
