"""Starlette app with ThoughtBuddy lifespan hooks for LangGraph dev."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from experiments.chess.plugin.routes import CHESS_ROUTES
from runtime.service import _set_active_runtime, create_runtime
from store.tasks import TaskStore
from utils.logging import configure_observability

logger = logging.getLogger(__name__)

_runtime = None


@asynccontextmanager
async def lifespan(app: Starlette):
    """Start ThoughtBuddy infrastructure when LangGraph boots."""
    global _runtime
    _runtime = create_runtime()
    configure_observability(_runtime.settings)
    app.state.runtime = _runtime
    _set_active_runtime(_runtime)
    await _runtime.start()
    try:
        yield
    finally:
        await _runtime.stop()
        _set_active_runtime(None)
        _runtime = None


async def health(_: Request) -> JSONResponse:
    """Return runtime health information."""
    if _runtime is None:
        return JSONResponse({"status": "starting"})
    stats = TaskStore(_runtime.conn).count_by_status()
    if _runtime.is_halted():
        from runtime.failure_report import experiment_failure_path

        failure_path = experiment_failure_path(_runtime.settings)
        return JSONResponse(
            {
                "status": "halted",
                "halt_reason": _runtime.halt_reason,
                "halt_task_ids": _runtime.halt_task_ids or [],
                "failure_report": str(failure_path) if failure_path else None,
                "heartbeat_interval_seconds": _runtime.heartbeat.interval_seconds,
                "task_counts": stats,
            },
            status_code=503,
        )
    return JSONResponse(
        {
            "status": "ok",
            "heartbeat_interval_seconds": _runtime.heartbeat.interval_seconds,
            "task_counts": stats,
        },
    )


async def enqueue_user_task(request: Request) -> JSONResponse:
    """Enqueue a user message as a task for the next wake cycle."""
    if _runtime is None:
        return JSONResponse({"error": "runtime not ready"}, status_code=503)
    if _runtime.is_halted():
        return JSONResponse(
            {"error": "runtime halted", "halt_reason": _runtime.halt_reason},
            status_code=503,
        )

    body = await request.json()
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        return JSONResponse({"error": "message is required"}, status_code=400)

    payload = body.get("payload")
    if payload is not None and not isinstance(payload, dict):
        return JSONResponse({"error": "payload must be an object"}, status_code=400)

    task = TaskStore(_runtime.conn).enqueue_user_task(
        message=message.strip(),
        payload=payload,
    )
    _runtime.conn.commit()
    _runtime.heartbeat.notify()
    return JSONResponse({"task_id": task.id, "status": task.status}, status_code=201)


async def enqueue_event_task(request: Request) -> JSONResponse:
    """Enqueue an event/callback task and signal the heartbeat."""
    if _runtime is None:
        return JSONResponse({"error": "runtime not ready"}, status_code=503)
    if _runtime.is_halted():
        return JSONResponse(
            {"error": "runtime halted", "halt_reason": _runtime.halt_reason},
            status_code=503,
        )

    secret = request.headers.get("X-ThoughtBuddy-Event-Secret", "")
    expected = _runtime.settings.event_webhook_secret
    if not expected or secret != expected:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    body = await request.json()
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        return JSONResponse({"error": "message is required"}, status_code=400)

    event_key = body.get("event_key")
    if not isinstance(event_key, str) or not event_key.strip():
        return JSONResponse({"error": "event_key is required"}, status_code=400)

    payload = body.get("payload")
    if payload is not None and not isinstance(payload, dict):
        return JSONResponse({"error": "payload must be an object"}, status_code=400)

    occurrence_at = body.get("occurrence_at")
    if occurrence_at is not None and not isinstance(occurrence_at, str):
        return JSONResponse({"error": "occurrence_at must be a string"}, status_code=400)

    result = _runtime.monitors.enqueue_external_event(
        message=message.strip(),
        event_key=event_key.strip(),
        payload=payload,
        occurrence_at=occurrence_at,
    )
    if result is None:
        return JSONResponse({"status": "duplicate", "deduped": True}, status_code=200)
    return JSONResponse(result, status_code=201)


async def queue_overview(_: Request) -> JSONResponse:
    """Return queue and alert overview for ops tooling."""
    if _runtime is None:
        return JSONResponse({"error": "runtime not ready"}, status_code=503)

    from store.alerts import AlertStore

    pending = TaskStore(_runtime.conn).list_pending()
    alerts = AlertStore(_runtime.conn).list_enabled()
    stats = TaskStore(_runtime.conn).count_by_status()
    return JSONResponse(
        {
            "task_counts": stats,
            "pending_tasks": [
                {
                    "id": task.id,
                    "source": task.source,
                    "created_at": task.created_at,
                    "payload": task.payload,
                }
                for task in pending
            ],
            "enabled_alerts": [
                {
                    "id": alert.id,
                    "description": alert.description,
                    "next_fire_at": alert.next_fire_at,
                    "schedule_kind": alert.schedule_kind,
                }
                for alert in alerts
            ],
        },
    )


def _build_routes() -> list[Route]:
    """Assemble ThoughtBuddy HTTP routes."""
    routes = [
        Route("/thoughtbuddy/health", health, methods=["GET"]),
        Route("/thoughtbuddy/queue", queue_overview, methods=["GET"]),
        Route("/thoughtbuddy/tasks/user", enqueue_user_task, methods=["POST"]),
        Route("/thoughtbuddy/tasks/event", enqueue_event_task, methods=["POST"]),
    ]
    for path, handler, methods in CHESS_ROUTES:
        routes.append(Route(path, handler, methods=methods))
    return routes


app = Starlette(lifespan=lifespan, routes=_build_routes())
