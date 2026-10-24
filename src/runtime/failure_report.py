"""Persist experiment failure reports for operators and run.sh."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import Settings

logger = logging.getLogger("thoughtbuddy.runtime.failure_report")


def experiment_failure_path(settings: Settings) -> Path | None:
    """Return the failure report path when an experiment is active.

    Args:
        settings: Application settings.

    Returns:
        Path under ``.data/experiments/<name>/failure.json``, or None.
    """
    if not settings.active_experiment:
        return None
    return Path(f".data/experiments/{settings.active_experiment}/failure.json")


def write_experiment_failure_report(
    settings: Settings,
    reason: str,
    *,
    task_ids: list[str] | None = None,
    wake_id: str | None = None,
    cause: str | None = None,
) -> Path | None:
    """Write a JSON failure report and log the path.

    Args:
        settings: Application settings.
        reason: Operator-facing failure summary.
        task_ids: Queue task ids involved in the failed wake.
        wake_id: Wake cycle identifier when available.
        cause: Optional exception type/message for debugging.

    Returns:
        Path written, or None when no experiment is active.
    """
    path = experiment_failure_path(settings)
    if path is None:
        return None

    payload: dict[str, Any] = {
        "status": "failed",
        "experiment": settings.active_experiment,
        "reason": reason,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "task_ids": task_ids or [],
        "wake_id": wake_id,
        "llm": {
            "agent_model": settings.agent_model,
            "openai_base_url": settings.openai_base_url,
            "max_retries": settings.llm_max_retries,
        },
    }
    if cause:
        payload["cause"] = cause[:2000]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.critical("Experiment failure report written: %s", path)
    return path


def clear_experiment_failure_report(settings: Settings) -> None:
    """Remove a stale failure report before a new experiment run.

    Args:
        settings: Application settings.
    """
    path = experiment_failure_path(settings)
    if path is not None and path.exists():
        path.unlink()
