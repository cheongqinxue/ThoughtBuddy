"""Cron iteration helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from croniter import croniter


def next_cron_fire(cron_expr: str, base: datetime) -> datetime:
    """Return the cron occurrence strictly after ``base``.

    Args:
        cron_expr: Cron expression.
        base: Base datetime (timezone-aware preferred).

    Returns:
        Next fire time in UTC.
    """
    iterator = croniter(cron_expr, base)
    nxt = iterator.get_next(datetime)
    if nxt.tzinfo is None:
        nxt = nxt.replace(tzinfo=UTC)
    return nxt.astimezone(UTC).replace(microsecond=0)
