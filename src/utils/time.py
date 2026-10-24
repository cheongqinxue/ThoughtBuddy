"""UTC timestamp helpers for SQLite storage."""

from __future__ import annotations

from datetime import UTC, datetime

SQLITE_DT_FORMAT = "%Y-%m-%d %H:%M:%S"


def utc_now_str() -> str:
    """Return the current UTC time formatted for SQLite comparisons.

    Returns:
        Timestamp string in ``YYYY-MM-DD HH:MM:SS`` UTC.
    """
    return datetime.now(tz=UTC).replace(microsecond=0).strftime(SQLITE_DT_FORMAT)


def parse_utc(timestamp: str) -> datetime:
    """Parse a stored UTC timestamp string.

    Args:
        timestamp: Timestamp from SQLite or legacy ISO-8601 values.

    Returns:
        Timezone-aware UTC datetime.
    """
    normalized = timestamp.replace("T", " ").replace("Z", "")
    if "+" in normalized:
        normalized = normalized.split("+", maxsplit=1)[0]
    parsed = datetime.strptime(normalized, SQLITE_DT_FORMAT)
    return parsed.replace(tzinfo=UTC)


def format_utc(value: datetime) -> str:
    """Format a datetime for SQLite storage.

    Args:
        value: Datetime to format.

    Returns:
        Timestamp string in ``YYYY-MM-DD HH:MM:SS`` UTC.
    """
    return value.astimezone(UTC).replace(microsecond=0).strftime(SQLITE_DT_FORMAT)
