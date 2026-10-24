"""Logging filters for ThoughtBuddy handlers."""

from __future__ import annotations

import logging
import re

_REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE), "Bearer [REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,}]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*)[^\s,}]+"), r"\1[REDACTED]"),
)


class RedactionFilter(logging.Filter):
    """Redact likely secrets from log message text."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive substrings before handlers emit the record.

        Args:
            record: Log record about to be emitted.

        Returns:
            Always True so the record is not dropped.
        """
        message = record.getMessage()
        redacted = message
        for pattern, replacement in _REDACT_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True
