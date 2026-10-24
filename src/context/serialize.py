"""Serialize LangChain messages for SQLite storage."""

from __future__ import annotations

import json

from langchain_core.load import dumpd, loads
from langchain_core.messages import BaseMessage


def serialize_message(message: BaseMessage) -> str:
    """Serialize a message to a JSON string.

    Args:
        message: LangChain message to persist.

    Returns:
        JSON string suitable for ``message_log.content_json``.
    """
    return json.dumps(dumpd(message))


def deserialize_message(content_json: str) -> BaseMessage:
    """Deserialize a message from JSON storage.

    Args:
        content_json: JSON string from ``message_log``.

    Returns:
        Restored LangChain message.
    """
    return loads(content_json, allowed_objects="messages")
