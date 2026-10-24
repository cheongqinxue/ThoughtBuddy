"""Plain-text extraction from archived LangChain messages."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage


def message_content_text(message: BaseMessage) -> str:
    """Extract human-readable text from a message body.

    Args:
        message: LangChain message.

    Returns:
        Plain-text content suitable for display and search indexing.
    """
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
        return "\n".join(part for part in parts if part)
    return str(content)


def message_search_text(message: BaseMessage) -> str:
    """Build searchable plain text for FTS indexing.

    Args:
        message: LangChain message to index.

    Returns:
        Concatenated text including tool names and arguments when present.
    """
    parts = [message_content_text(message)]
    tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            parts.append(str(tool_call.get("name", "")))
            args = tool_call.get("args")
            if args:
                parts.append(json.dumps(args, sort_keys=True))
        else:
            parts.append(str(getattr(tool_call, "name", "")))
    if isinstance(message, ToolMessage):
        parts.append(str(message.name or ""))
    return "\n".join(part for part in parts if part)


def message_to_record(row: dict[str, Any]) -> dict[str, Any]:
    """Format one archived message row for agent-facing history tools.

    Args:
        row: Dict with ``seq``, ``message_type``, ``created_at``, and ``message``.

    Returns:
        JSON-serializable history entry.
    """
    message: BaseMessage = row["message"]
    record: dict[str, Any] = {
        "seq": row["seq"],
        "created_at": row["created_at"],
        "type": row["message_type"],
        "content": message_content_text(message),
    }
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        record["tool_calls"] = [
            {
                "name": call.get("name") if isinstance(call, dict) else getattr(call, "name", None),
                "args": call.get("args") if isinstance(call, dict) else getattr(call, "args", None),
            }
            for call in tool_calls
        ]
    if isinstance(message, ToolMessage) and message.name:
        record["tool_name"] = message.name
    return record
