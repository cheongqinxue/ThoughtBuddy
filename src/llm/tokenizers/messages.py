"""Convert LangChain messages to OpenAI chat payloads for tokenization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_openai.chat_models.base import _convert_message_to_dict


def messages_to_openai_chat(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages to OpenAI chat/completions message dicts.

    Uses the same conversion path as ``ChatOpenAI`` so token counts align with
    what vLLM receives over the wire.

    Args:
        messages: LangChain messages to convert.

    Returns:
        OpenAI-style message dictionaries.
    """
    return [_convert_message_to_dict(message) for message in messages]
