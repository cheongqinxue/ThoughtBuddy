"""Token counting for context budget checks (delegates to ``llm.tokens``)."""

from __future__ import annotations

from langchain_core.messages import BaseMessage

from config import Settings
from llm.tokens import count_messages_tokens


def estimate_message_tokens(message: BaseMessage, settings: Settings | None = None) -> int:
    """Count tokens for one message using the configured model tokenizer.

    Args:
        message: LangChain message.
        settings: Application settings. Loads defaults when omitted.

    Returns:
        Token count for the message within a single-message chat template.
    """
    resolved = settings or Settings.load()
    return count_messages_tokens([message], resolved, add_generation_prompt=False)


def estimate_messages_tokens(
    messages: list[BaseMessage],
    settings: Settings | None = None,
    *,
    add_generation_prompt: bool = True,
) -> int:
    """Count tokens for a message list using the configured model tokenizer.

    Args:
        messages: Messages to count.
        settings: Application settings. Loads defaults when omitted.
        add_generation_prompt: Include generation prefix (default for wake budget).

    Returns:
        Token count for the serialized chat prompt.
    """
    if not messages:
        return 0
    resolved = settings or Settings.load()
    return count_messages_tokens(
        messages,
        resolved,
        add_generation_prompt=add_generation_prompt,
    )
