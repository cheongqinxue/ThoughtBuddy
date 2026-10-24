"""Token counter protocol for swappable counting backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import BaseMessage


class TokenCounter(Protocol):
    """Count tokens for a chat message list using a model tokenizer."""

    def count_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        """Return token count for the given messages.

        Args:
            messages: LangChain chat messages.
            add_generation_prompt: Whether to include the model's generation prefix.

        Returns:
            Token count for the serialized chat prompt.
        """
        ...
