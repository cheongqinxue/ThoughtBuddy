"""HuggingFace tokenizer-backed message token counting."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache

from langchain_core.messages import BaseMessage

from llm.tokenizers.messages import messages_to_openai_chat

logger = logging.getLogger("thoughtbuddy.llm.tokenizers.huggingface")


@lru_cache(maxsize=4)
def _load_tokenizer(tokenizer_hub_id: str):
    """Load and cache a HuggingFace tokenizer.

    Args:
        tokenizer_hub_id: Hugging Face model id for tokenizer weights.

    Returns:
        Loaded ``AutoTokenizer`` instance.
    """
    from transformers import AutoTokenizer

    logger.info("Loading HuggingFace tokenizer for %s", tokenizer_hub_id)
    return AutoTokenizer.from_pretrained(tokenizer_hub_id, trust_remote_code=True)


class HuggingFaceTokenCounter:
    """Count chat tokens with ``apply_chat_template`` + encode."""

    def __init__(
        self,
        tokenizer_hub_id: str,
        *,
        template_kwargs: dict | None = None,
    ) -> None:
        """Initialize the counter.

        Args:
            tokenizer_hub_id: Hugging Face model id (usually ``Settings.agent_model``).
            template_kwargs: Optional kwargs passed to ``apply_chat_template``.
        """
        self._tokenizer_hub_id = tokenizer_hub_id
        self._template_kwargs = template_kwargs or {}

    def count_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        """Count tokens for OpenAI-formatted chat messages.

        Args:
            messages: LangChain chat messages.
            add_generation_prompt: Include assistant generation prefix in count.

        Returns:
            Token count for the chat template payload.
        """
        if not messages:
            return 0

        tokenizer = _load_tokenizer(self._tokenizer_hub_id)
        openai_messages = messages_to_openai_chat(messages)
        prompt_text = tokenizer.apply_chat_template(
            openai_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self._template_kwargs,
        )
        token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        return len(token_ids)
