"""Public API for counting chat message tokens."""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import lru_cache

from langchain_core.messages import BaseMessage

from config import Settings
from llm.chat_template import chat_template_kwargs
from llm.profile import resolve_model_profile
from llm.tokenizers.huggingface import HuggingFaceTokenCounter
from llm.tokenizers.protocol import TokenCounter


@lru_cache(maxsize=8)
def _cached_hf_counter(tokenizer_hub_id: str, template_kwargs_key: str) -> HuggingFaceTokenCounter:
    """Return a cached HuggingFace token counter.

    Args:
        tokenizer_hub_id: Hugging Face tokenizer model id.
        template_kwargs_key: Stable cache key for chat-template kwargs.

    Returns:
        Token counter instance.
    """
    template_kwargs = json.loads(template_kwargs_key) if template_kwargs_key else {}
    return HuggingFaceTokenCounter(tokenizer_hub_id, template_kwargs=template_kwargs)


def get_token_counter(settings: Settings) -> TokenCounter:
    """Return the default token counter for budget and compaction checks.

    Uses the HuggingFace tokenizer for ``llm.agent_model`` (or
    ``llm.tokenizer_model`` when set). vLLM render counting is available via
    ``count_messages_tokens(..., backend=\"vllm\")`` for verification.

    Args:
        settings: Application settings.

    Returns:
        Token counter bound to the configured model tokenizer.
    """
    profile = resolve_model_profile(settings)
    kwargs_key = json.dumps(chat_template_kwargs(settings), sort_keys=True)
    return _cached_hf_counter(profile.tokenizer_hub_id, kwargs_key)


def count_messages_tokens(
    messages: Sequence[BaseMessage],
    settings: Settings,
    *,
    add_generation_prompt: bool = True,
    backend: str = "huggingface",
) -> int:
    """Count tokens for a LangChain message list.

    Args:
        messages: Messages to count (same shape as sent to ``ChatOpenAI``).
        settings: Application settings (model + tokenizer ids).
        add_generation_prompt: Include generation prefix (typical for pre-call budget).
        backend: ``huggingface`` (default) or ``vllm`` for render API verification.

    Returns:
        Token count for the serialized chat prompt.
    """
    if backend == "vllm":
        from llm.tokenizers.vllm import VllmRenderTokenCounter

        profile = resolve_model_profile(settings)
        counter = VllmRenderTokenCounter(
            base_url=profile.openai_base_url,
            model=profile.hub_id,
        )
        return counter.count_messages(
            messages,
            add_generation_prompt=add_generation_prompt,
        )

    counter = get_token_counter(settings)
    return counter.count_messages(messages, add_generation_prompt=add_generation_prompt)
