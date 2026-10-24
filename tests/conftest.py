"""Shared pytest fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import pytest

from config import Settings


@pytest.fixture
def settings() -> Settings:
    """Return deterministic settings for unit tests."""
    return Settings(
        openai_base_url="http://localhost:8000/v1",
        openai_api_key="test-key",
        agent_model="test-model",
        log_level="WARNING",
        langsearch_api_key="test-langsearch-key",
    )


@pytest.fixture
def fake_chat_model() -> FakeListChatModel:
    """Return a deterministic fake chat model."""
    return FakeListChatModel(responses=["Test assistant reply."])


@pytest.fixture
def mock_hf_token_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a deterministic tokenizer stub (no Hugging Face hub access)."""
    from llm.tokens import _cached_hf_counter

    fake_tokenizer = MagicMock()

    def _apply_chat_template(
        messages: list,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        parts = []
        for message in messages:
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            parts.append(str(content))
        return "\n".join(parts)

    fake_tokenizer.apply_chat_template.side_effect = _apply_chat_template
    fake_tokenizer.encode.side_effect = (
        lambda text, add_special_tokens=False: list(range(max(1, len(text) // 4)))
    )

    monkeypatch.setattr(
        "llm.tokenizers.huggingface._load_tokenizer",
        lambda _hub_id: fake_tokenizer,
    )
    _cached_hf_counter.cache_clear()
    yield
    _cached_hf_counter.cache_clear()
