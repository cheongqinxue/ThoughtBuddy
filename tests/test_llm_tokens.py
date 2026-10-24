"""Tests for LLM profile and token counting."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from config import Settings
from llm.profile import resolve_model_profile
from llm.tokenizers.huggingface import HuggingFaceTokenCounter
from llm.tokens import count_messages_tokens


class TestModelProfile:
    """Model profile resolution."""

    def test_tokenizer_defaults_to_agent_model(self) -> None:
        """Empty tokenizer_model should fall back to agent_model."""
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="Qwen/Qwen3-8B-AWQ",
            langsearch_api_key="key",
        )
        profile = resolve_model_profile(settings)
        assert profile.tokenizer_hub_id == "Qwen/Qwen3-8B-AWQ"
        assert profile.hub_id == "Qwen/Qwen3-8B-AWQ"
        assert profile.max_prompt_tokens == settings.llm_context_window_tokens - settings.llm_max_tokens

    def test_explicit_tokenizer_override(self) -> None:
        """tokenizer_model override should be honored."""
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="Qwen/Qwen3-8B-AWQ",
            llm_tokenizer_model="Qwen/Qwen3-8B",
            langsearch_api_key="key",
        )
        profile = resolve_model_profile(settings)
        assert profile.tokenizer_hub_id == "Qwen/Qwen3-8B"


class TestHuggingFaceTokenCounter:
    """HF-backed counting with a mocked tokenizer."""

    def test_count_messages_uses_chat_template_and_encode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Counter should apply chat template then encode the rendered prompt."""
        fake_tokenizer = MagicMock()
        fake_tokenizer.apply_chat_template.return_value = "<prompt>"
        fake_tokenizer.encode.return_value = [1, 2, 3, 4]

        monkeypatch.setattr(
            "llm.tokenizers.huggingface._load_tokenizer",
            lambda _hub_id: fake_tokenizer,
        )

        counter = HuggingFaceTokenCounter("test-model")
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="user"),
        ]
        count = counter.count_messages(messages)

        assert count == 4
        fake_tokenizer.apply_chat_template.assert_called_once()
        fake_tokenizer.encode.assert_called_once_with("<prompt>", add_special_tokens=False)

    def test_count_messages_tokens_uses_settings_profile(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Public API should route through the configured tokenizer hub id."""
        fake_tokenizer = MagicMock()
        fake_tokenizer.apply_chat_template.return_value = "hello"
        fake_tokenizer.encode.return_value = [10, 11]

        monkeypatch.setattr(
            "llm.tokenizers.huggingface._load_tokenizer",
            lambda _hub_id: fake_tokenizer,
        )

        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test-model",
            langsearch_api_key="key",
        )
        count = count_messages_tokens([HumanMessage(content="hi")], settings)
        assert count == 2
