"""Verify HuggingFace token counts match vLLM render API."""

from __future__ import annotations

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from config import Settings
from integration.support import require_vllm, vllm_model_name
from llm.tokens import count_messages_tokens

pytestmark = pytest.mark.integration


class TestVllmTokenParity:
    """HF and vLLM token counts should agree for the served model."""

    def test_hf_count_matches_vllm_render(self) -> None:
        """Local tokenizer count should match vLLM chat/completions/render."""
        base_url = require_vllm()
        model = vllm_model_name()
        settings = Settings.load()

        messages = [
            SystemMessage(content="You are a test assistant."),
            HumanMessage(content="Count my tokens accurately."),
        ]

        hf_count = count_messages_tokens(messages, settings, backend="huggingface")
        vllm_count = count_messages_tokens(messages, settings, backend="vllm")

        assert hf_count == vllm_count

    def test_render_endpoint_available(self) -> None:
        """Sanity check vLLM render returns token_ids."""
        base_url = require_vllm()
        model = vllm_model_name()
        response = httpx.post(
            f"{base_url}/chat/completions/render",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
            },
            timeout=10.0,
        )
        response.raise_for_status()
        assert response.json().get("token_ids")
