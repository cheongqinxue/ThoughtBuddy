"""Live vLLM integration tests — structured tool_calls required for the harness."""

from __future__ import annotations

import httpx
import pytest

from integration.support import vllm_model_name

pytestmark = pytest.mark.integration

_CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Expression to evaluate"},
            },
            "required": ["expression"],
        },
    },
}


class TestVllmToolCalls:
    """Verify the local inference server returns OpenAI-style tool_calls."""

    def test_models_endpoint_responds(self, live_vllm_base_url: str) -> None:
        """vLLM should expose the configured model via /v1/models."""
        response = httpx.get(f"{live_vllm_base_url}/models", timeout=10.0)
        response.raise_for_status()
        payload = response.json()
        model_ids = {item["id"] for item in payload.get("data", [])}
        assert vllm_model_name() in model_ids

    def test_chat_completion_returns_tool_calls(self, live_vllm_base_url: str) -> None:
        """tool_choice=auto should produce structured tool_calls, not narration."""
        request_body = {
            "model": vllm_model_name(),
            "messages": [
                {"role": "user", "content": "What is 2+2? Use the calculator tool."},
            ],
            "max_tokens": 256,
            "tool_choice": "auto",
            "tools": [_CALCULATOR_TOOL],
        }
        if "qwen3" in vllm_model_name().lower():
            request_body["chat_template_kwargs"] = {"enable_thinking": False}

        response = httpx.post(
            f"{live_vllm_base_url}/chat/completions",
            json=request_body,
            timeout=120.0,
        )
        response.raise_for_status()
        payload = response.json()

        choice = payload["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls")

        assert tool_calls, (
            "Expected structured tool_calls from vLLM. "
            "If the model narrates instead, check --tool-call-parser (hermes for Qwen)."
        )
        assert tool_calls[0]["function"]["name"] == "calculator"
        assert choice.get("finish_reason") == "tool_calls"
