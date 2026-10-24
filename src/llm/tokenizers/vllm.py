"""vLLM render API token counting (ground truth when inference server is up)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx
from langchain_core.messages import BaseMessage

from llm.tokenizers.messages import messages_to_openai_chat

logger = logging.getLogger("thoughtbuddy.llm.tokenizers.vllm")


class VllmRenderTokenCounter:
    """Count tokens via vLLM ``/v1/chat/completions/render``."""

    def __init__(self, *, base_url: str, model: str, timeout_seconds: float = 30.0) -> None:
        """Initialize the remote counter.

        Args:
            base_url: OpenAI-compatible base URL (e.g. ``http://localhost:8000/v1``).
            model: Served model name.
            timeout_seconds: HTTP timeout for render requests.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    def count_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        """Count tokens using vLLM's chat template rendering.

        Args:
            messages: LangChain chat messages.
            add_generation_prompt: Passed through to vLLM render when supported.

        Returns:
            Token count from vLLM ``token_ids``.

        Raises:
            httpx.HTTPError: When vLLM is unreachable or returns an error.
        """
        if not messages:
            return 0

        openai_messages = messages_to_openai_chat(messages)
        payload: dict = {
            "model": self._model,
            "messages": openai_messages,
        }
        if add_generation_prompt:
            payload["add_generation_prompt"] = True

        url = f"{self._base_url}/chat/completions/render"
        response = httpx.post(url, json=payload, timeout=self._timeout_seconds)
        response.raise_for_status()
        token_ids = response.json().get("token_ids") or []
        return len(token_ids)

    def is_available(self) -> bool:
        """Return whether the vLLM models endpoint responds.

        Returns:
            True when vLLM accepts HTTP connections.
        """
        try:
            response = httpx.get(f"{self._base_url}/models", timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            logger.debug("vLLM token counter unavailable at %s", self._base_url)
            return False
