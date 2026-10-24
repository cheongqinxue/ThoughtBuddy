"""Chat-template kwargs shared by inference and token counting."""

from __future__ import annotations

from typing import Any

from config import Settings


def chat_template_kwargs(settings: Settings) -> dict[str, Any]:
    """Return kwargs for Qwen3 ``apply_chat_template`` / vLLM ``extra_body``.

    Args:
        settings: Application settings.

    Returns:
        Empty dict when no template overrides apply.
    """
    hub_id = (settings.llm_tokenizer_model.strip() or settings.agent_model).lower()
    if "qwen3" in hub_id:
        return {"enable_thinking": settings.llm_enable_thinking}
    return {}
