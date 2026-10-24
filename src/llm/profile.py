"""Resolved LLM / tokenizer metadata from application settings."""

from __future__ import annotations

from dataclasses import dataclass

from config import Settings


@dataclass(frozen=True)
class ModelProfile:
    """Inference model identity and context limits."""

    hub_id: str
    tokenizer_hub_id: str
    openai_base_url: str
    context_window_tokens: int
    max_completion_tokens: int

    @property
    def max_prompt_tokens(self) -> int:
        """Maximum estimated prompt tokens before reserving completion budget."""
        return max(1, self.context_window_tokens - self.max_completion_tokens)


def resolve_model_profile(settings: Settings) -> ModelProfile:
    """Build a model profile from settings.

    Args:
        settings: Application settings.

    Returns:
        Resolved profile for inference and token counting.
    """
    tokenizer_hub = settings.llm_tokenizer_model.strip() or settings.agent_model
    return ModelProfile(
        hub_id=settings.agent_model,
        tokenizer_hub_id=tokenizer_hub,
        openai_base_url=settings.openai_base_url.rstrip("/"),
        context_window_tokens=settings.llm_context_window_tokens,
        max_completion_tokens=settings.llm_max_tokens,
    )
