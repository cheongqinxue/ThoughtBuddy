"""Chat model factory for OpenAI-compatible inference servers."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config import Settings
from llm.budget_guard import wrap_with_budget_guard
from llm.chat_template import chat_template_kwargs


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Create the chat model used by the agent graph.

    Args:
        settings: Application settings containing model and API configuration.

    Returns:
        Configured chat model instance.
    """
    template_kwargs = chat_template_kwargs(settings)
    extra_body = (
        {"chat_template_kwargs": template_kwargs} if template_kwargs else None
    )
    model = ChatOpenAI(
        model=settings.agent_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        max_retries=settings.llm_max_retries,
        extra_body=extra_body,
    )
    return wrap_with_budget_guard(model, settings)
