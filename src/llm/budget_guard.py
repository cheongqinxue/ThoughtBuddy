"""Chat model wrapper that enforces prompt budget before every LLM call."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field
from typing_extensions import Self, override

from config import Settings


class BudgetGuardChatModel(BaseChatModel):
    """Delegate to an inner chat model after token budget validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: Any = Field(exclude=True)
    settings: Settings = Field(exclude=True)

    @property
    @override
    def _llm_type(self) -> str:
        return f"budget-guard-{self.inner._llm_type}"

    @property
    @override
    def _identifying_params(self) -> dict[str, Any]:
        return self.inner._identifying_params

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool | Callable[..., Any]],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Return a budget-guard wrapper around the tool-bound inner model."""
        bound = self.inner.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        return self.__class__(inner=bound, settings=self.settings)

    def _prepare_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Validate token budget for one LLM request.

        Args:
            messages: Messages about to be sent to the inner model.

        Returns:
            The same message list when within budget.
        """
        from context.guard import ensure_llm_prompt_budget

        return ensure_llm_prompt_budget(messages, self.settings)

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prepared = self._prepare_messages(messages)
        return self.inner._generate(prepared, stop=stop, run_manager=run_manager, **kwargs)

    @override
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prepared = self._prepare_messages(messages)
        return await self.inner._agenerate(
            prepared,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prepared = self._prepare_messages(messages)
        yield from self.inner._stream(prepared, stop=stop, run_manager=run_manager, **kwargs)

    @override
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        prepared = self._prepare_messages(messages)
        async for chunk in self.inner._astream(
            prepared,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        ):
            yield chunk


def wrap_with_budget_guard(model: BaseChatModel, settings: Settings) -> BaseChatModel:
    """Wrap a chat model with pre-invocation budget validation.

    Args:
        model: Inner chat model.
        settings: Application settings for token budgets.

    Returns:
        Wrapped model, or the original instance when already guarded.
    """
    if isinstance(model, BudgetGuardChatModel):
        return model
    return BudgetGuardChatModel(inner=model, settings=settings)
