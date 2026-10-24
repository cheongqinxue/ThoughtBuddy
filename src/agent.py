"""High-level agent helpers."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from config import Settings
from graph.builder import build_graph
from llm import create_chat_model
from tools import get_tools
from utils.observability import (
    apply_studio_langfuse_config,
    flush_langfuse,
    langfuse_direct_trace,
)


def create_agent(settings: Settings | None = None) -> CompiledStateGraph:
    """Create a compiled ThoughtBuddy agent graph.

    Args:
        settings: Optional settings override. Defaults to environment-backed
            settings when omitted.

    Returns:
        Compiled LangGraph agent.
    """
    resolved_settings = settings or Settings.load()
    model = create_chat_model(resolved_settings)
    tools = get_tools(resolved_settings)
    return build_graph(model, tools=tools)


def run_agent(
    graph: CompiledStateGraph,
    user_message: str,
    *,
    thread_id: str | None = None,
    settings: Settings | None = None,
) -> str:
    """Run one turn of the agent and return the assistant reply text.

    Args:
        graph: Compiled ThoughtBuddy graph.
        user_message: User input for this turn.
        thread_id: Conversation thread identifier for checkpointing hooks.
        settings: Optional settings override used for the default thread id.

    Returns:
        Assistant response content as plain text.
    """
    resolved_settings = settings or Settings.load()
    resolved_thread_id = thread_id or resolved_settings.default_thread_id
    with langfuse_direct_trace(resolved_settings, thread_id=resolved_thread_id) as langfuse_trace:
        invoke_config = (
            langfuse_trace.invoke_config(resolved_thread_id)
            if langfuse_trace
            else {"configurable": {"thread_id": resolved_thread_id}}
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=invoke_config,
        )
        flush_langfuse(resolved_settings)
    last_message = result["messages"][-1]
    if not isinstance(last_message, AIMessage):
        msg = "Expected the final graph message to be an assistant reply."
        raise TypeError(msg)
    content = last_message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(text_parts)
    return str(content)


def make_graph(_config: RunnableConfig) -> CompiledStateGraph:
    """Build the ThoughtBuddy graph for LangGraph server loading.

    Args:
        _config: LangGraph runtime configuration (unused).

    Returns:
        Compiled ThoughtBuddy agent graph.
    """
    settings = Settings.load()
    graph = create_agent(settings)
    return apply_studio_langfuse_config(graph, settings)
