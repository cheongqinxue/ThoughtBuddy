"""Agent workflow graph — fixed-order wake DAG."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from context.wake_guard import get_wake_guard
from graph.phases import PHASE_INSTRUCTIONS, PHASE_ORDER, tools_for_phase
from store.messages import MessageLogStore

logger = logging.getLogger("thoughtbuddy.graph")


class GraphState(TypedDict):
    """State carried through the wake-cycle graph."""

    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
) -> CompiledStateGraph:
    """Build and compile the ThoughtBuddy wake-cycle graph.

    Phases run sequentially: Orient → Observe → Measure → Reason → Plan → Act/rest.
    Each phase uses a ReAct subgraph with phase-appropriate tools.

    Args:
        model: Chat model used by all phase subgraphs.
        tools: Full tool list for the wake.

    Returns:
        Compiled graph ready for invocation.
    """
    all_tools = list(tools or [])
    phase_graphs = {
        phase: create_react_agent(model, tools_for_phase(phase, all_tools))
        for phase in PHASE_ORDER
    }

    graph = StateGraph(GraphState)
    for phase in PHASE_ORDER:
        graph.add_node(phase, _make_phase_node(phase, phase_graphs[phase]))

    graph.add_edge(START, PHASE_ORDER[0])
    for left, right in zip(PHASE_ORDER[:-1], PHASE_ORDER[1:], strict=True):
        graph.add_edge(left, right)
    graph.add_edge(PHASE_ORDER[-1], END)
    return graph.compile()


def _make_phase_node(phase: str, react_graph: CompiledStateGraph):
    """Create one wake-phase node.

    Args:
        phase: Phase name.
        react_graph: ReAct subgraph for the phase.

    Returns:
        LangGraph node callable.
    """
    instruction = PHASE_INSTRUCTIONS[phase]

    def node(state: GraphState) -> dict[str, list[BaseMessage]]:
        guard = get_wake_guard()
        if guard is not None:
            from context.guard import build_phase_input_messages

            input_messages = build_phase_input_messages(
                guard.settings,
                guard.conn,
                guard.thread_id,
                phase,
                pending_tasks=guard.pending_tasks,
                tools=guard.tools,
                compaction_model=guard.compaction_model,
                callbacks=guard.callbacks,
            )
            base_len = len(input_messages)
        else:
            base_len = len(state["messages"])
            input_messages = [*state["messages"], HumanMessage(content=instruction)]

        logger.info("phase_start phase=%s input_messages=%s", phase, len(input_messages))
        try:
            result = react_graph.invoke({"messages": input_messages})
        except Exception:
            logger.exception("phase_failed phase=%s", phase)
            raise

        result_messages = result.get("messages", input_messages)
        new_messages = result_messages[base_len:]
        logger.info("phase_end phase=%s new_messages=%s", phase, len(new_messages))

        if guard is not None:
            if new_messages:
                MessageLogStore(guard.conn).append_many(guard.thread_id, new_messages)
                guard.conn.commit()
            return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

        return {"messages": new_messages}

    return node
