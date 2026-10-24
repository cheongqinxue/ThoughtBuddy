"""Wake-cycle phase definitions for the agent graph."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from langchain_core.tools import BaseTool

PHASE_ORDER: tuple[str, ...] = (
    "orient",
    "observe",
    "measure",
    "reason",
    "plan",
    "act",
)

PHASE_INSTRUCTIONS: dict[str, str] = {
    "orient": (
        "Phase: ORIENT. Confirm why you were woken, pending tasks, and schedule context. "
        "Use list_alerts if you need the full schedule map. Summarize your situation briefly."
    ),
    "observe": (
        "Phase: OBSERVE. Gather external or virtual world context relevant to your goal. "
        "Use websearch, webfetch, world_observe, and experiment world tools as needed. "
        "It is valid to report no new observations."
    ),
    "measure": (
        "Phase: MEASURE. Read world snapshot and plan readiness. "
        "Use plan_read, world_observe, history_read, and history_search. "
        "Review SCRATCHPAD.md (already in context) for self-tracked state. "
        "for your self-tracked state. Do not take action yet."
    ),
    "reason": (
        "Phase: REASON. Search and update knowledge; review history when needed. "
        "Use knowledge_search, knowledge_add, knowledge_edit, history_read, history_search."
    ),
    "plan": (
        "Phase: PLAN. Update the plan DAG for the central goal. "
        "Use plan_read, plan_add_node, plan_update_node, and plan_link."
    ),
    "act": (
        "Phase: ACT or REST. Execute prioritized work or deliberately rest. "
        "Use scheduling, world, experiment, scratchpad, and edit_progress tools. "
        "Refresh PROGRESS.md before finishing. Rest (no further tools) is valid."
    ),
}


def _name_in(*names: str) -> Callable[[str], bool]:
    allowed = set(names)

    def _match(name: str) -> bool:
        return name in allowed

    return _match


def _prefix(*prefixes: str) -> Callable[[str], bool]:
    def _match(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in prefixes)

    return _match


PHASE_TOOL_MATCHERS: dict[str, Callable[[str], bool]] = {
    "orient": _name_in("list_alerts"),
    "observe": lambda name: name in {"websearch", "webfetch", "world_observe"} or name.startswith(
        "world_",
    ),
    "measure": _name_in("plan_read", "world_observe", "history_read", "history_search"),
    "reason": lambda name: name.startswith("knowledge_") or name.startswith("history_"),
    "plan": _prefix("plan_"),
    "act": lambda _name: True,
}


def tools_for_phase(phase: str, tools: Sequence[BaseTool]) -> list[BaseTool]:
    """Filter tools available during one wake phase.

    Args:
        phase: Wake phase name.
        tools: Full tool list for the wake.

    Returns:
        Tools whose names match the phase policy.
    """
    matcher = PHASE_TOOL_MATCHERS[phase]
    return [tool for tool in tools if matcher(tool.name)]
