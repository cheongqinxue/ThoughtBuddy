# `graph/` — agent workflow (wake-cycle DAG)

Source of truth for the agent's LangGraph architecture. One compiled graph per wake:
**Orient → Observe → Measure → Reason → Plan → Act/rest**.

Docs: [`docs/HARNESS.md`](../../docs/HARNESS.md) — entry points, one wake round, phase tool policy.

## Files

| File | Role |
|---|---|
| [`builder.py`](builder.py) | `build_graph()` — compiles the fixed-order phase DAG |
| [`phases.py`](phases.py) | `PHASE_ORDER`, phase instructions, per-phase tool filters |

## Graph shape

Each top-level node is a **ReAct subgraph** with phase-filtered tools. State: `GraphState.messages` (append-only).

Entry: `agent.create_agent()` → `build_graph(model, tools)`.

## Tests

| Test file | Covers |
|---|---|
| `tests/test_graph.py` | Graph invoke + phase order / tool filters |
