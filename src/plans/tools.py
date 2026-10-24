"""Plan DAG tools for the agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config import Settings
from db.connection import connect
from store.plans import PlanStore


class PlanAddNodeInput(BaseModel):
    """Input schema for ``plan_add_node``."""

    description: str = Field(description="Human-readable node description.")
    status: str = Field(default="pending", description="Initial status.")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Optional progress metrics.")


class PlanUpdateNodeInput(BaseModel):
    """Input schema for ``plan_update_node``."""

    node_id: str = Field(description="Plan node id.")
    status: str | None = Field(default=None, description="New status, if changing.")
    description: str | None = Field(default=None, description="Updated description.")
    metrics: dict[str, Any] | None = Field(default=None, description="Updated metrics.")


class PlanLinkInput(BaseModel):
    """Input schema for ``plan_link``."""

    from_node_id: str = Field(description="Prerequisite node id.")
    to_node_id: str = Field(description="Dependent node id (waits for prerequisite).")


def create_tools(settings: Settings) -> list[StructuredTool]:
    """Create plan DAG tools.

    Args:
        settings: Application settings.

    Returns:
        LangChain tools for plan graph management.
    """
    db_path = Path(settings.thoughtbuddy_db_path)
    thread_id = settings.default_thread_id

    def with_store(action):
        conn = connect(db_path)
        try:
            result = action(PlanStore(conn))
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def plan_read() -> str:
        """Read plan nodes, dependency edges, ready nodes, and blocked nodes."""
        def _read(store: PlanStore) -> dict[str, Any]:
            nodes = store.list_nodes(thread_id)
            edges = store.list_edges(thread_id)
            return {
                "nodes": [
                    {
                        "id": node.id,
                        "status": node.status,
                        "payload": node.payload,
                    }
                    for node in nodes
                ],
                "edges": [
                    {"from": edge.from_node_id, "to": edge.to_node_id} for edge in edges
                ],
                "ready": [node.id for node in store.ready_nodes(thread_id)],
                "blocked": [node.id for node in store.blocked_nodes(thread_id)],
            }

        return json.dumps(with_store(_read), indent=2)

    def plan_add_node(
        description: str,
        status: str = "pending",
        metrics: dict[str, Any] | None = None,
    ) -> str:
        """Add a plan node."""
        payload = {"description": description, "metrics": metrics or {}}

        def _add(store: PlanStore) -> dict[str, Any]:
            node = store.add_node(thread_id, status=status, payload=payload)
            return {"ok": True, "node_id": node.id, "status": node.status}

        return json.dumps(with_store(_add), indent=2)

    def plan_update_node(
        node_id: str,
        status: str | None = None,
        description: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> str:
        """Update a plan node."""
        def _update(store: PlanStore) -> dict[str, Any]:
            existing = store.get_node(thread_id, node_id)
            if existing is None:
                return {"ok": False, "error": f"Plan node not found: {node_id}"}
            payload = dict(existing.payload)
            if description is not None:
                payload["description"] = description
            if metrics is not None:
                payload["metrics"] = metrics
            node = store.update_node(thread_id, node_id, status=status, payload=payload)
            return {"ok": True, "node_id": node.id, "status": node.status}

        return json.dumps(with_store(_update), indent=2)

    def plan_link(from_node_id: str, to_node_id: str) -> str:
        """Add a dependency edge (to waits for from)."""
        def _link(store: PlanStore) -> dict[str, Any]:
            try:
                store.add_edge(thread_id, from_node_id, to_node_id)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "from": from_node_id, "to": to_node_id}

        return json.dumps(with_store(_link), indent=2)

    return [
        StructuredTool.from_function(
            func=plan_read,
            name="plan_read",
            description="Read the plan DAG: nodes, edges, ready set, and blocked set.",
        ),
        StructuredTool.from_function(
            func=plan_add_node,
            name="plan_add_node",
            description="Add a subgoal/strategy node to the plan graph.",
            args_schema=PlanAddNodeInput,
        ),
        StructuredTool.from_function(
            func=plan_update_node,
            name="plan_update_node",
            description="Update plan node status, description, or metrics.",
            args_schema=PlanUpdateNodeInput,
        ),
        StructuredTool.from_function(
            func=plan_link,
            name="plan_link",
            description="Add dependency: to_node waits until from_node is done.",
            args_schema=PlanLinkInput,
        ),
    ]


from tools.registry import register_tools

register_tools(create_tools)
