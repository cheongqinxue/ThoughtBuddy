"""Build pillar snapshot blocks for model context."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from config import Settings
from knowledge.chroma import build_chroma_backend
from store.plans import PlanStore
from store.world import WorldStore

PLAN_PREFIX = "Plan summary:\n"
WORLD_PREFIX = "World snapshot:\n"
KNOWLEDGE_PREFIX = "Knowledge recall:\n"
SYSTEM_PREFIX = "Owner instructions:\n"


def build_plan_summary(conn: sqlite3.Connection, thread_id: str) -> str:
    """Build a plan DAG summary block.

    Args:
        conn: SQLite connection.
        thread_id: Conversation thread identifier.

    Returns:
        JSON summary string.
    """
    store = PlanStore(conn)
    nodes = store.list_nodes(thread_id)
    edges = store.list_edges(thread_id)
    payload: dict[str, Any] = {
        "nodes": [
            {"id": node.id, "status": node.status, "payload": node.payload} for node in nodes
        ],
        "edges": [{"from": edge.from_node_id, "to": edge.to_node_id} for edge in edges],
        "ready": [node.id for node in store.ready_nodes(thread_id)],
        "blocked": [node.id for node in store.blocked_nodes(thread_id)],
    }
    return json.dumps(payload, indent=2)


def build_world_snapshot(conn: sqlite3.Connection, thread_id: str) -> str:
    """Build a world observations and virtual state snapshot block.

    Args:
        conn: SQLite connection.
        thread_id: Conversation thread identifier.

    Returns:
        JSON snapshot string.
    """
    store = WorldStore(conn)
    observations = store.list_recent(thread_id, limit=5)
    payload = {
        "observations": [
            {
                "source": obs.source,
                "summary": obs.summary,
                "payload": obs.payload,
                "created_at": obs.created_at,
            }
            for obs in observations
        ],
        "state": store.list_state(thread_id),
    }
    return json.dumps(payload, indent=2)


def build_knowledge_recall(settings: Settings, *, query: str, top_k: int = 5) -> str:
    """Build a knowledge recall block for context assembly.

    Args:
        settings: Application settings.
        query: Recall query (typically goal text).
        top_k: Maximum entries to include.

    Returns:
        JSON recall string.
    """
    backend = build_chroma_backend(settings)
    hits = backend.search(query, top_k=top_k)
    payload = [
        {"id": hit.id, "text": hit.text, "metadata": hit.metadata, "score": hit.score}
        for hit in hits
    ]
    return json.dumps(payload, indent=2)
