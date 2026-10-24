"""SQLite store for plan DAG nodes and dependency edges."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

from utils.time import utc_now_str

PLAN_STATUSES = frozenset({"pending", "active", "blocked", "done", "abandoned"})


@dataclass(frozen=True)
class PlanNode:
    """One node in the plan DAG."""

    id: str
    thread_id: str
    status: str
    payload: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class PlanEdge:
    """Dependency edge: ``to`` cannot start until ``from`` is done."""

    thread_id: str
    from_node_id: str
    to_node_id: str


class PlanStore:
    """CRUD and dependency queries for plan graphs."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the store.

        Args:
            conn: SQLite connection.
        """
        self._conn = conn

    def add_node(
        self,
        thread_id: str,
        *,
        status: str = "pending",
        payload: dict[str, Any] | None = None,
        node_id: str | None = None,
    ) -> PlanNode:
        """Create a plan node.

        Args:
            thread_id: Conversation thread identifier.
            status: Initial node status.
            payload: Flexible node metadata.
            node_id: Optional stable node id.

        Returns:
            Persisted plan node.

        Raises:
            ValueError: When status is invalid.
        """
        if status not in PLAN_STATUSES:
            msg = f"Invalid plan status: {status!r}"
            raise ValueError(msg)
        now = utc_now_str()
        node = PlanNode(
            id=node_id or str(uuid.uuid4()),
            thread_id=thread_id,
            status=status,
            payload=payload or {},
            created_at=now,
            updated_at=now,
        )
        self._conn.execute(
            """
            INSERT INTO plan_nodes (id, thread_id, status, payload, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                node.id,
                node.thread_id,
                node.status,
                json.dumps(node.payload),
                node.created_at,
                node.updated_at,
            ),
        )
        return node

    def update_node(
        self,
        thread_id: str,
        node_id: str,
        *,
        status: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> PlanNode:
        """Update a plan node.

        Args:
            thread_id: Conversation thread identifier.
            node_id: Target node id.
            status: Optional new status.
            payload: Optional replacement payload.

        Returns:
            Updated plan node.

        Raises:
            ValueError: When the node is missing or status is invalid.
        """
        node = self.get_node(thread_id, node_id)
        if node is None:
            msg = f"Plan node not found: {node_id}"
            raise ValueError(msg)
        if status is not None and status not in PLAN_STATUSES:
            msg = f"Invalid plan status: {status!r}"
            raise ValueError(msg)
        new_status = status or node.status
        new_payload = payload if payload is not None else node.payload
        updated_at = utc_now_str()
        self._conn.execute(
            """
            UPDATE plan_nodes
            SET status = ?, payload = ?, updated_at = ?
            WHERE id = ? AND thread_id = ?
            """,
            (new_status, json.dumps(new_payload), updated_at, node_id, thread_id),
        )
        return PlanNode(
            id=node.id,
            thread_id=node.thread_id,
            status=new_status,
            payload=new_payload,
            created_at=node.created_at,
            updated_at=updated_at,
        )

    def add_edge(self, thread_id: str, from_node_id: str, to_node_id: str) -> PlanEdge:
        """Add a dependency edge after cycle check.

        Args:
            thread_id: Conversation thread identifier.
            from_node_id: Prerequisite node.
            to_node_id: Dependent node.

        Returns:
            Persisted edge.

        Raises:
            ValueError: When the edge would create a cycle.
        """
        if self._creates_cycle(thread_id, from_node_id, to_node_id):
            msg = f"Plan edge would create a cycle: {from_node_id} -> {to_node_id}"
            raise ValueError(msg)
        edge = PlanEdge(
            thread_id=thread_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
        )
        self._conn.execute(
            """
            INSERT INTO plan_edges (thread_id, from_node_id, to_node_id)
            VALUES (?, ?, ?)
            """,
            (edge.thread_id, edge.from_node_id, edge.to_node_id),
        )
        return edge

    def list_nodes(self, thread_id: str) -> list[PlanNode]:
        """Return all plan nodes for a thread.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            Plan nodes ordered by creation time.
        """
        rows = self._conn.execute(
            """
            SELECT id, thread_id, status, payload, created_at, updated_at
            FROM plan_nodes
            WHERE thread_id = ?
            ORDER BY created_at ASC
            """,
            (thread_id,),
        ).fetchall()
        return [self._row_to_node(row) for row in rows]

    def list_edges(self, thread_id: str) -> list[PlanEdge]:
        """Return all dependency edges for a thread.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            Plan edges.
        """
        rows = self._conn.execute(
            """
            SELECT thread_id, from_node_id, to_node_id
            FROM plan_edges
            WHERE thread_id = ?
            """,
            (thread_id,),
        ).fetchall()
        return [
            PlanEdge(thread_id=row[0], from_node_id=row[1], to_node_id=row[2]) for row in rows
        ]

    def get_node(self, thread_id: str, node_id: str) -> PlanNode | None:
        """Fetch one plan node.

        Args:
            thread_id: Conversation thread identifier.
            node_id: Node id.

        Returns:
            Plan node when found, else ``None``.
        """
        row = self._conn.execute(
            """
            SELECT id, thread_id, status, payload, created_at, updated_at
            FROM plan_nodes
            WHERE thread_id = ? AND id = ?
            """,
            (thread_id, node_id),
        ).fetchone()
        return self._row_to_node(row) if row else None

    def ready_nodes(self, thread_id: str) -> list[PlanNode]:
        """Return nodes that are ready to start.

        A node is ready when status is pending/active and all predecessors are done.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            Ready plan nodes.
        """
        nodes = self.list_nodes(thread_id)
        edges = self.list_edges(thread_id)
        done_ids = {node.id for node in nodes if node.status == "done"}
        ready: list[PlanNode] = []
        for node in nodes:
            if node.status not in {"pending", "active"}:
                continue
            predecessors = [edge.from_node_id for edge in edges if edge.to_node_id == node.id]
            if all(pred_id in done_ids for pred_id in predecessors):
                ready.append(node)
        return ready

    def blocked_nodes(self, thread_id: str) -> list[PlanNode]:
        """Return nodes blocked by incomplete dependencies.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            Blocked plan nodes.
        """
        nodes = self.list_nodes(thread_id)
        edges = self.list_edges(thread_id)
        done_ids = {node.id for node in nodes if node.status == "done"}
        blocked: list[PlanNode] = []
        for node in nodes:
            if node.status in {"done", "abandoned"}:
                continue
            predecessors = [edge.from_node_id for edge in edges if edge.to_node_id == node.id]
            if predecessors and not all(pred_id in done_ids for pred_id in predecessors):
                blocked.append(node)
        return blocked

    def _creates_cycle(self, thread_id: str, from_node_id: str, to_node_id: str) -> bool:
        if from_node_id == to_node_id:
            return True
        edges = self.list_edges(thread_id)
        adjacency: dict[str, list[str]] = {}
        for edge in edges:
            adjacency.setdefault(edge.from_node_id, []).append(edge.to_node_id)
        adjacency.setdefault(from_node_id, []).append(to_node_id)

        stack = [to_node_id]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current == from_node_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency.get(current, []))
        return False

    @staticmethod
    def _row_to_node(row: tuple[Any, ...]) -> PlanNode:
        return PlanNode(
            id=row[0],
            thread_id=row[1],
            status=row[2],
            payload=json.loads(row[3]),
            created_at=row[4],
            updated_at=row[5],
        )
