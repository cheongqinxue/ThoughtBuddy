"""World observation tools for the agent."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config import Settings
from db.connection import connect
from store.world import WorldStore


class WorldObserveInput(BaseModel):
    """Input schema for ``world_observe``."""

    limit: int = Field(default=10, ge=1, le=100, description="Maximum observations to return.")


def create_tools(settings: Settings) -> list[StructuredTool]:
    """Create world observation tools.

    Args:
        settings: Application settings.

    Returns:
        LangChain tools for reading world observations and state.
    """
    db_path = Path(settings.thoughtbuddy_db_path)
    thread_id = settings.default_thread_id

    def with_store(action):
        conn = connect(db_path)
        try:
            result = action(WorldStore(conn))
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def world_observe(limit: int = 10) -> str:
        """Read recent world observations and virtual state snapshot."""
        observations = with_store(lambda store: store.list_recent(thread_id, limit=limit))
        state = with_store(lambda store: store.list_state(thread_id))
        payload = {
            "observations": [
                {
                    "id": obs.id,
                    "source": obs.source,
                    "summary": obs.summary,
                    "payload": obs.payload,
                    "created_at": obs.created_at,
                }
                for obs in observations
            ],
            "state": state,
        }
        return json.dumps(payload, indent=2)

    return [
        StructuredTool.from_function(
            func=world_observe,
            name="world_observe",
            description=(
                "Read recent world observations (web and virtual) and current virtual state."
            ),
            args_schema=WorldObserveInput,
        ),
    ]


from tools.registry import register_tools

register_tools(create_tools)


def log_observation(
    settings: Settings,
    *,
    source: str,
    payload: dict,
    summary: str | None = None,
    thread_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Append a world observation (helper for web tools and experiments).

    Args:
        settings: Application settings.
        source: Observation channel name.
        payload: Raw observation payload.
        summary: Optional short summary.
        thread_id: Thread id override.
        conn: Optional existing SQLite connection (caller owns the transaction).
    """
    resolved_thread = thread_id or settings.default_thread_id
    if conn is not None:
        WorldStore(conn).append_observation(
            resolved_thread,
            source=source,
            payload=payload,
            summary=summary,
        )
        return

    db_path = Path(settings.thoughtbuddy_db_path)
    owned = connect(db_path)
    try:
        WorldStore(owned).append_observation(
            resolved_thread,
            source=source,
            payload=payload,
            summary=summary,
        )
        owned.commit()
    finally:
        owned.close()
