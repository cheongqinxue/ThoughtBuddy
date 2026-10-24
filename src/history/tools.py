"""Agent tools for reviewing the append-only message archive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config import Settings
from db.connection import connect
from store.messages import HistoryCursor, MessageLogStore


class HistoryReadInput(BaseModel):
    """Input schema for ``history_read``."""

    start_at: str | None = Field(
        default=None,
        description="Inclusive lower bound (UTC YYYY-MM-DD HH:MM:SS or ISO-8601).",
    )
    end_at: str | None = Field(
        default=None,
        description="Exclusive upper bound (UTC).",
    )
    limit: int = Field(default=50, ge=1, le=100, description="Maximum messages to return.")
    order: Literal["asc", "desc"] = Field(
        default="asc",
        description="Sort by timestamp ascending (oldest first) or descending (newest first).",
    )
    cursor: str | None = Field(
        default=None,
        description="Pagination cursor from a prior history_read response (next_cursor).",
    )


class HistorySearchInput(BaseModel):
    """Input schema for ``history_search``."""

    query: str = Field(description="Free-text search over archived messages (BM25).")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum hits to return.")


def create_tools(settings: Settings) -> list[StructuredTool]:
    """Create message-history review tools.

    Args:
        settings: Application settings.

    Returns:
        LangChain tools for paginated transcript read and BM25 search.
    """
    db_path = Path(settings.thoughtbuddy_db_path)
    thread_id = settings.default_thread_id

    def history_read(
        start_at: str | None = None,
        end_at: str | None = None,
        limit: int = 50,
        order: Literal["asc", "desc"] = "asc",
        cursor: str | None = None,
    ) -> str:
        """Read archived messages paginated by UTC timestamp."""
        conn = connect(db_path)
        try:
            parsed_cursor = HistoryCursor.decode(cursor) if cursor else None
            page = MessageLogStore(conn).list_by_time(
                thread_id,
                start_at=start_at,
                end_at=end_at,
                limit=limit,
                order=order,
                cursor=parsed_cursor,
            )
            payload = {
                "total_messages": page.total_count,
                "order": page.order,
                "count": len(page.messages),
                "next_cursor": page.next_cursor,
                "prev_cursor": page.prev_cursor,
                "messages": page.messages,
            }
            return json.dumps(payload, indent=2)
        finally:
            conn.close()

    def history_search(query: str, limit: int = 20) -> str:
        """Search archived messages with BM25 ranking."""
        conn = connect(db_path)
        try:
            hits = MessageLogStore(conn).search(thread_id, query=query, limit=limit)
            payload = {
                "query": query,
                "count": len(hits),
                "hits": [
                    {
                        "seq": hit.seq,
                        "created_at": hit.created_at,
                        "type": hit.message_type,
                        "snippet": hit.snippet,
                        "rank": hit.rank,
                    }
                    for hit in hits
                ],
            }
            return json.dumps(payload, indent=2)
        finally:
            conn.close()

    return [
        StructuredTool.from_function(
            func=history_read,
            name="history_read",
            description=(
                "Read archived wake history paginated by UTC timestamp. "
                "Use start_at/end_at to bound a time window (e.g. trading session). "
                "Pass next_cursor to continue."
            ),
            args_schema=HistoryReadInput,
        ),
        StructuredTool.from_function(
            func=history_search,
            name="history_search",
            description=(
                "Search the full message archive with BM25 keyword ranking. "
                "Find tool calls, moves, queries, and outcomes across all wakes."
            ),
            args_schema=HistorySearchInput,
        ),
    ]


from tools.registry import register_tools

register_tools(create_tools)
