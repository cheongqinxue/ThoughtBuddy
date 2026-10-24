"""Append-only message archive."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass

from langchain_core.messages import BaseMessage

from context.message_text import message_search_text
from context.serialize import deserialize_message, serialize_message
from utils.time import format_utc, parse_utc, utc_now_str


@dataclass(frozen=True)
class ArchivedMessage:
    """One row from ``message_log``."""

    thread_id: str
    seq: int
    message_type: str
    content_json: str
    created_at: str


@dataclass(frozen=True)
class HistoryCursor:
    """Pagination cursor keyed by timestamp and sequence."""

    created_at: str
    seq: int

    def encode(self) -> str:
        """Return a stable cursor token.

        Returns:
            Encoded cursor string ``created_at|seq``.
        """
        return f"{self.created_at}|{self.seq}"

    @classmethod
    def decode(cls, token: str) -> HistoryCursor:
        """Parse a cursor token.

        Args:
            token: Value previously returned as ``next_cursor`` or ``prev_cursor``.

        Returns:
            Parsed cursor.

        Raises:
            ValueError: When the token is malformed.
        """
        if "|" not in token:
            raise ValueError(f"Invalid history cursor: {token!r}")
        created_at, seq_str = token.rsplit("|", maxsplit=1)
        return cls(created_at=created_at, seq=int(seq_str))


@dataclass(frozen=True)
class HistoryPage:
    """One page of archived messages."""

    messages: list[dict]
    total_count: int
    order: str
    next_cursor: str | None
    prev_cursor: str | None


@dataclass(frozen=True)
class HistorySearchHit:
    """One BM25-ranked search hit."""

    seq: int
    created_at: str
    message_type: str
    snippet: str
    rank: float


class MessageLogStore:
    """Append-only per-thread message archive."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the store.

        Args:
            conn: SQLite connection.
        """
        self._conn = conn

    def append(self, thread_id: str, message: BaseMessage) -> int:
        """Append a message and return its sequence number.

        Args:
            thread_id: Conversation thread identifier.
            message: Message to persist.

        Returns:
            Monotonic sequence number for the thread.
        """
        seq = self.max_seq(thread_id) + 1
        now = utc_now_str()
        self._conn.execute(
            """
            INSERT INTO message_log (thread_id, seq, message_type, content_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, seq, message.type, serialize_message(message), now),
        )
        self._index_message(
            thread_id=thread_id,
            seq=seq,
            message_type=message.type,
            created_at=now,
            message=message,
        )
        return seq

    def append_many(self, thread_id: str, messages: list[BaseMessage]) -> list[int]:
        """Append multiple messages in order.

        Args:
            thread_id: Conversation thread identifier.
            messages: Messages to persist.

        Returns:
            Sequence numbers assigned to each message.
        """
        return [self.append(thread_id, message) for message in messages]

    def max_seq(self, thread_id: str) -> int:
        """Return the highest sequence number for a thread.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            Max sequence, or 0 when the thread has no messages.
        """
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM message_log WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return int(row[0])

    def list_range(
        self,
        thread_id: str,
        *,
        start_seq: int,
        end_seq: int | None = None,
    ) -> list[BaseMessage]:
        """Return messages in an inclusive sequence range.

        Args:
            thread_id: Conversation thread identifier.
            start_seq: First sequence to include.
            end_seq: Last sequence to include; when omitted, no upper bound.

        Returns:
            Messages ordered by sequence.
        """
        if end_seq is None:
            rows = self._conn.execute(
                """
                SELECT content_json FROM message_log
                WHERE thread_id = ? AND seq >= ?
                ORDER BY seq ASC
                """,
                (thread_id, start_seq),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT content_json FROM message_log
                WHERE thread_id = ? AND seq BETWEEN ? AND ?
                ORDER BY seq ASC
                """,
                (thread_id, start_seq, end_seq),
            ).fetchall()
        return [deserialize_message(row["content_json"]) for row in rows]

    def list_by_time(
        self,
        thread_id: str,
        *,
        start_at: str | None = None,
        end_at: str | None = None,
        limit: int = 50,
        order: str = "asc",
        cursor: HistoryCursor | None = None,
    ) -> HistoryPage:
        """Return one page of messages ordered by timestamp.

        Pagination uses ``(created_at, seq)`` so messages sharing a timestamp
        remain stable and no rows are skipped.

        Args:
            thread_id: Conversation thread identifier.
            start_at: Inclusive lower bound (UTC ``YYYY-MM-DD HH:MM:SS``).
            end_at: Exclusive upper bound (UTC).
            limit: Maximum rows to return (1–100).
            order: ``asc`` (oldest first) or ``desc`` (newest first).
            cursor: Continue from a prior ``next_cursor`` / ``prev_cursor``.

        Returns:
            Page of formatted message records plus pagination cursors.
        """
        normalized_order = order.lower()
        if normalized_order not in {"asc", "desc"}:
            raise ValueError(f"Unsupported order: {order!r}")

        clauses = ["thread_id = ?"]
        params: list[object] = [thread_id]

        if start_at is not None:
            clauses.append("created_at >= ?")
            params.append(_normalize_timestamp(start_at))
        if end_at is not None:
            clauses.append("created_at < ?")
            params.append(_normalize_timestamp(end_at))

        if cursor is not None:
            if normalized_order == "asc":
                clauses.append("(created_at > ? OR (created_at = ? AND seq > ?))")
            else:
                clauses.append("(created_at < ? OR (created_at = ? AND seq < ?))")
            params.extend([cursor.created_at, cursor.created_at, cursor.seq])

        order_sql = "created_at ASC, seq ASC" if normalized_order == "asc" else "created_at DESC, seq DESC"
        params.append(limit)

        rows = self._conn.execute(
            f"""
            SELECT seq, message_type, content_json, created_at
            FROM message_log
            WHERE {' AND '.join(clauses)}
            ORDER BY {order_sql}
            LIMIT ?
            """,
            params,
        ).fetchall()

        records = [
            message_to_history_row(
                seq=int(row["seq"]),
                message_type=row["message_type"],
                content_json=row["content_json"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

        total_count = self.count(thread_id)
        next_cursor = None
        prev_cursor = None
        if records:
            first = records[0]
            last = records[-1]
            first_cursor = HistoryCursor(created_at=first["created_at"], seq=first["seq"]).encode()
            last_cursor = HistoryCursor(created_at=last["created_at"], seq=last["seq"]).encode()
            if normalized_order == "asc":
                next_cursor = last_cursor
                prev_cursor = first_cursor
            else:
                next_cursor = last_cursor
                prev_cursor = first_cursor

        return HistoryPage(
            messages=records,
            total_count=total_count,
            order=normalized_order,
            next_cursor=next_cursor,
            prev_cursor=prev_cursor,
        )

    def search(
        self,
        thread_id: str,
        *,
        query: str,
        limit: int = 20,
    ) -> list[HistorySearchHit]:
        """Search archived messages with BM25 ranking (FTS5).

        Args:
            thread_id: Conversation thread identifier.
            query: Free-text query (FTS5 syntax).
            limit: Maximum hits to return.

        Returns:
            Ranked search hits with snippets.
        """
        fts_query = _sanitize_fts_query(query)
        if not fts_query:
            return []

        rows = self._conn.execute(
            """
            SELECT
                seq,
                created_at,
                message_type,
                snippet(message_log_fts, 0, '[', ']', '…', 48) AS snippet,
                bm25(message_log_fts) AS rank
            FROM message_log_fts
            WHERE message_log_fts MATCH ? AND thread_id = ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, thread_id, limit),
        ).fetchall()

        return [
            HistorySearchHit(
                seq=int(row["seq"]),
                created_at=row["created_at"],
                message_type=row["message_type"],
                snippet=row["snippet"],
                rank=float(row["rank"]),
            )
            for row in rows
        ]

    def count(self, thread_id: str) -> int:
        """Return the number of archived messages for a thread.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            Message count.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) FROM message_log WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return int(row[0])

    def _index_message(
        self,
        *,
        thread_id: str,
        seq: int,
        message_type: str,
        created_at: str,
        message: BaseMessage,
    ) -> None:
        """Insert one row into the FTS index.

        Args:
            thread_id: Conversation thread identifier.
            seq: Message sequence number.
            message_type: LangChain message type string.
            created_at: Stored UTC timestamp.
            message: Message being indexed.
        """
        search_text = message_search_text(message)
        self._conn.execute(
            """
            INSERT INTO message_log_fts (
                search_text, thread_id, seq, created_at, message_type
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (search_text, thread_id, seq, created_at, message_type),
        )


def backfill_message_fts(conn: sqlite3.Connection) -> None:
    """Populate the FTS index from existing ``message_log`` rows.

    Args:
        conn: Open SQLite connection.
    """
    existing = conn.execute(
        "SELECT COUNT(*) FROM message_log_fts",
    ).fetchone()[0]
    if existing:
        return

    rows = conn.execute(
        """
        SELECT thread_id, seq, message_type, content_json, created_at
        FROM message_log
        ORDER BY thread_id ASC, seq ASC
        """,
    ).fetchall()
    store = MessageLogStore(conn)
    for row in rows:
        message = deserialize_message(row["content_json"])
        store._index_message(
            thread_id=row["thread_id"],
            seq=int(row["seq"]),
            message_type=row["message_type"],
            created_at=row["created_at"],
            message=message,
        )


def message_to_history_row(
    *,
    seq: int,
    message_type: str,
    content_json: str,
    created_at: str,
) -> dict:
    """Deserialize and format one archived row for history tools.

    Args:
        seq: Message sequence number.
        message_type: Stored message type.
        content_json: Serialized LangChain message.
        created_at: Stored UTC timestamp.

    Returns:
        Agent-facing history record.
    """
    from context.message_text import message_to_record

    message = deserialize_message(content_json)
    return message_to_record(
        {
            "seq": seq,
            "message_type": message_type,
            "created_at": created_at,
            "message": message,
        },
    )


def _normalize_timestamp(value: str) -> str:
    """Normalize user timestamps to SQLite UTC format.

    Args:
        value: Timestamp string from tool input.

    Returns:
        Normalized ``YYYY-MM-DD HH:MM:SS`` UTC string.
    """
    return format_utc(parse_utc(value))


def _sanitize_fts_query(query: str) -> str:
    """Convert a user query into safe FTS5 prefix tokens.

    Args:
        query: Raw search string.

    Returns:
        FTS5 query string, or empty when no searchable tokens remain.
    """
    tokens = re.findall(r"[\w.-]+", query, flags=re.UNICODE)
    if not tokens:
        return ""
    return " ".join(f'"{token}"*' for token in tokens)
