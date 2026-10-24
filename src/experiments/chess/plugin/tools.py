"""Agent tools for interacting with the chess experiment."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config import Settings
from db.connection import connect
from experiments.chess.plugin.service import ChessGameService


class ChessMoveInput(BaseModel):
    """Input schema for ``world_chess_move``."""

    uci: str = Field(
        description="Move in UCI notation (e.g. e2e4, e7e8q for promotion).",
    )


def create_tools(settings: Settings) -> list[StructuredTool]:
    """Create chess virtual-world tools.

    Args:
        settings: Application settings.

    Returns:
        LangChain tools for chess board interaction.
    """
    db_path = Path(settings.thoughtbuddy_db_path)

    def _service(conn) -> ChessGameService:
        return ChessGameService(conn, settings)

    def world_chess_status() -> str:
        """Return current board state for the active agent session."""
        conn = connect(db_path)
        try:
            payload = _service(conn).get_agent_status()
            conn.commit()
            return json.dumps(payload, indent=2)
        finally:
            conn.close()

    def world_chess_move(uci: str) -> str:
        """Apply a legal agent move and let Stockfish reply when appropriate."""
        conn = connect(db_path)
        try:
            service = _service(conn)
            session = service.create_agent_session(force_new=False)
            payload = service.apply_agent_move(session["id"], uci)
            conn.commit()
            return json.dumps(payload, indent=2)
        finally:
            conn.close()

    def world_chess_reset() -> str:
        """Start a new agent vs Stockfish game for the current generation."""
        conn = connect(db_path)
        try:
            payload = _service(conn).reset_agent_session()
            conn.commit()
            return json.dumps({"ok": True, "session": payload}, indent=2)
        finally:
            conn.close()

    generation = settings.chess_agent_generation
    return [
        StructuredTool.from_function(
            func=world_chess_status,
            name="world_chess_status",
            description=(
                "Read the active agent chess session: FEN, side to move, check/game-over "
                "flags, legal UCI moves, session id, and agent generation."
            ),
        ),
        StructuredTool.from_function(
            func=world_chess_move,
            name="world_chess_move",
            description=(
                "Play your move in the active agent chess session (agent plays "
                f"{settings.chess_agent_color}, generation {generation}). Pass UCI "
                "notation (e.g. e2e4). Stockfish replies automatically when legal."
            ),
            args_schema=ChessMoveInput,
        ),
        StructuredTool.from_function(
            func=world_chess_reset,
            name="world_chess_reset",
            description=(
                "Abandon the current agent chess session and start a fresh game against "
                f"Stockfish for generation {generation}."
            ),
        ),
    ]
