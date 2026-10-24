"""HTTP routes for chess sessions and the browser UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from experiments.chess.plugin.service import ChessGameError, ChessGameService

logger = logging.getLogger("thoughtbuddy.experiments.chess.plugin.routes")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_UI_PATH = _STATIC_DIR / "index.html"
_PIECES_DIR = _STATIC_DIR / "img" / "chesspieces" / "wikipedia"
_VALID_PIECES = frozenset(
    {"wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"},
)


def _runtime(request: Request):
    """Return the ThoughtBuddy runtime or raise via JSON 503."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return None
    return runtime


def _service(request: Request) -> ChessGameService | None:
    """Build a chess service from the active runtime."""
    runtime = _runtime(request)
    if runtime is None:
        return None
    return ChessGameService(runtime.conn, runtime.settings)


def _error(message: str, *, status_code: int = 400) -> JSONResponse:
    """Return a JSON error payload."""
    return JSONResponse({"ok": False, "error": message}, status_code=status_code)


async def chess_ui(_: Request) -> FileResponse:
    """Serve the chess browser UI."""
    return FileResponse(_UI_PATH)


async def chess_piece(request: Request) -> FileResponse | JSONResponse:
    """Serve a chess piece image for the browser UI."""
    piece = request.path_params["piece"]
    if piece not in _VALID_PIECES:
        return _error("invalid piece", status_code=404)
    return FileResponse(_PIECES_DIR / f"{piece}.png")


async def list_sessions(request: Request) -> JSONResponse:
    """List stored chess sessions."""
    service = _service(request)
    if service is None:
        return _error("runtime not ready", status_code=503)

    opponent_type = request.query_params.get("opponent_type")
    if opponent_type not in {None, "human", "agent"}:
        return _error("opponent_type must be human or agent")

    sessions = service.list_sessions(
        opponent_type=opponent_type,  # type: ignore[arg-type]
        limit=int(request.query_params.get("limit", "50")),
    )
    return JSONResponse({"ok": True, "sessions": sessions})


async def get_active_agent_session(request: Request) -> JSONResponse:
    """Return the live agent session for the configured generation."""
    service = _service(request)
    if service is None:
        return _error("runtime not ready", status_code=503)

    session = service.get_active_agent_session()
    if session is None:
        return JSONResponse({"ok": True, "session": None})
    return JSONResponse({"ok": True, "session": session})


async def get_session(request: Request) -> JSONResponse:
    """Return one session with move history."""
    service = _service(request)
    if service is None:
        return _error("runtime not ready", status_code=503)

    session_id = request.path_params["session_id"]
    try:
        session = service.get_session_detail(session_id)
    except KeyError:
        return _error("session not found", status_code=404)
    return JSONResponse({"ok": True, "session": session})


async def create_session(request: Request) -> JSONResponse:
    """Create a new human vs Stockfish session."""
    service = _service(request)
    if service is None:
        return _error("runtime not ready", status_code=503)

    body = await request.json()
    player_color = body.get("player_color", "white")
    if player_color not in {"white", "black"}:
        return _error("player_color must be white or black")

    session = service.create_human_session(player_color=player_color)  # type: ignore[arg-type]
    runtime = _runtime(request)
    assert runtime is not None
    runtime.conn.commit()
    return JSONResponse({"ok": True, "session": session}, status_code=201)


async def post_move(request: Request) -> JSONResponse:
    """Apply a human move in a human vs Stockfish session."""
    service = _service(request)
    if service is None:
        return _error("runtime not ready", status_code=503)

    session_id = request.path_params["session_id"]
    body = await request.json()
    uci = body.get("uci")
    if not isinstance(uci, str) or not uci.strip():
        return _error("uci is required")

    try:
        session = service.apply_human_move(session_id, uci.strip())
    except KeyError:
        return _error("session not found", status_code=404)
    except ChessGameError as exc:
        return _error(str(exc))
    except Exception:
        logger.exception("chess_human_move_failed session_id=%s", session_id)
        return _error("failed to apply move", status_code=500)

    runtime = _runtime(request)
    assert runtime is not None
    runtime.conn.commit()
    return JSONResponse({"ok": True, "session": session})


CHESS_ROUTES = [
    ("/thoughtbuddy/chess/ui", chess_ui, ["GET"]),
    ("/thoughtbuddy/chess/pieces/{piece}.png", chess_piece, ["GET"]),
    ("/thoughtbuddy/chess/sessions", list_sessions, ["GET"]),
    ("/thoughtbuddy/chess/sessions/active/agent", get_active_agent_session, ["GET"]),
    ("/thoughtbuddy/chess/sessions", create_session, ["POST"]),
    ("/thoughtbuddy/chess/sessions/{session_id}", get_session, ["GET"]),
    ("/thoughtbuddy/chess/sessions/{session_id}/move", post_move, ["POST"]),
]
