"""Stockfish UCI engine integration for the chess experiment."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import chess
import chess.engine

from config import Settings

logger = logging.getLogger("thoughtbuddy.experiments.chess.plugin.engine")

_BUNDLED_ENGINE_PATH = Path(".data/stockfish/engine")
_CANDIDATE_PATHS = (
    "/usr/games/stockfish",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
)


class ChessEngineError(RuntimeError):
    """Raised when Stockfish is missing or fails to start."""


def resolve_engine_path(engine_path: str) -> Path:
    """Resolve the Stockfish executable path.

    Args:
        engine_path: Configured engine command or absolute path.

    Returns:
        Executable path.

    Raises:
        ChessEngineError: When Stockfish cannot be found or executed.
    """
    configured = engine_path.strip()
    if not configured:
        msg = "Chess engine path is empty. Set experiment.chess.engine_path in config.yaml."
        raise ChessEngineError(msg)

    candidates: list[Path] = []
    if os.path.isabs(configured):
        candidates.append(Path(configured))
    elif os.sep in configured or configured.startswith("."):
        candidates.append(Path.cwd() / configured)
    else:
        found = shutil.which(configured)
        if found:
            candidates.append(Path(found))
        candidates.extend(Path(path) for path in _CANDIDATE_PATHS)
        candidates.append(Path.cwd() / _BUNDLED_ENGINE_PATH)

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    checked_text = ", ".join(checked) if checked else configured
    msg = (
        f"Stockfish not found (looked for {checked_text!r}). "
        "Install it with ./scripts/install_stockfish"
    )
    raise ChessEngineError(msg)


def validate_stockfish(settings: Settings) -> Path:
    """Verify Stockfish starts and responds to UCI.

    Args:
        settings: Application settings.

    Returns:
        Resolved Stockfish executable path.

    Raises:
        ChessEngineError: When Stockfish is missing or fails to start.
    """
    engine_path = resolve_engine_path(settings.chess_engine_path)
    timeout = settings.chess_engine_timeout_seconds
    try:
        with chess.engine.SimpleEngine.popen_uci(str(engine_path), timeout=timeout) as engine:
            engine.ping()
    except chess.engine.EngineError as exc:
        msg = f"Stockfish failed to start at {engine_path}: {exc}"
        raise ChessEngineError(msg) from exc
    except OSError as exc:
        msg = f"Stockfish failed to start at {engine_path}: {exc}"
        raise ChessEngineError(msg) from exc

    logger.info("stockfish_ready path=%s", engine_path)
    return engine_path


@contextmanager
def open_stockfish(settings: Settings) -> Iterator[chess.engine.SimpleEngine]:
    """Open a short-lived Stockfish UCI engine session.

    Args:
        settings: Application settings.

    Yields:
        Configured ``SimpleEngine`` instance.

    Raises:
        ChessEngineError: When Stockfish cannot be opened.
    """
    engine_path = resolve_engine_path(settings.chess_engine_path)
    timeout = settings.chess_engine_timeout_seconds
    engine: chess.engine.SimpleEngine | None = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(engine_path), timeout=timeout)
        _configure_engine(engine, settings)
        yield engine
    except chess.engine.EngineError as exc:
        msg = f"Stockfish failed during UCI session at {engine_path}: {exc}"
        raise ChessEngineError(msg) from exc
    finally:
        if engine is not None:
            with contextlib.suppress(chess.engine.EngineError):
                engine.quit()


def choose_opponent_move(board: chess.Board, settings: Settings) -> chess.Move:
    """Ask Stockfish for the best move in the current position.

    Args:
        board: Current chess position.
        settings: Application settings.

    Returns:
        Engine-selected legal move.

    Raises:
        ChessEngineError: When Stockfish fails to return a move.
    """
    with open_stockfish(settings) as engine:
        limit = chess.engine.Limit(time=settings.chess_move_time_seconds)
        result = engine.play(board, limit)
        if result.move is None:
            msg = "Stockfish returned no move"
            raise ChessEngineError(msg)
        return result.move


def _configure_engine(engine: chess.engine.SimpleEngine, settings: Settings) -> None:
    """Apply strength and skill options to Stockfish.

    Args:
        engine: Open UCI engine session.
        settings: Application settings.
    """
    skill_level = max(0, min(settings.chess_skill_level, 20))
    engine.configure(
        {
            "UCI_LimitStrength": True,
            "Skill Level": skill_level,
        },
    )
