"""Chess experiment setup — environment checks and config sync for run.sh.

Single entry module at the chess experiment root. Registers nothing; the agent
plugin lives in ``experiments.chess.plugin``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import Settings
from experiments.runner.entrypoint import SetupError

if TYPE_CHECKING:
    from experiments.runner.config import ExperimentRunConfig


def _patch_scalar(text: str, *, key: str, value: Any, indent: int) -> str:
    """Replace one scalar YAML line under a fixed indent."""
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)
    spaces = " " * indent
    pattern = rf"(^{spaces}{re.escape(key)}:\s*)(.+)$"
    new_text, count = re.subn(
        pattern,
        rf"\g<1>{rendered}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        msg = f"config.yaml missing chess.{key!r} (expected {indent} spaces indent)"
        raise ValueError(msg)
    return new_text


def apply_settings(run_config: ExperimentRunConfig, text: str) -> str:
    """Patch ``experiment.chess.*`` fields from the experiment run config.

    Args:
        run_config: Loaded experiment run configuration.
        text: Current ``config.yaml`` contents.

    Returns:
        Updated config text.

    Raises:
        ValueError: When a required chess key is missing from config.yaml.
    """
    settings = run_config.experiment_settings
    defaults = {
        "engine_path": ".data/stockfish/engine",
        "agent_color": "white",
        "agent_generation": 0,
        "skill_level": 3,
        "move_time_seconds": 0.1,
        "engine_timeout_seconds": 10,
    }
    merged = {**defaults, **settings}
    for key, value in merged.items():
        text = _patch_scalar(text, key=key, value=value, indent=4)
    return text


def preflight(run_config: ExperimentRunConfig, *, repo_root: Path) -> None:
    """Verify Stockfish is installed before starting the chess stack.

    Args:
        run_config: Loaded experiment run configuration.
        repo_root: Repository root.

    Raises:
        SetupError: When Stockfish is missing or not executable.
    """
    settings = run_config.experiment_settings
    engine = Path(str(settings.get("engine_path", ".data/stockfish/engine")))
    engine_path = engine if engine.is_absolute() else repo_root / engine
    if not engine_path.is_file() or not engine_path.stat().st_mode & 0o111:
        msg = (
            f"Stockfish not installed or not executable: {engine_path}\n"
            "Run: ./scripts/install_stockfish"
        )
        raise SetupError(msg)


def analyze(run_config: ExperimentRunConfig, *, repo_root: Path) -> None:
    """Print chess session, queue, and observability summary.

    Args:
        run_config: Loaded experiment run configuration.
        repo_root: Repository root.
    """
    settings = Settings.load()
    db = Path(settings.db_path)
    if not db.is_file():
        msg = f"database not found at {db}"
        raise FileNotFoundError(msg)

    paths = run_config.resolved_paths(repo_root)
    print("=== Chess experiment summary ===")
    print(f"Experiment config: {run_config.config_path}")
    print(f"LLM profile:       {run_config.llm_profile} ({settings.agent_model})")
    print(f"Database:          {db}")
    print(f"Logs:              {paths['log_dir']}/")
    print()

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        _print_sessions(conn)
        _print_task_queue(conn)
        _print_chess_tasks(conn)
        _print_observability(settings, run_config)
    finally:
        conn.close()


def start_hints(run_config: ExperimentRunConfig, *, tb_port: int) -> list[str]:
    """Return post-start banner lines for run.sh.

    Args:
        run_config: Loaded experiment run configuration.
        tb_port: ThoughtBuddy HTTP port.

    Returns:
        Human-readable hint lines.
    """
    return [
        f"  Chess UI:  http://127.0.0.1:{tb_port}/thoughtbuddy/chess/ui",
        "  The heartbeat will enqueue chess_start / chess_continue automatically.",
    ]


def _print_sessions(conn: sqlite3.Connection) -> None:
    print("--- Agent chess sessions (latest 5) ---")
    rows = conn.execute(
        """
        SELECT id, status, result, agent_generation, player_color, created_at, updated_at
        FROM chess_sessions
        WHERE opponent_type = 'agent'
        ORDER BY created_at DESC
        LIMIT 5
        """
    ).fetchall()
    if not rows:
        print("(no agent sessions yet)")
        return
    for row in rows:
        moves = conn.execute(
            "SELECT COUNT(*) AS n FROM chess_moves WHERE session_id = ?",
            (row["id"],),
        ).fetchone()["n"]
        print(
            f"  {row['id'][:8]}… gen={row['agent_generation']} "
            f"{row['player_color']} status={row['status']} result={row['result'] or '-'} "
            f"moves={moves} updated={row['updated_at']}"
        )


def _print_task_queue(conn: sqlite3.Connection) -> None:
    print("\n--- Task queue ---")
    for status, count in conn.execute(
        "SELECT status, COUNT(*) FROM tasks GROUP BY status ORDER BY status"
    ):
        print(f"  {status}: {count}")


def _print_chess_tasks(conn: sqlite3.Connection) -> None:
    print("\n--- Recent chess-related tasks ---")
    task_rows = conn.execute(
        """
        SELECT id, status, source, substr(message, 1, 80) AS message, payload, created_at
        FROM tasks
        WHERE payload LIKE '%chess%'
        ORDER BY created_at DESC
        LIMIT 8
        """
    ).fetchall()
    if not task_rows:
        print("(none)")
        return
    for row in task_rows:
        payload = json.loads(row["payload"]) if row["payload"] else {}
        kind = payload.get("kind", "?")
        print(f"  [{row['status']}] {kind} — {row['message']} ({row['created_at']})")


def _print_observability(settings: Settings, run_config: ExperimentRunConfig) -> None:
    print("\n--- Wake / observability ---")
    log_path = run_config.resolved_paths(Path.cwd())["thoughtbuddy_log"]
    print(f"  Structured logs (wake_id): {log_path}")
    if settings.langfuse_enabled:
        print(f"  Langfuse UI: {settings.langfuse_base_url} (wake traces + tool spans)")
    else:
        print("  Langfuse: disabled in config.yaml")
    print("  Message archive: SQLite message_log + history_search tool")
    port = run_config.stack.thoughtbuddy_port
    print(f"  Live board replay: http://127.0.0.1:{port}/thoughtbuddy/chess/ui")
