"""SQLite schema definitions and initialization."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 8

_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    schedule_kind TEXT NOT NULL CHECK (schedule_kind IN ('cron', 'once')),
    cron_expr TEXT,
    run_at TEXT,
    next_fire_at TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    alert_id TEXT REFERENCES alerts(id),
    occurrence_at TEXT,
    event_key TEXT,
    source TEXT NOT NULL CHECK (source IN ('alert', 'user', 'event')),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'ignored')),
    payload TEXT NOT NULL DEFAULT '{}',
    disposition_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    UNIQUE (alert_id, occurrence_at),
    UNIQUE (event_key, occurrence_at)
);

CREATE INDEX IF NOT EXISTS idx_tasks_status_created ON tasks (status, created_at);

CREATE TABLE IF NOT EXISTS monitored_processes (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    command_json TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK (
        status IN (
            'running', 'completed', 'failed', 'timed_out',
            'interrupted', 'cancelled'
        )
    ),
    pid INTEGER,
    pgid INTEGER,
    log_dir TEXT NOT NULL,
    timeout_seconds INTEGER NOT NULL,
    exit_code INTEGER,
    signal_name TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_monitored_processes_thread_status
    ON monitored_processes (thread_id, status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_next_fire ON alerts (enabled, next_fire_at);

CREATE TABLE IF NOT EXISTS message_log (
    thread_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    message_type TEXT NOT NULL,
    content_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (thread_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_message_log_thread_created
    ON message_log (thread_id, created_at);

CREATE INDEX IF NOT EXISTS idx_message_log_thread_created_seq
    ON message_log (thread_id, created_at, seq);

CREATE VIRTUAL TABLE IF NOT EXISTS message_log_fts USING fts5(
    search_text,
    thread_id UNINDEXED,
    seq UNINDEXED,
    created_at UNINDEXED,
    message_type UNINDEXED,
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS compaction_points (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    boundary_seq INTEGER NOT NULL,
    retain_count INTEGER NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_compaction_thread
    ON compaction_points (thread_id, created_at);

CREATE TABLE IF NOT EXISTS world_observations (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    summary TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_world_observations_thread_created
    ON world_observations (thread_id, created_at);

CREATE TABLE IF NOT EXISTS world_state (
    thread_id TEXT NOT NULL,
    state_key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (thread_id, state_key)
);

CREATE TABLE IF NOT EXISTS plan_nodes (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'active', 'blocked', 'done', 'abandoned')
    ),
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_plan_nodes_thread_status
    ON plan_nodes (thread_id, status);

CREATE TABLE IF NOT EXISTS plan_edges (
    thread_id TEXT NOT NULL,
    from_node_id TEXT NOT NULL REFERENCES plan_nodes(id),
    to_node_id TEXT NOT NULL REFERENCES plan_nodes(id),
    PRIMARY KEY (thread_id, from_node_id, to_node_id)
);

CREATE TABLE IF NOT EXISTS chess_sessions (
    id TEXT PRIMARY KEY,
    opponent_type TEXT NOT NULL CHECK (opponent_type IN ('human', 'agent')),
    agent_generation INTEGER,
    player_color TEXT NOT NULL CHECK (player_color IN ('white', 'black')),
    stockfish_color TEXT NOT NULL CHECK (stockfish_color IN ('white', 'black')),
    status TEXT NOT NULL CHECK (status IN ('active', 'completed', 'abandoned')),
    result TEXT,
    starting_fen TEXT NOT NULL,
    current_fen TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_chess_sessions_created
    ON chess_sessions (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_chess_sessions_agent_active
    ON chess_sessions (opponent_type, agent_generation, status);

CREATE TABLE IF NOT EXISTS chess_moves (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chess_sessions(id),
    ply INTEGER NOT NULL,
    uci TEXT NOT NULL,
    san TEXT,
    player_role TEXT NOT NULL CHECK (player_role IN ('human', 'agent', 'stockfish')),
    agent_generation INTEGER,
    fen_after TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, ply)
);

CREATE INDEX IF NOT EXISTS idx_chess_moves_session_ply
    ON chess_moves (session_id, ply);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Apply schema migrations to the database connection.

    Args:
        conn: Open SQLite connection.
    """
    conn.executescript(_DDL)
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    current = row[0] if row and row[0] is not None else 0
    if current < SCHEMA_VERSION:
        if current < 2:
            conn.execute(
                "UPDATE tasks SET status = 'ignored' WHERE status = 'deferred'",
            )
        if current < 5:
            conn.execute("DROP TABLE IF EXISTS asset_snapshots")
            conn.execute("DROP TABLE IF EXISTS asset_events")
        if current < 6:
            from store.messages import backfill_message_fts

            backfill_message_fts(conn)
        if current < 8:
            _migrate_to_v8(conn)
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, datetime('now'))",
            (SCHEMA_VERSION,),
        )
    conn.commit()


def _migrate_to_v8(conn: sqlite3.Connection) -> None:
    """Add event task source and monitored subprocess table."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tasks_v8 (
            id TEXT PRIMARY KEY,
            alert_id TEXT REFERENCES alerts(id),
            occurrence_at TEXT,
            event_key TEXT,
            source TEXT NOT NULL CHECK (source IN ('alert', 'user', 'event')),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'completed', 'ignored')),
            payload TEXT NOT NULL DEFAULT '{}',
            disposition_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            UNIQUE (alert_id, occurrence_at),
            UNIQUE (event_key, occurrence_at)
        );

        INSERT INTO tasks_v8 (
            id, alert_id, occurrence_at, event_key, source, status, payload,
            disposition_reason, created_at, updated_at, started_at, completed_at
        )
        SELECT
            id, alert_id, occurrence_at, NULL, source, status, payload,
            disposition_reason, created_at, updated_at, started_at, completed_at
        FROM tasks;

        DROP TABLE tasks;
        ALTER TABLE tasks_v8 RENAME TO tasks;

        CREATE INDEX IF NOT EXISTS idx_tasks_status_created ON tasks (status, created_at);

        CREATE TABLE IF NOT EXISTS monitored_processes (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            command_json TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL CHECK (
                status IN (
                    'running', 'completed', 'failed', 'timed_out',
                    'interrupted', 'cancelled'
                )
            ),
            pid INTEGER,
            pgid INTEGER,
            log_dir TEXT NOT NULL,
            timeout_seconds INTEGER NOT NULL,
            exit_code INTEGER,
            signal_name TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_monitored_processes_thread_status
            ON monitored_processes (thread_id, status, started_at DESC);
        """,
    )
