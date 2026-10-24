"""SQLite persistence for ThoughtBuddy."""

from db.connection import connect, ensure_data_dir
from db.schema import init_db

__all__ = ["connect", "ensure_data_dir", "init_db"]
