"""Single-instance PID lock file."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import TextIO


class InstanceLockError(RuntimeError):
    """Raised when another ThoughtBuddy instance holds the lock."""


class InstanceLock:
    """Exclusive process lock backed by a lock file."""

    def __init__(self, path: Path) -> None:
        """Initialize the lock.

        Args:
            path: Filesystem path for the lock file.
        """
        self._path = path
        self._handle: TextIO | None = None

    @property
    def path(self) -> Path:
        """Return the lock file path."""
        return self._path

    def acquire(self) -> None:
        """Acquire the exclusive lock or raise if another instance holds it."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("w+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            msg = f"Another ThoughtBuddy instance holds {self._path}"
            raise InstanceLockError(msg) from exc

        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        """Release the lock and close the lock file."""
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> InstanceLock:
        """Acquire the lock when entering a context manager."""
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        """Release the lock when exiting a context manager."""
        self.release()
