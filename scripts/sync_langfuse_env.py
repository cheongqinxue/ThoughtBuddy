#!/usr/bin/env python3
"""Sync Langfuse credentials from the local Docker stack into project ``.env``."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANGFUSE_ENV = ROOT / ".data" / "langfuse-docker" / ".env"
DEFAULT_PROJECT_ENV = ROOT / ".env"
DEFAULT_BASE_URL = "http://localhost:3001"

_LANGFUSE_KEY_MAP = {
    "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_INIT_PROJECT_SECRET_KEY": "LANGFUSE_SECRET_KEY",
}
_MANAGED_KEYS = frozenset({*_LANGFUSE_KEY_MAP.values(), "LANGFUSE_BASE_URL"})
_KEY_PATTERN = re.compile(r"^([A-Z0-9_]+)=(.*)$")


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines from a dotenv file.

    Args:
        path: Dotenv file path.

    Returns:
        Mapping of variable names to unquoted values.
    """
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _split_env_lines(lines: list[str]) -> tuple[list[str], dict[str, int]]:
    """Locate managed env vars within existing dotenv lines.

    Args:
        lines: Dotenv lines without trailing file newline.

    Returns:
        Tuple of original lines and mapping of managed keys to line indexes.
    """
    indexes: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = _KEY_PATTERN.match(line.strip())
        if match is None:
            continue
        key = match.group(1)
        if key in _MANAGED_KEYS:
            indexes[key] = index
    return lines, indexes


def _ensure_project_env(project_env: Path) -> list[str]:
    """Return dotenv lines, seeding from template when the project file is missing.

    Args:
        project_env: ThoughtBuddy project ``.env`` path.

    Returns:
        Existing or seeded dotenv lines.
    """
    if project_env.exists():
        text = project_env.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            text += "\n"
        return text.splitlines()

    template = ROOT / ".env-template"
    if template.exists():
        text = template.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            text += "\n"
        return text.splitlines()
    return []


def sync_langfuse_env(
    *,
    langfuse_env: Path,
    project_env: Path,
    base_url: str,
) -> dict[str, str]:
    """Copy Langfuse project keys into the ThoughtBuddy ``.env`` file.

    Args:
        langfuse_env: Langfuse Docker stack ``.env`` path.
        project_env: ThoughtBuddy project ``.env`` path.
        base_url: Langfuse base URL written to ``LANGFUSE_BASE_URL``.

    Returns:
        Mapping of updated ThoughtBuddy env var names to values.

    Raises:
        FileNotFoundError: When the Langfuse env file is missing.
        ValueError: When required Langfuse init keys are absent.
    """
    if not langfuse_env.exists():
        msg = f"Missing Langfuse env file: {langfuse_env}"
        raise FileNotFoundError(msg)

    source_values = _read_env_file(langfuse_env)
    updates: dict[str, str] = {"LANGFUSE_BASE_URL": base_url}
    for source_key, target_key in _LANGFUSE_KEY_MAP.items():
        value = source_values.get(source_key, "")
        if not value:
            msg = f"{langfuse_env} is missing {source_key}"
            raise ValueError(msg)
        updates[target_key] = value

    lines = _ensure_project_env(project_env)
    lines, indexes = _split_env_lines(lines)

    for key, value in updates.items():
        rendered = f"{key}={value}"
        if key in indexes:
            lines[indexes[key]] = rendered
        else:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(rendered)

    project_env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return updates


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--langfuse-env",
        type=Path,
        default=DEFAULT_LANGFUSE_ENV,
        help="Path to Langfuse Docker .env",
    )
    parser.add_argument(
        "--project-env",
        type=Path,
        default=DEFAULT_PROJECT_ENV,
        help="Path to ThoughtBuddy .env",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Langfuse base URL to write into project .env",
    )
    args = parser.parse_args()

    updates = sync_langfuse_env(
        langfuse_env=args.langfuse_env,
        project_env=args.project_env,
        base_url=args.base_url,
    )
    print(f"Updated {args.project_env}:")
    for key in sorted(updates):
        print(f"  {key}=***")


if __name__ == "__main__":
    main()
