#!/usr/bin/env python3
"""Normalize fenced mermaid blocks for stable rendering in Cursor and GitHub.

- Dedent block bodies to column 0 (no mixed indent inside fences).
- Quote node and edge labels that contain characters which break preview parsers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".data", ".venv", "node_modules", ".git", "wandb"}
SPECIAL_LABEL_CHARS = frozenset("/:*")

NODE_LABEL = re.compile(r"(\b[A-Za-z_][A-Za-z0-9_]*)\[([^\]\"']+)\]")
EDGE_LABEL = re.compile(r"(\|)([^|\"']+)(\|)")
PARTICIPANT_ALIAS = re.compile(
    r"^(participant\s+\S+\s+as\s+)([^\"'\n]+)$",
    re.MULTILINE,
)


def _needs_quotes(label: str) -> bool:
    """Return whether a mermaid label should be wrapped in double quotes."""
    stripped = label.strip()
    if not stripped or stripped.startswith('"'):
        return False
    return any(char in stripped for char in SPECIAL_LABEL_CHARS)


def _quote_label(label: str) -> str:
    """Wrap a label in mermaid double quotes."""
    return f'"{label.strip()}"'


def normalize_line(line: str) -> str:
    """Normalize one mermaid source line."""
    def node_repl(match: re.Match[str]) -> str:
        node_id, label = match.group(1), match.group(2)
        if _needs_quotes(label):
            return f"{node_id}[{_quote_label(label)}]"
        return match.group(0)

    line = NODE_LABEL.sub(node_repl, line)

    def edge_repl(match: re.Match[str]) -> str:
        label = match.group(2)
        if _needs_quotes(label):
            return f'|{_quote_label(label)}|'
        return match.group(0)

    if "-->" in line or "---" in line:
        line = EDGE_LABEL.sub(edge_repl, line)

    alias_match = PARTICIPANT_ALIAS.match(line.strip())
    if alias_match and _needs_quotes(alias_match.group(2)):
        prefix, alias = alias_match.group(1), alias_match.group(2).strip()
        line = f"{prefix}{_quote_label(alias)}"

    return line


def normalize_block(body: str) -> str:
    """Dedent and normalize a mermaid diagram body."""
    lines = body.splitlines()
    normalized = [normalize_line(line.lstrip()) for line in lines]
    while normalized and not normalized[0].strip():
        normalized.pop(0)
    while normalized and not normalized[-1].strip():
        normalized.pop()
    return "\n".join(normalized)


def normalize_markdown(text: str) -> tuple[str, int]:
    """Return updated markdown and number of blocks changed."""
    pattern = re.compile(r"```mermaid\n(.*?)\n```", re.DOTALL)
    changes = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal changes
        original = match.group(1)
        updated = normalize_block(original)
        if updated != original.strip("\n"):
            changes += 1
        return f"```mermaid\n{updated}\n```"

    return pattern.sub(repl, text), changes


def iter_markdown_files() -> list[Path]:
    """Yield project markdown files that may contain mermaid."""
    files: list[Path] = []
    for path in ROOT.rglob("*.md"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def main() -> int:
    """Normalize all mermaid blocks in project markdown files."""
    check_only = "--check" in sys.argv
    total_blocks = 0
    dirty_files: list[Path] = []

    for path in iter_markdown_files():
        original = path.read_text(encoding="utf-8")
        updated, block_changes = normalize_markdown(original)
        if block_changes:
            total_blocks += block_changes
            dirty_files.append(path)
            if not check_only:
                path.write_text(updated, encoding="utf-8")

    if check_only:
        if dirty_files:
            print("Mermaid blocks need normalization:")
            for path in dirty_files:
                print(f"  {path.relative_to(ROOT)}")
            return 1
        print("All mermaid blocks normalized.")
        return 0

    if dirty_files:
        print(f"Normalized {total_blocks} mermaid block(s) in {len(dirty_files)} file(s):")
        for path in dirty_files:
            print(f"  {path.relative_to(ROOT)}")
    else:
        print("No mermaid blocks changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
