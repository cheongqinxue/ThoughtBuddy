"""Anchored delta edits for markdown files on disk."""

from __future__ import annotations

from pathlib import Path

BOF_MARKER = "<BOF>"
EOF_MARKER = "<EOF>"


class AnchoredEditError(Exception):
    """Raised when an anchored file edit fails."""


def read_file(path: str | Path) -> str:
    """Read a file exactly as stored on disk.

    Args:
        path: Path to the target file.

    Returns:
        File contents, or an empty string when the file is missing.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    return file_path.read_text(encoding="utf-8")


def edit_file(
    path: str | Path,
    *,
    pre: str,
    post: str,
    original: str,
    new: str,
) -> str:
    """Apply an anchored delta edit and persist the result.

    ``pre`` and ``post`` are the single lines immediately before and after
    ``original``. Use ``BOF_MARKER`` (``<BOF>``) when anchoring at the start of
    the file and ``EOF_MARKER`` (``<EOF>``) at the end. Pass an empty string
    for ``new`` to delete ``original``.

    Args:
        path: Path to the file to edit.
        pre: Line before ``original``, or ``<BOF>`` at file start.
        post: Line after ``original``, or ``<EOF>`` at file end.
        original: Exact text to replace (may span multiple lines).
        new: Replacement text (empty string deletes ``original``).

    Returns:
        Updated file contents after the edit.

    Raises:
        AnchoredEditError: When anchors do not match or the edit is ambiguous.
    """
    content = read_file(path)
    updated = apply_edit(content, pre=pre, post=post, original=original, new=new)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(updated, encoding="utf-8")
    return updated


def apply_edit(
    content: str,
    *,
    pre: str,
    post: str,
    original: str,
    new: str,
) -> str:
    """Apply an anchored delta edit to in-memory file content.

    Args:
        content: Current file text.
        pre: Line before ``original``, or ``<BOF>`` at file start.
        post: Line after ``original``, or ``<EOF>`` at file end.
        original: Exact text to replace (may span multiple lines).
        new: Replacement text (empty string deletes ``original``).

    Returns:
        Updated file contents.

    Raises:
        AnchoredEditError: When anchors do not match or the edit is ambiguous.
    """
    lines, ends_with_newline = _parse_lines(content)
    original_lines, _ = _parse_lines(original)
    new_lines, _ = _parse_lines(new)
    match_index = _find_unique_match(lines, pre=pre, post=post, original_lines=original_lines)
    updated_lines = lines[:match_index] + new_lines + lines[match_index + len(original_lines) :]
    return _format_lines(updated_lines, ends_with_newline)


def _parse_lines(text: str) -> tuple[list[str], bool]:
    if text == "":
        return [], False
    ends_with_newline = text.endswith("\n")
    body = text[:-1] if ends_with_newline else text
    if body == "":
        return [], ends_with_newline
    return body.split("\n"), ends_with_newline


def _format_lines(lines: list[str], ends_with_newline: bool) -> str:
    if not lines:
        return "\n" if ends_with_newline else ""
    text = "\n".join(lines)
    if ends_with_newline:
        text += "\n"
    return text


def _find_unique_match(
    lines: list[str],
    *,
    pre: str,
    post: str,
    original_lines: list[str],
) -> int:
    matches = [
        index
        for index in range(len(lines) - len(original_lines) + 1)
        if _matches_at(lines, index, pre=pre, post=post, original_lines=original_lines)
    ]
    if not matches:
        msg = "File edit failed: no region matched the given pre, original, and post."
        raise AnchoredEditError(msg)
    if len(matches) > 1:
        msg = "File edit failed: multiple regions matched; refine anchors."
        raise AnchoredEditError(msg)
    return matches[0]


def _matches_at(
    lines: list[str],
    index: int,
    *,
    pre: str,
    post: str,
    original_lines: list[str],
) -> bool:
    if lines[index : index + len(original_lines)] != original_lines:
        return False
    if not _line_matches_pre(lines, index, pre):
        return False
    return _line_matches_post(lines, index + len(original_lines), post)


def _line_matches_pre(lines: list[str], index: int, pre: str) -> bool:
    if pre == BOF_MARKER:
        return index == 0
    if index == 0:
        return False
    return lines[index - 1] == pre


def _line_matches_post(lines: list[str], index: int, post: str) -> bool:
    if post == EOF_MARKER:
        return index == len(lines)
    if index >= len(lines):
        return False
    return lines[index] == post
