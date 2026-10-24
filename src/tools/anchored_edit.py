"""Shared LangChain tool factory for anchored file edits."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from utils.anchored_edit import BOF_MARKER, EOF_MARKER, AnchoredEditError, edit_file


class AnchoredEditInput(BaseModel):
    """Input schema for anchored file edit tools."""

    pre: str = Field(
        description=(
            "The single line immediately before the region to change. "
            f"Use {BOF_MARKER!r} when editing at the start of the file."
        ),
    )
    post: str = Field(
        description=(
            "The single line immediately after the region to change. "
            f"Use {EOF_MARKER!r} when editing at the end of the file."
        ),
    )
    original: str = Field(
        description=(
            "Exact text to replace (may span multiple lines). "
            "Use an empty string to insert at the anchor point."
        ),
    )
    new: str = Field(
        description="Replacement text. Use an empty string to delete original.",
    )


def create_edit_tool(
    *,
    path: Path,
    name: str,
    description: str,
    success_message: str,
) -> StructuredTool:
    """Build a LangChain tool that edits one file with anchored deltas.

    Args:
        path: Absolute or relative path to the target markdown file.
        name: Tool name exposed to the agent.
        description: Tool description for the agent.
        success_message: Message returned after a successful edit.

    Returns:
        Configured structured edit tool.
    """
    target_path = path

    def edit_file_tool(pre: str, post: str, original: str, new: str) -> str:
        try:
            edit_file(target_path, pre=pre, post=post, original=original, new=new)
        except AnchoredEditError as exc:
            return str(exc)
        return success_message

    return StructuredTool.from_function(
        func=edit_file_tool,
        name=name,
        description=description,
        args_schema=AnchoredEditInput,
    )
