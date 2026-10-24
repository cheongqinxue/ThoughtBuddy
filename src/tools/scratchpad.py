"""Agent tool for editing the scratchpad."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool

from config import Settings
from runtime.scratchpad_layout import SCRATCHPAD_FILENAME
from tools.anchored_edit import create_edit_tool
from utils.anchored_edit import BOF_MARKER, EOF_MARKER


def create_tools(settings: Settings) -> list[StructuredTool]:
    """Create scratchpad tools from application settings.

    Args:
        settings: Application settings.

    Returns:
        LangChain tools for scratchpad access.
    """
    path = Path(settings.scratchpad_directory) / SCRATCHPAD_FILENAME
    return [
        create_edit_tool(
            path=path,
            name="edit_scratchpad",
            description=(
                "Edit SCRATCHPAD.md — front-of-mind notes persisted on disk. "
                "Use anchored deltas: pre (line before), post (line after), "
                "original (exact text to replace), and new (replacement). "
                f"Use {BOF_MARKER} for file start and {EOF_MARKER} for file end. "
                "Use an empty original to insert; an empty new to delete."
            ),
            success_message="Scratchpad updated.",
        ),
    ]


from tools.registry import register_tools

register_tools(create_tools)
