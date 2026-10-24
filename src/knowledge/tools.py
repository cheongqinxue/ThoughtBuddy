"""Knowledge tools for the agent."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config import Settings
from knowledge.chroma import build_chroma_backend


class KnowledgeAddInput(BaseModel):
    """Input schema for ``knowledge_add``."""

    text: str = Field(description="Knowledge body text to store.")
    tags: list[str] = Field(default_factory=list, description="Optional tags.")
    source: str = Field(default="", description="Optional source note (url, observation, etc.).")


class KnowledgeEditInput(BaseModel):
    """Input schema for ``knowledge_edit``."""

    entry_id: str = Field(description="Knowledge entry id to edit or supersede.")
    text: str = Field(description="Replacement or updated knowledge text.")
    supersede: bool = Field(
        default=True,
        description="When true, mark the old entry superseded and create a new active entry.",
    )
    tags: list[str] = Field(default_factory=list, description="Optional tags for new entry.")


class KnowledgeSearchInput(BaseModel):
    """Input schema for ``knowledge_search``."""

    query: str = Field(description="Natural-language search query.")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum results.")


def create_tools(settings: Settings) -> list[StructuredTool]:
    """Create knowledge tools backed by the configured backend.

    Args:
        settings: Application settings.

    Returns:
        LangChain knowledge tools.
    """
    backend = build_chroma_backend(settings)

    def knowledge_add(text: str, tags: list[str] | None = None, source: str = "") -> str:
        """Add knowledge entry."""
        metadata: dict[str, Any] = {}
        if tags:
            metadata["tags"] = tags
        if source:
            metadata["source"] = source
        entry = backend.add(text, metadata=metadata)
        return json.dumps({"ok": True, "id": entry.id, "text": entry.text}, indent=2)

    def knowledge_edit(
        entry_id: str,
        text: str,
        supersede: bool = True,
        tags: list[str] | None = None,
    ) -> str:
        """Edit or supersede a knowledge entry."""
        metadata: dict[str, Any] = {}
        if tags:
            metadata["tags"] = tags
        try:
            entry = backend.edit(entry_id, text=text, metadata=metadata, supersede=supersede)
        except ValueError as exc:
            return json.dumps({"ok": False, "error": str(exc)}, indent=2)
        return json.dumps({"ok": True, "id": entry.id, "text": entry.text}, indent=2)

    def knowledge_search(query: str, top_k: int = 5) -> str:
        """Search knowledge entries."""
        hits = backend.search(query, top_k=top_k)
        payload = [
            {
                "id": hit.id,
                "text": hit.text,
                "metadata": hit.metadata,
                "score": hit.score,
            }
            for hit in hits
        ]
        return json.dumps(payload, indent=2)

    return [
        StructuredTool.from_function(
            func=knowledge_add,
            name="knowledge_add",
            description="Add durable knowledge (facts, heuristics, definitions).",
            args_schema=KnowledgeAddInput,
        ),
        StructuredTool.from_function(
            func=knowledge_edit,
            name="knowledge_edit",
            description="Edit or supersede existing knowledge when facts change.",
            args_schema=KnowledgeEditInput,
        ),
        StructuredTool.from_function(
            func=knowledge_search,
            name="knowledge_search",
            description="Search stored knowledge relevant to a query.",
            args_schema=KnowledgeSearchInput,
        ),
    ]


from tools.registry import register_tools

register_tools(create_tools)
