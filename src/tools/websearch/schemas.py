"""Input schemas for the websearch agent tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tools.websearch.backend import Freshness


class WebSearchInput(BaseModel):
    """Input schema for the websearch tool."""

    query: str = Field(description="The search query.")
    count: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of results to return (1-10).",
    )
    freshness: Freshness = Field(
        default="noLimit",
        description="Time range filter for search results.",
    )
