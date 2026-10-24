"""Input schemas for the webfetch agent tool."""

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class WebFetchInput(BaseModel):
    """Input schema for the webfetch tool."""

    url: HttpUrl = Field(description="Public HTTP or HTTPS URL to fetch.")
    max_chars: int = Field(
        default=40_000,
        ge=1_000,
        le=100_000,
        description="Maximum characters of extracted page text to return.",
    )
