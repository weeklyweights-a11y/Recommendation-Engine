"""Pydantic schemas for GitHub preview API."""

from typing import Optional

from pydantic import BaseModel, Field


class GitHubPreviewResponse(BaseModel):
    """Lightweight GitHub user preview."""

    username: str
    name: str
    avatar_url: Optional[str] = None
    public_repos: int = 0
    top_languages: list[str] = Field(default_factory=list)
