"""GitHub preview API routes."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from config.settings import Settings
from src.api.dependencies import get_settings_dep
from src.api.schemas.github import GitHubPreviewResponse
from src.ingestion.exceptions import GitHubRateLimitedError, GitHubUserNotFoundError
from src.ingestion.github_fetcher import fetch_github_preview, sanitize_github_username

router = APIRouter(prefix="/github", tags=["github"])


@router.get("/preview", response_model=GitHubPreviewResponse)
async def github_preview(
    username: str = Query(..., min_length=1, max_length=39),
    settings: Settings = Depends(get_settings_dep),
) -> GitHubPreviewResponse:
    """Return a lightweight GitHub profile preview for onboarding."""
    try:
        sanitize_github_username(username, settings=settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        data = await fetch_github_preview(username, settings=settings)
    except GitHubUserNotFoundError as exc:
        raise HTTPException(status_code=404, detail="GitHub user not found") from exc
    except GitHubRateLimitedError as exc:
        raise HTTPException(
            status_code=503,
            detail="GitHub rate limit reached. Try again shortly or skip GitHub.",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not reach GitHub. Try again in a moment.",
        ) from exc

    return GitHubPreviewResponse.model_validate(data)
