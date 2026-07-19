"""Candidate profile API routes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings
from src.api.dependencies import get_db_session, get_settings_dep
from src.api.sanitize import sanitize_text
from src.api.schemas.candidate import CandidatePreferences, CandidateProfile, CandidateResponse
from src.db.candidate_repository import upsert_candidate_profile
from src.db.models import Candidate
from src.ingestion.exceptions import FileTooLargeError, UnsupportedFileTypeError
from src.ingestion.profile_builder import build_profile
from src.ingestion.resume_parser import validate_resume_file

router = APIRouter(prefix="/candidates", tags=["candidates"])


@router.post("", status_code=201, response_model=CandidateResponse)
async def create_candidate(
    resume: UploadFile = File(...),
    github_username: Optional[str] = Form(None),
    preferences: Optional[str] = Form(None),
    settings: Settings = Depends(get_settings_dep),
) -> CandidateResponse:
    """Upload resume and create a candidate profile."""
    suffix = Path(resume.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".doc"}:
        raise HTTPException(status_code=422, detail="Please upload a PDF or DOCX file")

    content = await resume.read()
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="The uploaded file appears to be empty")
    max_mb = settings.ingestion.resume_max_file_bytes // (1024 * 1024)
    if len(content) > settings.ingestion.resume_max_file_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the {max_mb}MB limit",
        )

    pref_model: Optional[CandidatePreferences] = None
    if preferences:
        try:
            pref_model = CandidatePreferences.model_validate(json.loads(preferences))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="Invalid preferences JSON") from exc

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        validate_resume_file(tmp_path, settings)
        profile, embeddings = await build_profile(
            tmp_path,
            github_username=github_username,
            preferences=pref_model,
            settings=settings,
        )
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        from src.ingestion.exceptions import ExtractionFailedError

        if isinstance(exc, ExtractionFailedError) or "read" in str(exc).lower():
            raise HTTPException(
                status_code=422,
                detail="We couldn't read this file. Please try a different version of your resume.",
            ) from exc
        raise HTTPException(status_code=500, detail="Profile build failed. Please try again.") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    from src.db.sync_database import get_sync_session

    with get_sync_session() as session:
        candidate = upsert_candidate_profile(
            session,
            profile,
            resume_text=profile.summary or "",
            resume_filename=resume.filename or "resume",
            github_username=github_username,
            embeddings=embeddings,
        )
        from src.cache.invalidation import invalidate_candidate_recommendation_cache

        invalidate_candidate_recommendation_cache(candidate.id)
        return CandidateResponse(
            id=candidate.id,
            name=candidate.name,
            email=candidate.email,
            github_username=candidate.github_username,
            profile=candidate.profile,
            preferences=candidate.preferences,
            utility_weights=candidate.utility_weights,
            created_at=candidate.created_at,
            updated_at=candidate.updated_at,
        )


@router.get("/{candidate_id}", response_model=CandidateResponse)
async def get_candidate(
    candidate_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> CandidateResponse:
    """Return candidate profile by ID."""
    candidate = await session.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return CandidateResponse(
        id=candidate.id,
        name=candidate.name,
        email=candidate.email,
        github_username=candidate.github_username,
        profile=candidate.profile,
        preferences=candidate.preferences,
        utility_weights=candidate.utility_weights,
        created_at=candidate.created_at,
        updated_at=candidate.updated_at,
    )


@router.patch("/{candidate_id}/preferences", response_model=CandidateResponse)
async def update_preferences(
    candidate_id: UUID,
    body: CandidatePreferences,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings_dep),
) -> CandidateResponse:
    """Update candidate preferences and invalidate cached recommendations."""
    candidate = await session.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")

    if candidate.profile:
        profile = CandidateProfile.model_validate(candidate.profile)
        sanitized = body.model_copy(deep=True)
        for field_name in sanitized.model_dump(exclude_unset=True):
            value = getattr(sanitized, field_name)
            if isinstance(value, list):
                setattr(
                    sanitized,
                    field_name,
                    [
                        sanitize_text(str(v), settings.api.preference_text_max_length)
                        for v in value
                    ],
                )
        from src.matching.preference_utils import patch_merged_preferences

        profile.preferences = patch_merged_preferences(profile.preferences, sanitized)
        candidate.profile = profile.model_dump(mode="json")

    candidate.preferences = body.model_dump(mode="json")
    await session.commit()
    await session.refresh(candidate)

    from src.cache.invalidation import invalidate_candidate_recommendation_cache

    invalidate_candidate_recommendation_cache(candidate_id)

    return CandidateResponse(
        id=candidate.id,
        name=candidate.name,
        email=candidate.email,
        github_username=candidate.github_username,
        profile=candidate.profile,
        preferences=candidate.preferences,
        utility_weights=candidate.utility_weights,
        created_at=candidate.created_at,
        updated_at=candidate.updated_at,
    )
