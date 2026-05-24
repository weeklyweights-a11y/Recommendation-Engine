"""Feedback API routes."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db_session
from src.api.schemas.feedback import FeedbackCreate, FeedbackResponse
from src.db.models import Candidate, Feedback, Job
from src.db.recommendation_repository import delete_recommendations_for_candidate

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", status_code=201, response_model=FeedbackResponse)
async def create_feedback(
    body: FeedbackCreate,
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackResponse:
    """Record user feedback on a job recommendation."""
    candidate = await session.get(Candidate, body.candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    job = await session.get(Job, body.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if body.action not in {"saved", "dismissed", "applied"}:
        raise HTTPException(status_code=422, detail="Invalid feedback action")

    if body.action == "applied" and job.source_url:
        import logging

        logging.getLogger(__name__).info(
            "Applied click tracked candidate=%s job=%s url=%s",
            body.candidate_id,
            body.job_id,
            job.source_url,
        )

    row = Feedback(
        candidate_id=body.candidate_id,
        job_id=body.job_id,
        action=body.action,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)

    from src.db.sync_database import get_sync_session

    with get_sync_session() as sync_sess:
        delete_recommendations_for_candidate(sync_sess, body.candidate_id)

    return FeedbackResponse.model_validate(row)


@router.get("/{candidate_id}", response_model=list[FeedbackResponse])
async def list_feedback(
    candidate_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> list[FeedbackResponse]:
    """List feedback entries for a candidate."""
    stmt = (
        select(Feedback)
        .where(Feedback.candidate_id == candidate_id)
        .order_by(Feedback.created_at.desc())
    )
    rows = (await session.scalars(stmt)).all()
    return [FeedbackResponse.model_validate(row) for row in rows]
