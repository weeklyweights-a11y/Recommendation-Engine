"""Recommendation API routes."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings
from src.api.dependencies import get_db_session, get_settings_dep
from src.api.schemas.job import JobResponse
from src.api.schemas.recommendation import (
    FeedSections,
    PaginationMeta,
    PipelineStatsResponse,
    RecommendationListResponse,
    RecommendationResponse,
)
from src.db.models import Candidate
from src.db.recommendation_repository import load_cached_recommendations
from src.db.sync_database import get_sync_session
from src.matching.recommendation_pipeline import (
    _materialize_ranked_jobs,
    _ranked_from_stored,
    run_recommendation_pipeline,
)
from src.matching.schemas import PipelineResult, RankedJob

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def _parse_explanation(raw: Optional[str]) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _job_response(job: object | None) -> JobResponse | None:
    if job is None:
        return None
    if isinstance(job, JobResponse):
        return job
    return JobResponse.model_validate(job)


def _to_response(candidate_id: UUID, item: RankedJob, rec_id: Optional[UUID] = None) -> RecommendationResponse:
    return RecommendationResponse(
        id=rec_id,
        candidate_id=candidate_id,
        job_id=UUID(item.job_id),
        match_score=item.match_score,
        match_percentage=item.match_percentage,
        factor_scores=item.factor_scores,
        retrieval_scores=item.retrieval_scores,
        explanation=_parse_explanation(item.explanation),
        rank=item.rank,
        feed_section=item.feed_section,
        job=_job_response(item.job),
    )


def _stats_response(result: PipelineResult) -> Optional[PipelineStatsResponse]:
    if result.stats is None:
        return None
    return PipelineStatsResponse(
        filter_funnel=result.stats.filter_funnel,
        retrieval_overlap=result.stats.retrieval_overlap,
        timing_ms=result.stats.timing_ms,
        warnings=result.stats.warnings,
    )


@router.get("/{candidate_id}", response_model=RecommendationListResponse)
async def get_recommendations(
    candidate_id: UUID,
    page: int = Query(default=None, ge=1),
    per_page: int = Query(default=None, ge=1, le=1000),
    refresh: bool = Query(default=False),
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings_dep),
) -> RecommendationListResponse:
    """Return personalized job recommendations for a candidate."""
    candidate = await session.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if not candidate.profile:
        raise HTTPException(status_code=400, detail="Candidate profile missing")

    page_val = page if page is not None else settings.api.default_page
    per_page_val = per_page if per_page is not None else settings.api.default_per_page
    per_page_val = min(per_page_val, settings.api.max_per_page)

    if refresh:
        result = await asyncio.to_thread(
            run_recommendation_pipeline,
            candidate_id,
            refresh=True,
        )
    else:
        result = None
        with get_sync_session() as sync_sess:
            cached_rows = load_cached_recommendations(sync_sess, candidate_id, settings=settings)
            if cached_rows:
                result = PipelineResult(
                    ranked_jobs=_materialize_ranked_jobs(_ranked_from_stored(cached_rows)),
                    from_cache=True,
                )
        if result is None:
            result = await asyncio.to_thread(
                run_recommendation_pipeline,
                candidate_id,
                refresh=False,
            )

    total = len(result.ranked_jobs)
    pages = max(1, math.ceil(total / per_page_val)) if total else 0
    start = (page_val - 1) * per_page_val
    end = start + per_page_val
    page_items = result.ranked_jobs[start:end]

    strong = sum(1 for r in result.ranked_jobs if r.feed_section == "strong_match")
    exploring = sum(1 for r in result.ranked_jobs if r.feed_section == "worth_exploring")

    responses = [_to_response(candidate_id, item) for item in page_items]
    return RecommendationListResponse(
        recommendations=responses,
        pagination=PaginationMeta(
            page=page_val,
            per_page=per_page_val,
            total=total,
            total_pages=pages,
        ),
        pipeline_stats=_stats_response(result),
        feed_sections=FeedSections(strong_matches=strong, worth_exploring=exploring),
        items=responses,
        total=total,
        page=page_val,
        per_page=per_page_val,
        pages=pages,
    )
