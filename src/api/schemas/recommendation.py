"""Pydantic schemas for recommendations and retrieval scores."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.api.schemas.job import JobResponse


class ScoredJob(BaseModel):
    """A job with a retrieval or match score from one source."""

    job_id: str
    score: float
    source: str
    title: Optional[str] = None
    company: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Single recommendation with factor breakdown."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    candidate_id: UUID
    job_id: UUID
    match_score: float
    factor_scores: dict[str, Any]
    retrieval_scores: Optional[dict[str, Any]] = None
    explanation: Optional[str] = None
    rank: Optional[int] = None
    created_at: datetime
    job: Optional[JobResponse] = None


class RecommendationListResponse(BaseModel):
    """Paginated recommendation feed."""

    items: list[RecommendationResponse]
    total: int
    page: int
    per_page: int
    pages: int = Field(description="Total number of pages.")
