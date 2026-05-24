"""Pydantic schemas for recommendations and retrieval scores."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.api.schemas.job import JobResponse
from src.matching.schemas import FilterFunnel, PipelineTiming, RetrievalStats


class ScoredJob(BaseModel):
    """A job with a retrieval or match score from one source."""

    job_id: str
    score: float
    source: str
    title: Optional[str] = None
    company: Optional[str] = None
    dimension_scores: Optional[dict[str, float]] = None
    matched_skills: Optional[list[dict]] = None


class PaginationMeta(BaseModel):
    """Pagination metadata."""

    page: int
    per_page: int
    total: int
    total_pages: int


class PipelineStatsResponse(BaseModel):
    """Pipeline diagnostics for API responses."""

    filter_funnel: Optional[FilterFunnel] = None
    retrieval_overlap: Optional[RetrievalStats] = None
    timing_ms: Optional[PipelineTiming] = None
    warnings: list[str] = Field(default_factory=list)


class FeedSections(BaseModel):
    """Counts by feed section."""

    strong_matches: int = 0
    worth_exploring: int = 0


class RecommendationResponse(BaseModel):
    """Single recommendation with factor breakdown."""

    model_config = ConfigDict(from_attributes=True)

    id: Optional[UUID] = None
    candidate_id: UUID
    job_id: UUID
    match_score: float
    match_percentage: int = 0
    factor_scores: dict[str, Any]
    retrieval_scores: Optional[dict[str, Any]] = None
    explanation: Optional[Any] = None
    rank: Optional[int] = None
    feed_section: str = "strong_match"
    created_at: Optional[datetime] = None
    job: Optional[JobResponse] = None


class RecommendationListResponse(BaseModel):
    """Paginated recommendation feed."""

    recommendations: list[RecommendationResponse]
    pagination: PaginationMeta
    pipeline_stats: Optional[PipelineStatsResponse] = None
    feed_sections: Optional[FeedSections] = None
    items: list[RecommendationResponse] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    per_page: int = 20
    pages: int = 0
