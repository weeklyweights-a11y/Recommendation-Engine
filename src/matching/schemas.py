"""Matching pipeline internal schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SkillOverlap(BaseModel):
    """ESCO skill overlap between candidate and job."""

    direct_matches: list[dict[str, Any]] = Field(default_factory=list)
    one_hop_matches: list[dict[str, Any]] = Field(default_factory=list)
    two_hop_matches: list[dict[str, Any]] = Field(default_factory=list)
    unmatched_job_skills: list[str] = Field(default_factory=list)
    unmatched_candidate_skills: list[str] = Field(default_factory=list)


class RetrievalStats(BaseModel):
    """Overlap statistics across retrieval sources."""

    all_three: int = 0
    exactly_two: int = 0
    only_one: int = 0
    total_unique: int = 0


class FusedResult(BaseModel):
    """A job with fused hybrid retrieval scores."""

    job_id: str
    fused_score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    graph_score: float = 0.0
    sources: list[str] = Field(default_factory=list)
    vector_dimension_scores: Optional[dict[str, float]] = None
    graph_matched_skills: Optional[list[dict[str, Any]]] = None
