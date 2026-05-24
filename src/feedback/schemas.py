"""Pydantic schemas for feedback analysis and weight adjustment."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeedbackSummary(BaseModel):
    """Aggregated feedback patterns for a candidate."""

    total_actions: int = 0
    saved_count: int = 0
    dismissed_count: int = 0
    applied_count: int = 0
    preferred_industries: list[tuple[str, float]] = Field(default_factory=list)
    preferred_stages: list[tuple[str, float]] = Field(default_factory=list)
    preferred_sizes: list[tuple[str, float]] = Field(default_factory=list)
    preferred_remote_types: list[tuple[str, float]] = Field(default_factory=list)
    preferred_skills: list[tuple[str, float]] = Field(default_factory=list)
    avoided_industries: list[tuple[str, float]] = Field(default_factory=list)
    avoided_stages: list[tuple[str, float]] = Field(default_factory=list)
    avoided_remote_types: list[tuple[str, float]] = Field(default_factory=list)
    strong_positive_signals: list[str] = Field(default_factory=list)
    strong_negative_signals: list[str] = Field(default_factory=list)
    has_enough_data: bool = False
    avg_skill_fit_saved: float | None = None
    avg_semantic_saved: float | None = None
    avg_other_factors_saved: float | None = None


class AdjustedWeights(BaseModel):
    """Result of a weight adjustment pass."""

    weights: dict[str, float]
    adjustments_made: list[str] = Field(default_factory=list)
    previous_weights: dict[str, float] = Field(default_factory=dict)
    adjustment_magnitude: float = 0.0
