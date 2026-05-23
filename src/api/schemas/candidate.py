"""Pydantic schemas for candidates."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CandidatePreferences(BaseModel):
    """Explicit user preferences from onboarding (Phase 5)."""

    job_types: list[str] = Field(default_factory=list)
    work_models: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    visa_sponsorship_needed: Optional[bool] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    company_stages: list[str] = Field(default_factory=list)
    company_sizes: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    industries_target: list[str] = Field(default_factory=list)
    industries_avoid: list[str] = Field(default_factory=list)
    priority_ranking: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    """Structured candidate profile (LLM output, Phase 2)."""

    skills: list[dict[str, Any]] = Field(default_factory=list)
    experience: list[dict[str, Any]] = Field(default_factory=list)
    education: list[dict[str, Any]] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    role_archetype: Optional[str] = None
    career_trajectory: Optional[str] = None
    inferred_preferences: dict[str, Any] = Field(default_factory=dict)


class CandidateCreate(BaseModel):
    """Schema for creating a candidate."""

    name: Optional[str] = None
    email: Optional[str] = None
    resume_text: Optional[str] = None
    github_username: Optional[str] = None
    preferences: Optional[CandidatePreferences] = None


class CandidateResponse(BaseModel):
    """Candidate response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: Optional[str] = None
    email: Optional[str] = None
    github_username: Optional[str] = None
    profile: Optional[dict[str, Any]] = None
    preferences: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
