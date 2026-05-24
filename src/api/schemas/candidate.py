"""Pydantic schemas for candidates."""

from datetime import datetime
from typing import Any, Generic, Literal, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")
PreferenceSource = Literal["explicit", "inferred", "default"]


class PreferenceField(BaseModel, Generic[T]):
    """A preference value with provenance tracking."""

    value: Optional[T] = None
    source: PreferenceSource = "default"


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


class MergedPreferences(BaseModel):
    """Merged explicit and inferred preferences with source tracking."""

    job_types: PreferenceField[list[str]] = Field(default_factory=PreferenceField)
    work_models: PreferenceField[list[str]] = Field(default_factory=PreferenceField)
    locations: PreferenceField[list[str]] = Field(default_factory=PreferenceField)
    needs_sponsorship: PreferenceField[bool] = Field(default_factory=PreferenceField)
    salary_min: PreferenceField[int] = Field(default_factory=PreferenceField)
    salary_max: PreferenceField[int] = Field(default_factory=PreferenceField)
    company_stages: PreferenceField[list[str]] = Field(default_factory=PreferenceField)
    company_sizes: PreferenceField[list[str]] = Field(default_factory=PreferenceField)
    target_roles: PreferenceField[list[str]] = Field(default_factory=PreferenceField)
    target_industries: PreferenceField[list[str]] = Field(default_factory=PreferenceField)
    avoid_industries: PreferenceField[list[str]] = Field(default_factory=PreferenceField)
    priorities: PreferenceField[list[str]] = Field(default_factory=PreferenceField)
    preferred_company_stage: PreferenceField[str] = Field(default_factory=PreferenceField)
    preferred_team_size: PreferenceField[str] = Field(default_factory=PreferenceField)
    preferred_work_style: PreferenceField[str] = Field(default_factory=PreferenceField)
    likely_looking_for: PreferenceField[str] = Field(default_factory=PreferenceField)


class ProfileSkill(BaseModel):
    """A skill on the canonical candidate profile."""

    name: str
    category: str = "other"
    proficiency: str = "intermediate"
    depth_score: float = 0.0
    years_used: Optional[float] = None
    context: Optional[str] = None
    sources: list[str] = Field(default_factory=list)
    esco_uri: Optional[str] = None
    esco_label: Optional[str] = None
    esco_match_type: Optional[str] = None
    esco_match_confidence: float = 0.0


class ProfileExperience(BaseModel):
    """Work experience on the canonical candidate profile."""

    company: str
    title: str
    start_date: str
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    description: str = ""
    domain: str = ""
    company_size_estimate: str = ""
    company_stage_estimate: str = ""
    role_type: str = "ic"
    key_achievements: list[str] = Field(default_factory=list)


class ProfileEducation(BaseModel):
    """Education on the canonical candidate profile."""

    institution: str
    degree: str = ""
    field: str = ""
    graduation_year: Optional[int] = None


class GitHubSummary(BaseModel):
    """Condensed GitHub analysis attached to a profile."""

    username: str
    overall_assessment: str = ""
    top_languages: list[str] = Field(default_factory=list)
    activity_level: str = ""
    total_repos: int = 0
    repos_last_6_months: int = 0
    followers: int = 0
    inferred_skills: list[str] = Field(default_factory=list)
    top_repo_names: list[str] = Field(default_factory=list)


class ESCOLinkedSkill(BaseModel):
    """A skill successfully linked to an ESCO concept."""

    original_name: str
    esco_uri: str
    esco_label: str
    match_type: str
    confidence: float


class CandidateProfile(BaseModel):
    """Structured candidate profile used by the matching engine."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    skills: list[ProfileSkill] = Field(default_factory=list)
    experience: list[ProfileExperience] = Field(default_factory=list)
    education: list[ProfileEducation] = Field(default_factory=list)
    total_years_experience: float = 0.0
    domains: list[str] = Field(default_factory=list)
    role_archetype: str = "generalist"
    career_trajectory: str = "lateral"
    github_summary: Optional[GitHubSummary] = None
    preferences: MergedPreferences = Field(default_factory=MergedPreferences)
    esco_linked_skills: list[ESCOLinkedSkill] = Field(default_factory=list)
    summary: str = ""


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
    utility_weights: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
