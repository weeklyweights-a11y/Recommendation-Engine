"""Pydantic models for LLM extraction and GitHub analysis."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ExtractedSkill(BaseModel):
    """Skill extracted from resume text."""

    name: str
    category: str = "other"
    proficiency: str = "intermediate"
    years_used: Optional[float] = None
    context: Optional[str] = None


class ExtractedExperience(BaseModel):
    """Work experience entry from resume."""

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


class ExtractedEducation(BaseModel):
    """Education entry from resume."""

    institution: str
    degree: str = ""
    field: str = ""
    graduation_year: Optional[int] = None


class InferredPreferences(BaseModel):
    """Preferences inferred from career history."""

    preferred_company_stage: Optional[str] = None
    preferred_team_size: Optional[str] = None
    preferred_work_style: Optional[str] = None
    likely_looking_for: Optional[str] = None


class ExtractedProfile(BaseModel):
    """Structured profile from LLM extraction."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    skills: list[ExtractedSkill] = Field(default_factory=list)
    experience: list[ExtractedExperience] = Field(default_factory=list)
    education: list[ExtractedEducation] = Field(default_factory=list)
    total_years_experience: float = 0.0
    domains: list[str] = Field(default_factory=list)
    role_archetype: str = "generalist"
    career_trajectory: str = "lateral"
    inferred_preferences: InferredPreferences = Field(default_factory=InferredPreferences)
    summary: str = ""

    @field_validator("skills")
    @classmethod
    def skills_not_empty(cls, value: list[ExtractedSkill]) -> list[ExtractedSkill]:
        """Require at least one skill."""
        if not value:
            raise ValueError("skills must not be empty")
        return value

    @field_validator("experience")
    @classmethod
    def experience_not_empty(
        cls,
        value: list[ExtractedExperience],
    ) -> list[ExtractedExperience]:
        """Require at least one experience entry."""
        if not value:
            raise ValueError("experience must not be empty")
        return value

    @field_validator("total_years_experience")
    @classmethod
    def non_negative_experience(cls, value: float) -> float:
        """Experience years must be non-negative."""
        if value < 0:
            raise ValueError("total_years_experience must be >= 0")
        return value


class ActivityMetrics(BaseModel):
    """GitHub activity aggregates."""

    total_repos: int = 0
    repos_last_6_months: int = 0
    repos_last_year: int = 0
    most_active_language: str = ""
    avg_stars: float = 0.0
    total_stars: int = 0


class RepoAnalysis(BaseModel):
    """Analysis for a single GitHub repository."""

    name: str
    complexity: str = "low"
    languages: list[str] = Field(default_factory=list)
    description: str = ""
    stars: int = 0
    last_active: str = ""
    readme_summary: Optional[str] = None
    production_signals: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    has_wiki: bool = False
    default_branch: str = "main"


class GitHubProfile(BaseModel):
    """Structured GitHub public profile analysis."""

    username: str
    name: Optional[str] = None
    bio: Optional[str] = None
    public_repos: int = 0
    followers: int = 0
    following: int = 0
    account_age_years: float = 0.0
    languages_distribution: dict[str, float] = Field(default_factory=dict)
    activity_metrics: ActivityMetrics = Field(default_factory=ActivityMetrics)
    top_repos: list[RepoAnalysis] = Field(default_factory=list)
    inferred_skills: list[str] = Field(default_factory=list)
    overall_assessment: str = "inactive"


class TokenUsage(BaseModel):
    """LLM token usage for cost tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ExtractionResult(BaseModel):
    """LLM extraction output with optional usage metadata."""

    profile: ExtractedProfile
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
