"""Pydantic models for LLM extraction and GitHub analysis."""

from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class ExtractedSkill(BaseModel):
    """Skill extracted from resume text."""

    name: str = Field(validation_alias=AliasChoices("name", "skill"))
    category: str = "other"
    proficiency: str = "intermediate"
    years_used: Optional[float] = None
    context: Optional[str] = None


class ExtractedExperience(BaseModel):
    """Work experience entry from resume."""

    model_config = ConfigDict(populate_by_name=True)

    company: str
    title: str = Field(validation_alias=AliasChoices("title", "role", "job_title"))
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

    @field_validator("degree", "field", mode="before")
    @classmethod
    def empty_str_if_null(cls, value: object) -> str:
        """Coerce null LLM values to empty strings."""
        return "" if value is None else str(value)


class InferredPreferences(BaseModel):
    """Preferences inferred from career history."""

    preferred_company_stage: Optional[str] = None
    preferred_team_size: Optional[str] = None
    preferred_work_style: Optional[str] = None
    likely_looking_for: Optional[str] = None

    @field_validator(
        "preferred_company_stage",
        "preferred_team_size",
        "preferred_work_style",
        "likely_looking_for",
        mode="before",
    )
    @classmethod
    def coerce_preference_text(cls, value: object) -> Optional[str]:
        """Allow lists from the LLM by joining into a single string."""
        if value is None:
            return None
        if isinstance(value, list):
            parts = [str(item) for item in value if item]
            return ", ".join(parts) if parts else None
        return str(value)


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

    @field_validator("inferred_preferences", mode="before")
    @classmethod
    def coerce_inferred_preferences(cls, value: object) -> object:
        """Accept dict, model, or list of strings from the LLM."""
        if value is None:
            return InferredPreferences()
        if isinstance(value, list):
            joined = ", ".join(str(item) for item in value if item)
            return InferredPreferences(likely_looking_for=joined or None)
        return value

    @field_validator("total_years_experience", mode="before")
    @classmethod
    def coerce_total_years(cls, value: object) -> float:
        """Coerce string numerics from LLM output."""
        if value is None or value == "":
            return 0.0
        return float(value)

    @field_validator("total_years_experience")
    @classmethod
    def non_negative_experience(cls, value: float) -> float:
        """Experience years must be non-negative."""
        if value < 0:
            raise ValueError("total_years_experience must be >= 0")
        return value

    @model_validator(mode="after")
    def required_sections_present(self) -> "ExtractedProfile":
        """Reject incomplete LLM output (Pydantic skips field validators on omitted keys)."""
        if not self.skills:
            raise ValueError("skills must not be empty")
        if not self.experience:
            raise ValueError("experience must not be empty")
        if not (self.summary or "").strip():
            raise ValueError("summary must not be empty")
        return self


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
