"""Pydantic schemas for job listings."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobBase(BaseModel):
    """Shared job fields."""

    title: str
    company: str
    description: str
    location: Optional[str] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    currency: Optional[str] = "USD"
    remote_type: Optional[str] = None
    sponsorship_available: Optional[bool] = None
    company_size: Optional[str] = None
    company_stage: Optional[str] = None
    industry: Optional[str] = None
    experience_level: Optional[str] = None
    skills_extracted: Optional[list[Any]] = None
    source_url: Optional[str] = None
    source_platform: Optional[str] = None
    posted_date: Optional[datetime] = None

    @field_validator("skills_extracted", mode="before")
    @classmethod
    def normalize_skills_extracted(cls, value: object) -> Optional[list[Any]]:
        """Accept legacy dict wrapper {\"skills\": [...]} stored in JSONB."""
        if value is None:
            return None
        if isinstance(value, dict):
            skills = value.get("skills")
            return skills if isinstance(skills, list) else []
        if isinstance(value, list):
            return value
        return None


class JobCreate(JobBase):
    """Schema for creating a job."""


class JobResponse(JobBase):
    """Job response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    is_embedded: bool = False
    created_at: datetime
    updated_at: datetime


class JobListResponse(BaseModel):
    """Paginated job list."""

    items: list[JobResponse]
    total: int
    page: int
    per_page: int
    pages: int = Field(description="Total number of pages.")
