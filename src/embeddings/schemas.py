"""Embedding result models."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class CandidateEmbeddings(BaseModel):
    """Four semantic vectors describing a candidate profile."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skill: np.ndarray = Field(...)
    domain: np.ndarray = Field(...)
    role: np.ndarray = Field(...)
    environment: np.ndarray = Field(...)


class JobEmbeddings(BaseModel):
    """Four semantic vectors describing a job listing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skill: np.ndarray = Field(...)
    domain: np.ndarray = Field(...)
    role: np.ndarray = Field(...)
    environment: np.ndarray = Field(...)


class JobFields(BaseModel):
    """Structured fields extracted from a job description."""

    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    domain: str = ""
    role_level: str = ""
    role_type: str = ""
    responsibilities_summary: str = ""
    company_description: str = ""
    team_info: str = ""
    work_style_signals: str = ""
    industry_keywords_from_description: str = ""
    job_title: str = ""
    industry: str = ""
    company_stage: str = ""
    company_size: str = ""
    remote_type: str = ""
    extraction_method: Literal["rule", "llm"] = "rule"


class LinkedJobSkill(BaseModel):
    """A skill linked to ESCO for storage in skills_extracted JSONB."""

    name: str
    esco_uri: Optional[str] = None
    esco_label: Optional[str] = None
    match_type: Optional[str] = None
    confidence: Optional[float] = None


class SkillsExtractedPayload(BaseModel):
    """Canonical wrapper stored in jobs.skills_extracted."""

    extraction_method: Literal["rule", "llm"] = "rule"
    skills: list[LinkedJobSkill] = Field(default_factory=list)
