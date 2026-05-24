"""Rule-based fallback when LLM extraction fails."""

from __future__ import annotations

import re

from src.ingestion.schemas import (
    ExtractedEducation,
    ExtractedExperience,
    ExtractedProfile,
    ExtractedSkill,
    InferredPreferences,
)

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")

_SKILL_KEYWORDS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "go",
    "rust",
    "sql",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "react",
    "node",
    "fastapi",
    "django",
    "flask",
    "machine learning",
    "deep learning",
    "pytorch",
    "tensorflow",
    "pandas",
    "numpy",
    "spark",
    "kafka",
    "postgresql",
    "mongodb",
    "redis",
    "elasticsearch",
    "git",
    "ci/cd",
    "terraform",
    "linux",
]


def extract_fallback_profile(resume_text: str) -> ExtractedProfile:
    """Build a minimal profile from regex and keyword heuristics."""
    lowered = resume_text.lower()
    email_match = _EMAIL_RE.search(resume_text)
    phone_match = _PHONE_RE.search(resume_text)

    skills: list[ExtractedSkill] = []
    for keyword in _SKILL_KEYWORDS:
        if keyword in lowered:
            skills.append(
                ExtractedSkill(
                    name=keyword.title() if keyword.islower() else keyword,
                    category="other",
                    proficiency="intermediate",
                ),
            )

    if not skills:
        skills.append(ExtractedSkill(name="Generalist", category="other", proficiency="intermediate"))

    experience = [
        ExtractedExperience(
            company="Unknown",
            title="Professional",
            start_date="2020-01",
            end_date="present",
            description=resume_text[:500],
            domain="general",
        ),
    ]

    return ExtractedProfile(
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0).strip() if phone_match else None,
        skills=skills,
        experience=experience,
        education=[],
        total_years_experience=1.0,
        domains=["general"],
        role_archetype="generalist",
        career_trajectory="lateral",
        inferred_preferences=InferredPreferences(),
        summary="Fallback profile extracted without LLM.",
    )
