"""Parse jobs.skills_extracted JSONB (wrapper or legacy list)."""

from __future__ import annotations

from typing import Any, Optional

from src.embeddings.schemas import LinkedJobSkill, SkillsExtractedPayload


def parse_skills_extracted(raw: Any) -> list[LinkedJobSkill]:
    """Return normalized skill entries from JSONB value."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        if "skills" in raw:
            payload = SkillsExtractedPayload.model_validate(raw)
            return payload.skills
        return []
    if isinstance(raw, list):
        skills: list[LinkedJobSkill] = []
        for item in raw:
            if isinstance(item, dict):
                skills.append(LinkedJobSkill.model_validate(item))
            elif isinstance(item, str) and item.strip():
                skills.append(LinkedJobSkill(name=item.strip()))
        return skills
    return []


def skills_to_search_text(skills: list[LinkedJobSkill]) -> str:
    """Flatten skills for Elasticsearch BM25 text field."""
    parts: list[str] = []
    for skill in skills:
        if skill.name:
            parts.append(skill.name)
        if skill.esco_label and skill.esco_label != skill.name:
            parts.append(skill.esco_label)
    return " ".join(parts)


def build_skills_payload(
    skills: list[LinkedJobSkill],
    extraction_method: str = "rule",
) -> dict[str, Any]:
    """Build canonical wrapper dict for PostgreSQL JSONB."""
    payload = SkillsExtractedPayload(
        extraction_method=extraction_method,  # type: ignore[arg-type]
        skills=skills,
    )
    return payload.model_dump(mode="json")


def parse_extraction_method(raw: Any) -> Optional[str]:
    """Read extraction_method from wrapper JSONB if present."""
    if isinstance(raw, dict):
        method = raw.get("extraction_method")
        if isinstance(method, str):
            return method
    return None
