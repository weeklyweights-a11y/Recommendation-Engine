"""Add resume/GitHub skills the LLM missed — without ingesting resume prose."""

from __future__ import annotations

import re
from typing import Optional

from config.settings import Settings, get_settings
from src.ingestion.schemas import ExtractedProfile, ExtractedSkill, GitHubProfile
from src.ingestion.skill_filters import filter_skill_names, is_plausible_skill

# Technologies to detect inside the dedicated skills section only.
_SKILL_SECTION_KEYWORDS: tuple[str, ...] = (
    "python",
    "java",
    "javascript",
    "typescript",
    "sql",
    "postgresql",
    "postgres",
    "mysql",
    "mongodb",
    "redis",
    "snowflake",
    "duckdb",
    "spark",
    "pyspark",
    "kafka",
    "airflow",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "terraform",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "sklearn",
    "pandas",
    "polars",
    "numpy",
    "xgboost",
    "langchain",
    "langgraph",
    "huggingface",
    "transformers",
    "mlflow",
    "fastapi",
    "streamlit",
    "tableau",
    "git",
    "github",
    "machine learning",
    "deep learning",
    "nlp",
    "rag",
    "llm",
    "neo4j",
    "chromadb",
    "pinecone",
    "plotly",
    "grafana",
    "prometheus",
    "gitlab",
    "ci/cd",
    "rest api",
)

_SKILL_SECTION_RE = re.compile(
    r"(?is)(?:technical\s+skills?|core\s+competencies|skills?|technologies|tools?)"
    r"\s*[:\-]\s*(.+?)(?=\n\s*\n|\n[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\n|\Z)",
)


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


_COMPOUND_SPLIT_RE = re.compile(r"\s*[/|&]\s*")


def _split_compound_name(name: str) -> list[str]:
    parts: list[str] = []
    for piece in _COMPOUND_SPLIT_RE.split(name):
        token = piece.strip()
        if 2 <= len(token) <= 56:
            parts.append(token)
    return parts if len(parts) > 1 else [name.strip()]


def _title_case(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        return cleaned
    if cleaned.isupper() and len(cleaned) <= 5:
        return cleaned
    return cleaned[0].upper() + cleaned[1:]


def _extract_skills_section_text(resume_text: str) -> str:
    blocks: list[str] = []
    for match in _SKILL_SECTION_RE.finditer(resume_text):
        blocks.append(match.group(1))
    return "\n".join(blocks)


def _skills_from_section_block(section_text: str) -> list[str]:
    """Parse list items only from the skills section block."""
    found: list[str] = []
    for line in section_text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\s•\-\*·]+\s*", "", line)
        if ":" in line and line.index(":") < 40:
            _, _, rest = line.partition(":")
            line = rest.strip()
        for part in re.split(r"[,|•·;/]", line):
            token = part.strip()
            if is_plausible_skill(token):
                found.append(token)
    return filter_skill_names(found)


def _skills_from_section_keywords(section_text: str) -> list[str]:
    lowered = section_text.lower()
    found: list[str] = []
    for keyword in _SKILL_SECTION_KEYWORDS:
        if keyword in lowered:
            found.append(_title_case(keyword))
    return filter_skill_names(found)


def _skills_from_existing_compounds(skills: list[ExtractedSkill]) -> list[str]:
    found: list[str] = []
    for skill in skills:
        for part in _split_compound_name(skill.name):
            if _normalize(part) != _normalize(skill.name) and is_plausible_skill(part):
                found.append(part)
    return filter_skill_names(found)


def _skills_from_github(github: Optional[GitHubProfile]) -> list[str]:
    if github is None:
        return []
    names: list[str] = []
    for skill in github.inferred_skills:
        names.append(skill)
    for lang in github.languages_distribution:
        names.append(lang)
    for repo in github.top_repos:
        for lang in repo.languages:
            names.append(lang)
        for topic in repo.topics:
            names.append(str(topic).replace("-", " "))
    return filter_skill_names(names)


def enrich_extracted_skills(
    profile: ExtractedProfile,
    resume_text: str,
    github: Optional[GitHubProfile] = None,
    settings: Optional[Settings] = None,
) -> ExtractedProfile:
    """
    Append vetted skills from the skills section and GitHub.

    Does not scan full resume bullets (that produced email/prose false positives).
    """
    cfg = (settings or get_settings()).ingestion
    max_skills = cfg.extraction_max_skills

    merged = [s for s in profile.skills if is_plausible_skill(s.name)]
    existing = {_normalize(s.name) for s in merged if s.name}

    section_text = _extract_skills_section_text(resume_text)
    candidates: list[str] = []
    candidates.extend(_skills_from_existing_compounds(merged))
    if section_text.strip():
        candidates.extend(_skills_from_section_block(section_text))
        candidates.extend(_skills_from_section_keywords(section_text))
    candidates.extend(_skills_from_github(github))
    candidates = filter_skill_names(candidates)

    for raw in candidates:
        key = _normalize(raw)
        if not key or key in existing:
            continue
        existing.add(key)
        merged.append(
            ExtractedSkill(
                name=_title_case(raw),
                category="other",
                proficiency="intermediate",
                context="skills section or GitHub",
            ),
        )
        if max_skills > 0 and len(merged) >= max_skills:
            break

    if not merged:
        merged.append(
            ExtractedSkill(name="Generalist", category="other", proficiency="intermediate"),
        )

    return profile.model_copy(update={"skills": merged})
