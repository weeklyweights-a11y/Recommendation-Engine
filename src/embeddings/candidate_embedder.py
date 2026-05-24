"""Multi-vector candidate embedding generation."""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import Settings, get_settings
from src.api.schemas.candidate import CandidateProfile, ProfileSkill
from src.embeddings.encoder import get_encoder
from src.embeddings.schemas import CandidateEmbeddings
from src.knowledge_graph.skill_expander import expand_skill

logger = logging.getLogger(__name__)


def _skill_line(skill: ProfileSkill, settings: Settings) -> str:
    parts = [f"{skill.name} ({skill.category}, {skill.proficiency})"]
    if skill.context:
        parts.append(f"used for {skill.context}")
    if skill.esco_uri:
        try:
            related = expand_skill(skill.esco_uri, max_hops=settings.ingestion.skill_expansion_max_hops_embed)
            labels = [item.label for item in related[:5]]
            if labels:
                parts.append(f"also related to: {', '.join(labels)}")
        except Exception as exc:
            logger.debug("Skill expansion skipped for %s: %s", skill.name, exc)
    return ", ".join(parts)


def _build_skill_text(profile: CandidateProfile, settings: Settings) -> str:
    ordered = sorted(profile.skills, key=lambda s: s.depth_score, reverse=True)
    lines: list[str] = []
    for skill in ordered[:30]:
        line = _skill_line(skill, settings)
        lines.append(line)
        if skill.depth_score >= 0.7:
            lines.append(line)
    return "Skills: " + "; ".join(lines) if lines else ""


def _build_domain_text(profile: CandidateProfile) -> str:
    parts: list[str] = []
    for exp in profile.experience[:8]:
        duration = f"{exp.duration_months} months" if exp.duration_months else "unknown duration"
        parts.append(
            f"{duration} in {exp.domain or 'general'} working on {exp.description[:200]}",
        )
    domain_list = ", ".join(profile.domains)
    if domain_list:
        parts.append(f"Domains: {domain_list}")
    return "Domain experience: " + ". ".join(parts) if parts else ""


def _build_role_text(profile: CandidateProfile) -> str:
    recent_roles = []
    for exp in profile.experience[:5]:
        recent_roles.append(
            f"{exp.title} at {exp.company} ({exp.role_type}, {exp.company_stage_estimate})",
        )
    achievements: list[str] = []
    for exp in profile.experience[:3]:
        achievements.extend(exp.key_achievements[:2])
    achievement_text = ", ".join(achievements[:6])
    return (
        f"Role profile: {profile.role_archetype}. "
        f"Career trajectory: {profile.career_trajectory}. "
        f"Recent roles: {', '.join(recent_roles)}. "
        f"Key achievements: {achievement_text}."
    )


def _pref_value(field) -> str:
    return str(field.value) if field and field.value is not None else ""


def _build_environment_text(profile: CandidateProfile) -> str:
    stages = {exp.company_stage_estimate for exp in profile.experience if exp.company_stage_estimate}
    sizes = {exp.company_size_estimate for exp in profile.experience if exp.company_size_estimate}
    prefs = profile.preferences
    return (
        "Work environment: "
        f"Has worked at {', '.join(sorted(stages)) or 'unknown'} companies "
        f"with team sizes {', '.join(sorted(sizes)) or 'unknown'}. "
        f"Prefers {_pref_value(prefs.preferred_work_style)}. "
        f"Values {_pref_value(prefs.priorities)}. "
        f"Looking for {_pref_value(prefs.likely_looking_for)}."
    )


def embed_candidate(
    profile: CandidateProfile,
    settings: Settings | None = None,
) -> CandidateEmbeddings:
    """Generate skill, domain, role, and environment embedding vectors."""
    cfg = settings or get_settings()
    encoder = get_encoder(cfg)
    return CandidateEmbeddings(
        skill=encoder.encode(_build_skill_text(profile, cfg)),
        domain=encoder.encode(_build_domain_text(profile)),
        role=encoder.encode(_build_role_text(profile)),
        environment=encoder.encode(_build_environment_text(profile)),
    )
