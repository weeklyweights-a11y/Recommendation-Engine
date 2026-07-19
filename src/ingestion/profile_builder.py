"""Unified profile builder merging resume, GitHub, and preferences."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from src.embeddings.schemas import CandidateEmbeddings

from config.settings import Settings, get_settings
from src.api.schemas.candidate import (
    CandidatePreferences,
    CandidateProfile,
    ESCOLinkedSkill,
    GitHubSummary,
    MergedPreferences,
    PreferenceField,
    ProfileEducation,
    ProfileExperience,
    ProfileSkill,
)
from src.db.candidate_repository import upsert_candidate_profile
from src.ingestion.exceptions import ExtractionFailedError, GitHubUserNotFoundError
from src.ingestion.fallback_extractor import extract_fallback_profile
from src.ingestion.github_fetcher import fetch_github_profile, format_github_for_llm
from src.ingestion.llm_extractor import extract_profile
from src.ingestion.skill_filters import is_plausible_skill
from src.ingestion.skill_supplement import enrich_extracted_skills
from src.ingestion.resume_parser import parse_resume, validate_resume_file
from src.ingestion.schemas import ExtractedProfile, ExtractedSkill, GitHubProfile
from src.knowledge_graph.entity_linker import link_skills

logger = logging.getLogger(__name__)

_PRESENT_END = "9999-99"


def _normalize_skill(name: str) -> str:
    return name.strip().lower()


def _parse_end_date(end_date: Optional[str]) -> str:
    if not end_date:
        return "0000-00"
    if end_date.lower() in {"present", "current", "now"}:
        return _PRESENT_END
    return end_date


def _parse_year_month(value: Optional[str]) -> Optional[tuple[int, int]]:
    """Parse YYYY-MM or YYYY from a date string."""
    if not value:
        return None
    cleaned = value.strip().lower()
    if cleaned in {"present", "current", "now"}:
        now = datetime.now(timezone.utc)
        return now.year, now.month
    match = re.match(r"^(\d{4})(?:-(\d{1,2}))?", cleaned)
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2) or 1)
    return year, min(max(month, 1), 12)


def _estimate_total_years(experience: list[ProfileExperience]) -> float:
    """Estimate total years of experience from job date ranges."""
    if not experience:
        return 0.0
    month_total = 0
    for job in experience:
        if job.duration_months and job.duration_months > 0:
            month_total += job.duration_months
            continue
        start = _parse_year_month(job.start_date)
        end = _parse_year_month(job.end_date) or _parse_year_month("present")
        if not start or not end:
            continue
        months = (end[0] - start[0]) * 12 + (end[1] - start[1])
        month_total += max(months, 1)
    return round(month_total / 12.0, 1)


def _sort_experience(experience: list[ProfileExperience]) -> list[ProfileExperience]:
    return sorted(experience, key=lambda item: _parse_end_date(item.end_date), reverse=True)


def _github_context(github: Optional[GitHubProfile]) -> dict[str, Any]:
    if github is None:
        return {
            "inferred": set(),
            "languages": set(),
            "recent_languages": set(),
            "production_languages": set(),
            "language_repo_counts": {},
        }

    inferred = {_normalize_skill(s) for s in github.inferred_skills}
    languages = {_normalize_skill(k) for k in github.languages_distribution}
    inferred.update(languages)

    recent_languages: set[str] = set()
    production_languages: set[str] = set()
    language_repo_counts: dict[str, int] = {}

    for repo in github.top_repos:
        repo_langs = {_normalize_skill(lang) for lang in repo.languages}
        for lang in repo_langs:
            language_repo_counts[lang] = language_repo_counts.get(lang, 0) + 1
            inferred.add(lang)

        if repo.last_active and any(token in repo.last_active for token in ("day", "week")):
            recent_languages.update(repo_langs)
        elif repo.last_active and "month" in repo.last_active:
            try:
                months = int(repo.last_active.split()[0])
                if months <= 6:
                    recent_languages.update(repo_langs)
            except ValueError:
                pass

        if repo.production_signals:
            production_languages.update(repo_langs)

    return {
        "inferred": inferred,
        "languages": languages,
        "recent_languages": recent_languages,
        "production_languages": production_languages,
        "language_repo_counts": language_repo_counts,
    }


def _skill_in_github(skill_name: str, ctx: dict[str, Any]) -> bool:
    key = _normalize_skill(skill_name)
    if key in ctx["inferred"]:
        return True
    return any(key in lang or lang in key for lang in ctx["inferred"])


def _compute_depth_score(
    skill: ExtractedSkill,
    on_resume: bool,
    github_ctx: dict[str, Any],
    settings: Settings,
) -> tuple[float, list[str]]:
    """Return depth score capped at 1.0 and source tags."""
    ing = settings.ingestion
    sources: list[str] = []
    score = 0.0

    if on_resume:
        sources.append("resume")
        score = ing.skill_depth_base_resume_only
        score += ing.skill_depth_resume_mention
        if skill.proficiency in {"advanced", "expert"}:
            score += ing.skill_depth_resume_proficiency
    else:
        sources.append("github")
        score = ing.skill_depth_base_github_only

    if _skill_in_github(skill.name, github_ctx):
        if "github" not in sources:
            sources.append("github")
        score += ing.skill_depth_github_presence
        key = _normalize_skill(skill.name)
        if key in github_ctx["recent_languages"] or any(
            key in lang or lang in key for lang in github_ctx["recent_languages"]
        ):
            score += ing.skill_depth_github_recency
        if key in github_ctx["production_languages"] or any(
            key in lang or lang in key for lang in github_ctx["production_languages"]
        ):
            score += ing.skill_depth_github_production
        repo_count = github_ctx["language_repo_counts"].get(key, 0)
        if repo_count >= 3:
            score += ing.skill_depth_github_volume

    return min(score, 1.0), sources


def _build_github_summary(github: GitHubProfile) -> GitHubSummary:
    langs = sorted(
        github.languages_distribution.keys(),
        key=lambda k: github.languages_distribution[k],
        reverse=True,
    )
    return GitHubSummary(
        username=github.username,
        overall_assessment=github.overall_assessment,
        top_languages=langs[:5],
        activity_level=github.overall_assessment,
        total_repos=github.activity_metrics.total_repos,
        repos_last_6_months=github.activity_metrics.repos_last_6_months,
        followers=github.followers,
        inferred_skills=list(github.inferred_skills),
        top_repo_names=[repo.name for repo in github.top_repos[:5]],
    )


def _merge_preferences(
    extracted: ExtractedProfile,
    explicit: Optional[CandidatePreferences],
) -> MergedPreferences:
    """Merge inferred and explicit preferences; explicit wins when set."""
    inferred = extracted.inferred_preferences
    merged = MergedPreferences(
        preferred_company_stage=PreferenceField(
            value=inferred.preferred_company_stage,
            source="inferred" if inferred.preferred_company_stage else "default",
        ),
        preferred_team_size=PreferenceField(
            value=inferred.preferred_team_size,
            source="inferred" if inferred.preferred_team_size else "default",
        ),
        preferred_work_style=PreferenceField(
            value=inferred.preferred_work_style,
            source="inferred" if inferred.preferred_work_style else "default",
        ),
        likely_looking_for=PreferenceField(
            value=inferred.likely_looking_for,
            source="inferred" if inferred.likely_looking_for else "default",
        ),
    )

    if explicit is None:
        return merged

    def _apply_list(
        field_name: str,
        explicit_values: list[str],
        target_attr: str,
    ) -> None:
        if explicit_values:
            setattr(
                merged,
                target_attr,
                PreferenceField(value=explicit_values, source="explicit"),
            )

    _apply_list("job_types", explicit.job_types, "job_types")
    _apply_list("work_models", explicit.work_models, "work_models")
    _apply_list("locations", explicit.locations, "locations")
    _apply_list("company_stages", explicit.company_stages, "company_stages")
    _apply_list("company_sizes", explicit.company_sizes, "company_sizes")
    _apply_list("target_roles", explicit.target_roles, "target_roles")
    _apply_list("industries_target", explicit.industries_target, "target_industries")
    _apply_list("industries_avoid", explicit.industries_avoid, "avoid_industries")
    _apply_list("priority_ranking", explicit.priority_ranking, "priorities")

    if explicit.visa_sponsorship_needed is not None:
        merged.needs_sponsorship = PreferenceField(
            value=explicit.visa_sponsorship_needed,
            source="explicit",
        )
    if explicit.salary_min is not None:
        merged.salary_min = PreferenceField(value=explicit.salary_min, source="explicit")
    if explicit.salary_max is not None:
        merged.salary_max = PreferenceField(value=explicit.salary_max, source="explicit")

    return merged


def _collect_skill_names(
    extracted: ExtractedProfile,
    github: Optional[GitHubProfile],
) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for skill in extracted.skills:
        key = _normalize_skill(skill.name)
        if key not in seen:
            seen.add(key)
            names.append(skill.name)
    if github:
        for skill_name in github.inferred_skills:
            key = _normalize_skill(skill_name)
            if key not in seen:
                seen.add(key)
                names.append(skill_name)
    return names


def _build_profile_skills(
    extracted: ExtractedProfile,
    github: Optional[GitHubProfile],
    settings: Settings,
) -> tuple[list[ProfileSkill], list[ESCOLinkedSkill]]:
    github_ctx = _github_context(github)
    skill_names = _collect_skill_names(extracted, github)

    linked_results: list[Any] = []
    try:
        linked_results = link_skills(skill_names)
    except Exception as exc:
        logger.warning("ESCO linking failed; continuing without links: %s", exc)
        linked_results = [None] * len(skill_names)

    link_by_name = {_normalize_skill(name): link for name, link in zip(skill_names, linked_results)}

    esco_linked: list[ESCOLinkedSkill] = []
    profile_skills: list[ProfileSkill] = []
    built: set[str] = set()

    for skill in extracted.skills:
        key = _normalize_skill(skill.name)
        built.add(key)
        link = link_by_name.get(key)
        depth, sources = _compute_depth_score(skill, on_resume=True, github_ctx=github_ctx, settings=settings)
        if link:
            esco_linked.append(
                ESCOLinkedSkill(
                    original_name=skill.name,
                    esco_uri=link.esco_uri,
                    esco_label=link.esco_label,
                    match_type=link.match_type,
                    confidence=link.confidence,
                ),
            )
        profile_skills.append(
            ProfileSkill(
                name=skill.name,
                category=skill.category,
                proficiency=skill.proficiency,
                depth_score=depth,
                years_used=skill.years_used,
                context=skill.context,
                sources=sources,
                esco_uri=link.esco_uri if link else None,
                esco_label=link.esco_label if link else None,
                esco_match_type=link.match_type if link else None,
                esco_match_confidence=link.confidence if link else 0.0,
            ),
        )

    for skill_name in skill_names:
        key = _normalize_skill(skill_name)
        if key in built:
            continue
        built.add(key)
        link = link_by_name.get(key)
        placeholder = ExtractedSkill(name=skill_name, category="other", proficiency="intermediate")
        depth, sources = _compute_depth_score(
            placeholder,
            on_resume=False,
            github_ctx=github_ctx,
            settings=settings,
        )
        if link:
            esco_linked.append(
                ESCOLinkedSkill(
                    original_name=skill_name,
                    esco_uri=link.esco_uri,
                    esco_label=link.esco_label,
                    match_type=link.match_type,
                    confidence=link.confidence,
                ),
            )
        profile_skills.append(
            ProfileSkill(
                name=skill_name,
                category="other",
                proficiency="intermediate",
                depth_score=depth,
                sources=sources,
                esco_uri=link.esco_uri if link else None,
                esco_label=link.esco_label if link else None,
                esco_match_type=link.match_type if link else None,
                esco_match_confidence=link.confidence if link else 0.0,
            ),
        )

    profile_skills = [s for s in profile_skills if is_plausible_skill(s.name)]
    profile_skills.sort(key=lambda s: s.depth_score, reverse=True)
    return profile_skills, esco_linked


async def _assemble_profile(
    resume_file_path: str,
    github_username: str | None = None,
    preferences: CandidatePreferences | None = None,
    settings: Settings | None = None,
) -> tuple[CandidateProfile, str, Optional[GitHubProfile], str]:
    """Run steps 1–7 and return profile plus persistence metadata."""
    cfg = settings or get_settings()
    path = Path(resume_file_path)

    validate_resume_file(str(path), settings=cfg)
    resume_text = parse_resume(str(path), settings=cfg)

    github_profile: Optional[GitHubProfile] = None
    github_summary_text: str | None = None
    if github_username:
        try:
            github_profile = await fetch_github_profile(github_username, settings=cfg)
            github_summary_text = format_github_for_llm(github_profile, settings=cfg)
        except GitHubUserNotFoundError as exc:
            logger.warning("GitHub user not found, continuing without GitHub: %s", exc)
        except Exception as exc:
            logger.warning("GitHub fetch failed, continuing without GitHub: %s", exc)

    try:
        extracted = extract_profile(resume_text, github_summary=github_summary_text, settings=cfg)
    except ExtractionFailedError as exc:
        logger.warning("LLM extraction failed, using fallback: %s", exc)
        extracted = extract_fallback_profile(resume_text)

    extracted = enrich_extracted_skills(extracted, resume_text, github_profile, settings=cfg)
    logger.info("Profile skills after enrichment: %s", len(extracted.skills))

    skills, esco_linked = _build_profile_skills(extracted, github_profile, cfg)
    experience = _sort_experience(
        [ProfileExperience.model_validate(item.model_dump()) for item in extracted.experience],
    )
    education = [ProfileEducation.model_validate(item.model_dump()) for item in extracted.education]
    total_years = extracted.total_years_experience
    if total_years <= 0 and experience:
        total_years = _estimate_total_years(experience)

    profile = CandidateProfile(
        name=extracted.name,
        email=extracted.email,
        phone=extracted.phone,
        location=extracted.location,
        skills=skills,
        experience=experience,
        education=education,
        total_years_experience=total_years,
        domains=sorted(set(extracted.domains)),
        role_archetype=extracted.role_archetype,
        career_trajectory=extracted.career_trajectory,
        github_summary=_build_github_summary(github_profile) if github_profile else None,
        preferences=_merge_preferences(extracted, preferences),
        esco_linked_skills=esco_linked,
        summary=extracted.summary,
    )
    return profile, resume_text, github_profile, path.name


async def build_profile(
    resume_file_path: str,
    github_username: str | None = None,
    preferences: CandidatePreferences | None = None,
    settings: Settings | None = None,
) -> tuple[CandidateProfile, "CandidateEmbeddings"]:
    """Build profile and four semantic embedding vectors."""
    from src.embeddings.candidate_embedder import embed_candidate

    profile, _, _, _ = await _assemble_profile(
        resume_file_path,
        github_username=github_username,
        preferences=preferences,
        settings=settings,
    )
    embeddings = embed_candidate(profile, settings=settings)
    return profile, embeddings


async def build_and_save_profile(
    resume_file_path: str,
    github_username: str | None = None,
    preferences: CandidatePreferences | None = None,
    session: Session | None = None,
    settings: Settings | None = None,
) -> tuple[CandidateProfile, "CandidateEmbeddings"]:
    """Build profile, embed, and persist to PostgreSQL."""
    from src.embeddings.candidate_embedder import embed_candidate

    profile, resume_text, github_profile, resume_filename = await _assemble_profile(
        resume_file_path,
        github_username=github_username,
        preferences=preferences,
        settings=settings,
    )
    embeddings = embed_candidate(profile, settings=settings)
    github_data = github_profile.model_dump(mode="json") if github_profile else None
    if session is not None:
        upsert_candidate_profile(
            session,
            profile,
            resume_text=resume_text,
            resume_filename=resume_filename,
            github_username=github_username,
            github_data=github_data,
            embeddings=embeddings,
        )
    return profile, embeddings
