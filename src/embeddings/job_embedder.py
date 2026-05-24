"""Job embedding facade: extract fields, link skills, embed."""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import Settings, get_settings
from src.db.models import Job
from src.embeddings.encoder import get_encoder
from src.embeddings.job_field_extractor import extract_job_fields_rule
from src.embeddings.job_llm_extractor import extract_job_fields_llm
from src.embeddings.schemas import JobEmbeddings, JobFields, LinkedJobSkill
from src.embeddings.skills_extracted_parser import build_skills_payload
from src.knowledge_graph.entity_linker import link_skill
from src.knowledge_graph.schemas import LinkedSkill
from src.knowledge_graph.skill_expander import expand_skill

logger = logging.getLogger(__name__)

DIMENSIONS = ("skill", "domain", "role", "environment")


def extract_job_fields(
    job: Job,
    *,
    use_llm: bool = False,
    settings: Optional[Settings] = None,
) -> JobFields:
    """Extract structured fields from a job (rule-based or LLM)."""
    if use_llm:
        return extract_job_fields_llm(job, settings=settings)
    return extract_job_fields_rule(job, settings=settings)


def link_job_skills(skill_names: list[str]) -> list[LinkedJobSkill]:
    """Link skill names to ESCO; degrade gracefully per skill."""
    linked: list[LinkedJobSkill] = []
    seen: set[str] = set()
    for name in skill_names:
        key = name.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            result: Optional[LinkedSkill] = link_skill(name)
            if result:
                linked.append(
                    LinkedJobSkill(
                        name=name,
                        esco_uri=result.esco_uri,
                        esco_label=result.esco_label,
                        match_type=result.match_type,
                        confidence=result.confidence,
                    ),
                )
            else:
                linked.append(LinkedJobSkill(name=name))
        except Exception as exc:
            logger.warning("ESCO link failed for skill %r: %s", name, exc)
            linked.append(LinkedJobSkill(name=name))
    return linked


def _expand_skill_labels(linked: list[LinkedJobSkill], settings: Settings) -> str:
    """Build expanded ESCO label suffix for skill embedding text."""
    parts: list[str] = []
    hops = settings.ingestion.skill_expansion_max_hops_embed
    for item in linked[:20]:
        if not item.esco_uri:
            parts.append(item.name)
            continue
        try:
            related = expand_skill(item.esco_uri, max_hops=hops)
            labels = [r.label for r in related[:5]]
            if labels:
                parts.append(f"{item.name} ({', '.join(labels)})")
            else:
                parts.append(item.name)
        except Exception as exc:
            logger.debug("Job skill expansion skipped for %s: %s", item.name, exc)
            parts.append(item.name)
    return ", ".join(parts)


def _build_skill_text(
    fields: JobFields,
    raw_description: str,
    linked: list[LinkedJobSkill],
    settings: Settings,
) -> str:
    """Construct skill-dimension embedding text."""
    required = ", ".join(fields.required_skills)
    preferred = ", ".join(fields.preferred_skills)
    expanded = _expand_skill_labels(linked, settings)
    focus = fields.responsibilities_summary
    if len(fields.required_skills) < 3 and raw_description:
        fallback = raw_description[:1000]
        return (
            f"Required skills: {required}. Preferred skills: {preferred}. "
            f"Technical focus: {focus}. {fallback}"
        )
    return (
        f"Required skills: {required}. Preferred skills: {preferred}. "
        f"Technical focus: {focus}. {expanded}"
    )


def _build_domain_text(fields: JobFields, raw_description: str) -> str:
    """Construct domain-dimension embedding text."""
    domain = fields.domain or fields.industry or "unknown"
    company = fields.company_description or "unknown"
    sector = fields.industry_keywords_from_description
    if not domain and raw_description:
        return raw_description[:1500]
    return f"Industry: {domain}. Company: {company}. Sector focus: {sector}"


def _build_role_text(fields: JobFields) -> str:
    """Construct role-dimension embedding text."""
    return (
        f"Role: {fields.job_title}. Level: {fields.role_level}. Type: {fields.role_type}. "
        f"Responsibilities: {fields.responsibilities_summary}. Team: {fields.team_info}"
    )


def _build_environment_text(fields: JobFields) -> str:
    """Construct environment-dimension embedding text."""
    return (
        f"Company stage: {fields.company_stage or 'unknown'}. "
        f"Company size: {fields.company_size or 'unknown'}. "
        f"Work environment: {fields.work_style_signals or 'unknown'}. "
        f"Remote: {fields.remote_type or 'unknown'}. "
        f"Culture: {fields.company_description[:300] if fields.company_description else 'unknown'}"
    )


def embed_job(
    fields: JobFields,
    raw_description: str,
    linked_skills: list[LinkedJobSkill],
    settings: Optional[Settings] = None,
) -> JobEmbeddings:
    """Generate four embedding vectors for a job."""
    cfg = settings or get_settings()
    encoder = get_encoder(cfg)
    texts = {
        "skill": _build_skill_text(fields, raw_description, linked_skills, cfg),
        "domain": _build_domain_text(fields, raw_description),
        "role": _build_role_text(fields),
        "environment": _build_environment_text(fields),
    }
    vectors = encoder.encode_batch([texts[d] for d in DIMENSIONS])
    return JobEmbeddings(
        skill=vectors[0],
        domain=vectors[1],
        role=vectors[2],
        environment=vectors[3],
    )


def embed_job_record(
    job: Job,
    *,
    use_llm: bool = False,
    settings: Optional[Settings] = None,
) -> tuple[JobEmbeddings, dict]:
    """Full pipeline: extract, link, embed; return embeddings and skills_extracted JSON."""
    cfg = settings or get_settings()
    fields = extract_job_fields(job, use_llm=use_llm, settings=cfg)
    all_names = list(dict.fromkeys(fields.required_skills + fields.preferred_skills))
    linked = link_job_skills(all_names)
    embeddings = embed_job(fields, job.description or "", linked, settings=cfg)
    skills_json = build_skills_payload(linked, extraction_method=fields.extraction_method)
    return embeddings, skills_json


def build_skills_extracted_for_job(
    job: Job,
    *,
    use_llm: bool = False,
    settings: Optional[Settings] = None,
) -> tuple[JobFields, list[LinkedJobSkill], dict]:
    """Extract fields and link skills without embedding."""
    cfg = settings or get_settings()
    fields = extract_job_fields(job, use_llm=use_llm, settings=cfg)
    all_names = list(dict.fromkeys(fields.required_skills + fields.preferred_skills))
    linked = link_job_skills(all_names)
    return fields, linked, build_skills_payload(linked, extraction_method=fields.extraction_method)
