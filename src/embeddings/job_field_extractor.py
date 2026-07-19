"""Rule-based job field extraction (Approach B)."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from config.settings import Settings, get_settings
from src.db.models import Job
from src.embeddings.schemas import JobFields

logger = logging.getLogger(__name__)

_LEVEL_PATTERNS = [
    (re.compile(r"\b(intern|entry[\s-]?level|junior)\b", re.I), "entry"),
    (re.compile(r"\b(senior|sr\.?)\b", re.I), "senior"),
    (re.compile(r"\b(lead|staff|principal)\b", re.I), "lead"),
    (re.compile(r"\b(director|vp|vice president|head of)\b", re.I), "executive"),
    (re.compile(r"\b(manager|engineering manager)\b", re.I), "manager"),
]

_DOMAIN_KEYWORDS = {
    "healthcare": ["healthcare", "medical", "hospital", "clinical", "pharma"],
    "fintech": ["fintech", "financial", "banking", "payments", "trading"],
    "saas": ["saas", "b2b software", "enterprise software"],
    "ecommerce": ["ecommerce", "e-commerce", "retail", "marketplace"],
    "ai": ["machine learning", "artificial intelligence", "deep learning", "nlp"],
}

_TECH_SKILLS = [
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "Go",
    "Rust",
    "C++",
    "SQL",
    "PostgreSQL",
    "MongoDB",
    "Redis",
    "Docker",
    "Kubernetes",
    "AWS",
    "GCP",
    "Azure",
    "PyTorch",
    "TensorFlow",
    "scikit-learn",
    "Spark",
    "Kafka",
    "React",
    "Node.js",
    "FastAPI",
    "Django",
    "Flask",
    "Git",
    "Linux",
    "Terraform",
    "LangChain",
    "LLM",
    "RAG",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Computer Vision",
    "MLOps",
    "CI/CD",
    "Agile",
]


@lru_cache(maxsize=1)
def _load_extra_labels(path: str) -> frozenset[str]:
    """Load supplemental alias labels for dictionary scan."""
    labels: set[str] = set(_TECH_SKILLS)
    file_path = Path(path)
    if file_path.exists():
        try:
            import csv

            with file_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    for key in ("alias", "label", "name"):
                        if row.get(key):
                            labels.add(str(row[key]).strip())
        except Exception as exc:
            logger.warning("Could not load skill dict from %s: %s", path, exc)
    return frozenset(labels)


def _infer_role_level(title: str, description: str) -> str:
    """Infer role level from title and description keywords."""
    blob = f"{title} {description[:500]}"
    for pattern, level in _LEVEL_PATTERNS:
        if pattern.search(blob):
            return level
    return "mid"


def _infer_role_type(title: str, description: str) -> str:
    """Infer role type from title keywords."""
    blob = f"{title} {description[:500]}".lower()
    if "founding" in blob or "founder" in blob:
        return "founding"
    if "manager" in blob or "director" in blob:
        return "manager"
    if "tech lead" in blob or "tech-lead" in blob or "lead" in title.lower():
        return "tech-lead"
    if any(x in blob for x in ("vp", "vice president", "cto", "ceo")):
        return "executive"
    return "ic"


def _infer_domain(company: str, description: str) -> str:
    """Infer industry domain from company and description."""
    blob = f"{company} {description[:2000]}".lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in blob for kw in keywords):
            return domain
    return ""


def _extract_industry_keywords(description: str) -> str:
    """Keyword scan for domain embedding sector focus."""
    blob = description.lower()
    found: list[str] = []
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in blob for kw in keywords):
            found.append(domain)
    for skill in _TECH_SKILLS:
        if skill.lower() in blob:
            found.append(skill)
    return ", ".join(list(dict.fromkeys(found))[:15])


def _scan_skills(description: str, settings: Settings) -> tuple[list[str], list[str]]:
    """Scan description for known skill labels."""
    labels = _load_extra_labels(settings.job_embedding.job_skill_dict_path)
    required: list[str] = []
    preferred: list[str] = []
    pref_section = re.search(
        r"(nice to have|preferred|bonus|desired)[\s\S]{0,800}",
        description,
        re.I,
    )
    pref_blob = pref_section.group(0).lower() if pref_section else ""
    for label in labels:
        if len(label) < 2:
            continue
        pattern = re.compile(rf"\b{re.escape(label)}\b", re.I)
        if pattern.search(description):
            if pattern.search(pref_blob):
                preferred.append(label)
            else:
                required.append(label)
    return required, preferred


def _extract_company_section(description: str) -> str:
    """Extract 'About us' style company blurb."""
    match = re.search(
        r"(about (?:us|the company|our company)|who we are)[\s:]*([\s\S]{0,600})",
        description,
        re.I,
    )
    if match:
        return match.group(2).strip()[:500]
    return ""


def _extract_work_style(description: str) -> str:
    """Extract culture and pace signals."""
    signals = []
    for phrase in (
        "fast-paced",
        "move quickly",
        "collaborative",
        "research-driven",
        "startup",
        "remote-first",
        "agile",
        "cross-functional",
    ):
        if phrase in description.lower():
            signals.append(phrase)
    return ", ".join(signals)


def extract_job_fields_rule(job: Job, settings: Optional[Settings] = None) -> JobFields:
    """Extract JobFields using rule-based heuristics."""
    cfg = settings or get_settings()
    description = job.description or ""
    title = job.title or ""
    company = job.company or ""
    required, preferred = _scan_skills(description, cfg)
    company_desc = _extract_company_section(description)
    return JobFields(
        required_skills=required,
        preferred_skills=preferred,
        domain=_infer_domain(company, description) or (job.industry or ""),
        role_level=_infer_role_level(title, description),
        role_type=_infer_role_type(title, description),
        responsibilities_summary=description[:500].strip(),
        company_description=company_desc,
        team_info="",
        work_style_signals=_extract_work_style(description),
        industry_keywords_from_description=_extract_industry_keywords(description),
        job_title=title,
        industry=job.industry or "",
        company_stage=job.company_stage or "",
        company_size=job.company_size or "",
        remote_type=job.remote_type or "",
        extraction_method="rule",
    )
