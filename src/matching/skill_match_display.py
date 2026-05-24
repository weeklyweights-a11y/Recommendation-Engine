"""Build matched / gap skill lists aligned with ESCO overlap and profile skills."""

from __future__ import annotations

import re
from typing import Any, Optional

from src.api.schemas.candidate import CandidateProfile, ProfileSkill
from src.db.models import Job
from src.embeddings.schemas import LinkedJobSkill
from src.embeddings.skills_extracted_parser import parse_skills_extracted
from src.matching.schemas import SkillOverlap

_ALIAS_GROUPS: list[set[str]] = [
    {"machine learning", "ml"},
    {"deep learning", "dl"},
    {"tensorflow", "tensor flow", "tf"},
    {"pytorch", "torch"},
    {"kubernetes", "k8s"},
    {"amazon web services", "aws"},
    {"google cloud platform", "gcp", "google cloud"},
    {"microsoft azure", "azure"},
    {"natural language processing", "nlp"},
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", text.lower()).strip()


def _alias_keys(text: str) -> set[str]:
    norm = _normalize(text)
    keys = {norm} if norm else set()
    for group in _ALIAS_GROUPS:
        if norm in group:
            keys |= group
    return keys


def _profile_skill_keys(skill: ProfileSkill) -> set[str]:
    keys: set[str] = set()
    for value in (skill.name, skill.esco_label):
        if value:
            keys |= _alias_keys(str(value))
    return keys


def _job_skill_label(skill: LinkedJobSkill) -> str:
    return str(skill.name or skill.esco_label or "skill")


def _job_skill_keys(skill: LinkedJobSkill) -> set[str]:
    keys: set[str] = set()
    for value in (skill.name, skill.esco_label):
        if value:
            keys |= _alias_keys(str(value))
    return keys


def _names_overlap(candidate_keys: set[str], job_keys: set[str]) -> bool:
    if not candidate_keys or not job_keys:
        return False
    if candidate_keys & job_keys:
        return True
    for ck in candidate_keys:
        for jk in job_keys:
            if ck == jk or ck in jk or jk in ck:
                return True
    return False


def _collect_candidate_keys(profile: CandidateProfile) -> set[str]:
    keys: set[str] = set()
    for skill in profile.skills:
        keys |= _profile_skill_keys(skill)
    return keys


def _hop_label(hop: int) -> str:
    if hop <= 0:
        return "direct"
    if hop == 1:
        return "1-hop"
    return f"{hop}-hop"


def build_skill_match_display(
    profile: CandidateProfile,
    job: Job,
    overlap: Optional[SkillOverlap] = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Return {"matched": [...], "gaps": [...]} for UI.

    Each matched item: {skill, via?}
    Each gap item: {skill}
    """
    job_skills = parse_skills_extracted(job.skills_extracted)
    candidate_keys = _collect_candidate_keys(profile)

    uri_to_label: dict[str, str] = {}
    for skill in job_skills:
        if skill.esco_uri:
            uri_to_label[skill.esco_uri] = _job_skill_label(skill)

    matched_uris: dict[str, int] = {}
    if overlap:
        for entry in overlap.direct_matches:
            matched_uris[str(entry["uri"])] = int(entry.get("hop", 0))
        for entry in overlap.one_hop_matches:
            matched_uris[str(entry["uri"])] = int(entry.get("hop", 1))
        for entry in overlap.two_hop_matches:
            matched_uris[str(entry["uri"])] = int(entry.get("hop", 2))

    matched: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    seen_matched: set[str] = set()
    seen_gaps: set[str] = set()

    for job_skill in job_skills:
        label = _job_skill_label(job_skill)
        norm_label = _normalize(label)
        job_keys = _job_skill_keys(job_skill)
        uri = job_skill.esco_uri

        via: Optional[str] = None
        if uri and uri in matched_uris:
            via = _hop_label(matched_uris[uri])
        elif _names_overlap(candidate_keys, job_keys):
            via = "profile"
        elif norm_label and norm_label in candidate_keys:
            via = "profile"

        if via:
            if norm_label not in seen_matched:
                seen_matched.add(norm_label)
                item: dict[str, Any] = {"skill": label}
                if via != "direct":
                    item["via"] = via
                matched.append(item)
        else:
            if norm_label not in seen_gaps:
                seen_gaps.add(norm_label)
                gaps.append({"skill": label})

    if not job_skills and overlap:
        for uri, hop in matched_uris.items():
            label = uri_to_label.get(uri, uri.rsplit("/", 1)[-1])
            norm = _normalize(label)
            if norm not in seen_matched:
                seen_matched.add(norm)
                item = {"skill": label}
                if hop > 0:
                    item["via"] = _hop_label(hop)
                matched.append(item)
        for uri in overlap.unmatched_job_skills:
            label = uri_to_label.get(uri, uri.rsplit("/", 1)[-1])
            job_keys = _alias_keys(label)
            if _names_overlap(candidate_keys, job_keys):
                norm = _normalize(label)
                if norm not in seen_matched:
                    seen_matched.add(norm)
                    matched.append({"skill": label, "via": "profile"})
                continue
            norm = _normalize(label)
            if norm not in seen_gaps:
                seen_gaps.add(norm)
                gaps.append({"skill": label})

    return {"matched": matched[:16], "gaps": gaps[:12]}
