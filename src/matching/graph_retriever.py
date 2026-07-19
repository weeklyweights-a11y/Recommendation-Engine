"""Knowledge-graph-enhanced job retrieval via ESCO skill expansion."""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from src.api.schemas.candidate import CandidateProfile, ESCOLinkedSkill
from src.api.schemas.recommendation import ScoredJob
from src.db.models import Job
from src.embeddings.skills_extracted_parser import parse_skills_extracted
from src.knowledge_graph.entity_linker import link_skills
from src.knowledge_graph.skill_expander import expand_skill
from src.knowledge_graph.schemas import ExpandedSkill
from src.matching.schemas import SkillOverlap

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieve jobs by ESCO skill expansion and reverse index lookup."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize with settings; reverse index built lazily."""
        self._settings = settings or get_settings()
        self._reverse_index: dict[str, set[str]] = {}
        self._job_skill_counts: dict[str, int] = {}
        self._index_built = False

    def rebuild(self, session: Session) -> dict[str, int]:
        """Rebuild reverse index from PostgreSQL skills_extracted."""
        self._reverse_index = {}
        self._job_skill_counts = {}
        jobs_without_skills = 0
        total_links = 0

        jobs = session.scalars(select(Job).where(Job.skills_extracted.isnot(None))).all()
        for job in jobs:
            skills = parse_skills_extracted(job.skills_extracted)
            uris = [s.esco_uri for s in skills if s.esco_uri]
            if not uris:
                jobs_without_skills += 1
                continue
            job_id = str(job.id)
            self._job_skill_counts[job_id] = len(uris)
            for uri in uris:
                self._reverse_index.setdefault(uri, set()).add(job_id)
                total_links += 1

        self._index_built = True
        stats = {
            "skill_entries": len(self._reverse_index),
            "job_skill_links": total_links,
            "jobs_without_skills": jobs_without_skills,
        }
        logger.info("Graph reverse index built: %s", stats)
        return stats

    def _ensure_index(self, session: Session) -> None:
        if not self._index_built:
            self.rebuild(session)

    def _resolve_candidate_skills(
        self,
        candidate_skills: list[ESCOLinkedSkill],
        profile: Optional[CandidateProfile],
    ) -> list[ESCOLinkedSkill]:
        if candidate_skills:
            return candidate_skills
        if profile is None:
            return []
        if profile.esco_linked_skills:
            return profile.esco_linked_skills
        names = [s.name for s in profile.skills if s.name]
        if not names:
            return []
        try:
            linked = link_skills(names)
            result: list[ESCOLinkedSkill] = []
            for name, item in zip(names, linked):
                if item:
                    result.append(
                        ESCOLinkedSkill(
                            original_name=name,
                            esco_uri=item.esco_uri,
                            esco_label=item.esco_label,
                            match_type=item.match_type,
                            confidence=item.confidence,
                        ),
                    )
            return result
        except Exception as exc:
            logger.error("Entity linker fallback failed: %s", exc)
            return []

    def _hop_weight(self, hop_distance: int) -> float:
        r = self._settings.retrieval
        if hop_distance <= 0:
            return r.graph_direct_match_weight
        if hop_distance == 1:
            return r.graph_one_hop_weight
        return r.graph_two_hop_weight

    def _expand_candidate_skills(
        self,
        candidate_skills: list[ESCOLinkedSkill],
    ) -> dict[str, tuple[float, int, str]]:
        """Return uri -> (tier_weight, hop_distance, candidate_skill_name)."""
        max_hops = self._settings.retrieval.graph_expansion_max_hops
        expanded: dict[str, tuple[float, int, str]] = {}
        for skill in candidate_skills:
            if not skill.esco_uri:
                continue
            label = getattr(skill, "original_name", None) or getattr(skill, "name", "")
            expanded[skill.esco_uri] = (
                self._hop_weight(0),
                0,
                label,
            )
            try:
                related: list[ExpandedSkill] = expand_skill(skill.esco_uri, max_hops=max_hops)
            except Exception as exc:
                logger.error("expand_skill failed for %s: %s", skill.esco_uri, exc)
                continue
            for item in related:
                hop = item.hop_distance
                tier_w = self._hop_weight(hop)
                prev = expanded.get(item.uri)
                if prev is None or tier_w > prev[0]:
                    expanded[item.uri] = (tier_w, hop, label)
        return expanded

    def retrieve(
        self,
        candidate_esco_skills: list[ESCOLinkedSkill],
        top_k: Optional[int] = None,
        allowed_job_ids: Optional[set[str]] = None,
        session: Optional[Session] = None,
        profile: Optional[CandidateProfile] = None,
    ) -> list[ScoredJob]:
        """Retrieve jobs ranked by ESCO skill graph overlap."""
        if session is None:
            logger.warning("Graph retrieve called without session; returning empty")
            return []

        skills = self._resolve_candidate_skills(candidate_esco_skills, profile)
        if not skills:
            logger.warning("No ESCO-linked skills for graph retrieval")
            return []

        self._ensure_index(session)
        k = top_k if top_k is not None else self._settings.retrieval.hybrid_top_k
        expanded = self._expand_candidate_skills(skills)

        job_scores: dict[str, float] = {}
        job_matches: dict[str, list[dict[str, Any]]] = {}

        for uri, (weight, hop, cand_name) in expanded.items():
            for job_id in self._reverse_index.get(uri, set()):
                if allowed_job_ids is not None and job_id not in allowed_job_ids:
                    continue
                job_scores[job_id] = job_scores.get(job_id, 0.0) + weight
                job_matches.setdefault(job_id, []).append(
                    {
                        "candidate_skill": cand_name,
                        "job_skill": uri,
                        "match_type": "direct" if hop == 0 else f"{hop}-hop",
                        "weight": weight,
                    },
                )

        if not job_scores:
            return []

        normalized: dict[str, float] = {}
        for job_id, raw_score in job_scores.items():
            total_required = self._job_skill_counts.get(job_id)
            if total_required and total_required > 0:
                normalized[job_id] = raw_score / total_required
            else:
                normalized[job_id] = raw_score

        max_score = max(normalized.values()) or 1.0
        ranked = sorted(
            ((jid, score / max_score) for jid, score in normalized.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        return [
            ScoredJob(
                job_id=job_id,
                score=score,
                source="graph",
                matched_skills=job_matches.get(job_id, []),
            )
            for job_id, score in ranked
        ]

    def get_skill_overlap(
        self,
        candidate_esco_uris: list[str],
        job_esco_uris: list[str],
    ) -> SkillOverlap:
        """Compute skill overlap between candidate and job URI sets."""
        overlap = SkillOverlap()
        max_hops = self._settings.retrieval.graph_expansion_max_hops
        candidate_expanded: dict[str, int] = {}
        for uri in candidate_esco_uris:
            candidate_expanded[uri] = 0
            try:
                for item in expand_skill(uri, max_hops=max_hops):
                    if item.uri not in candidate_expanded:
                        candidate_expanded[item.uri] = item.hop_distance
            except Exception:
                continue

        job_set = set(job_esco_uris)
        for uri, hop in candidate_expanded.items():
            if uri not in job_set:
                continue
            entry = {"uri": uri, "hop": hop}
            if hop == 0:
                overlap.direct_matches.append(entry)
            elif hop == 1:
                overlap.one_hop_matches.append(entry)
            else:
                overlap.two_hop_matches.append(entry)

        matched_job_uris = set(candidate_expanded) & job_set
        overlap.unmatched_job_skills = [u for u in job_esco_uris if u not in matched_job_uris]
        overlap.unmatched_candidate_skills = [
            u for u in candidate_esco_uris if u not in matched_job_uris
        ]
        return overlap
