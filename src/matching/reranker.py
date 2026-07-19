"""Multi-factor utility reranker for hybrid retrieval results."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from src.api.schemas.candidate import CandidateProfile, MergedPreferences
from src.db.models import Job
from src.embeddings.job_field_extractor import extract_job_fields_rule
from src.embeddings.skills_extracted_parser import parse_skills_extracted
from src.knowledge_graph.entity_linker import link_skills
from src.matching.graph_retriever import GraphRetriever
from src.matching.preference_utils import pref_list
from src.matching.schemas import FusedResult, RankedJob, SkillOverlap
from src.matching.skill_match_display import build_skill_match_display

logger = logging.getLogger(__name__)

FACTOR_KEYS = (
    "skill_fit",
    "experience_alignment",
    "domain_relevance",
    "role_shape_match",
    "location_fit",
    "company_stage_alignment",
    "semantic_similarity",
)

_STAGE_ORDER = ["pre-seed", "seed", "series-a", "series-b", "growth", "enterprise"]


def _normalize_stage(stage: str) -> str:
    return stage.lower().replace("_", "-").strip()


def _load_yaml(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    with file_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class Reranker:
    """Deterministic multi-factor reranking."""

    def __init__(
        self,
        session: Session,
        settings: Optional[Settings] = None,
        graph_retriever: Optional[GraphRetriever] = None,
    ) -> None:
        """Initialize with DB session and config."""
        self._session = session
        self._settings = settings or get_settings()
        self._cfg = self._settings.reranker
        self._graph = graph_retriever or GraphRetriever(self._settings)
        self._domain_map = _load_yaml(self._settings.paths.domain_similarity_path)
        self._role_map = _load_yaml(self._settings.paths.role_compatibility_path)

    def _default_weights(self) -> dict[str, float]:
        c = self._cfg
        return {
            "skill_fit": c.skill_fit_weight,
            "experience_alignment": c.experience_alignment_weight,
            "domain_relevance": c.domain_relevance_weight,
            "role_shape_match": c.role_shape_weight,
            "location_fit": c.location_fit_weight,
            "company_stage_alignment": c.company_stage_weight,
            "semantic_similarity": c.semantic_similarity_weight,
        }

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.get(k, 0.0) for k in FACTOR_KEYS)
        if total <= 0:
            even = 1.0 / len(FACTOR_KEYS)
            return {k: even for k in FACTOR_KEYS}
        return {k: weights.get(k, 0.0) / total for k in FACTOR_KEYS}

    def _job_skill_uris(self, job: Job) -> list[str]:
        skills = parse_skills_extracted(job.skills_extracted)
        return [s.esco_uri for s in skills if s.esco_uri]

    def _candidate_skill_uris(self, profile: CandidateProfile) -> list[str]:
        seen: set[str] = set()
        uris: list[str] = []
        for skill in profile.esco_linked_skills:
            if skill.esco_uri and skill.esco_uri not in seen:
                seen.add(skill.esco_uri)
                uris.append(skill.esco_uri)
        for skill in profile.skills:
            if skill.esco_uri and skill.esco_uri not in seen:
                seen.add(skill.esco_uri)
                uris.append(skill.esco_uri)
        if uris:
            return uris
        missing_names = [s.name for s in profile.skills if s.name and not s.esco_uri]
        if not missing_names:
            return []
        try:
            for name, link in zip(missing_names, link_skills(missing_names)):
                if link and link.esco_uri and link.esco_uri not in seen:
                    seen.add(link.esco_uri)
                    uris.append(link.esco_uri)
        except Exception as exc:
            logger.warning("On-the-fly skill linking failed: %s", exc)
        return uris

    def _compute_skill_fit(
        self,
        profile: CandidateProfile,
        job: Job,
        overlap: Optional[SkillOverlap],
    ) -> float:
        cfg = self._cfg
        if overlap and (
            overlap.direct_matches or overlap.one_hop_matches or overlap.two_hop_matches
        ):
            weighted = (
                len(overlap.direct_matches) * cfg.skill_direct_weight
                + len(overlap.one_hop_matches) * cfg.skill_one_hop_weight
                + len(overlap.two_hop_matches) * cfg.skill_two_hop_weight
            )
            required = len(overlap.unmatched_job_skills) + len(overlap.direct_matches) + len(
                overlap.one_hop_matches,
            )
            if required <= 0:
                required = cfg.skill_default_required_count
            score = min(1.0, weighted / required)
        else:
            cand_names = [s.name.lower() for s in profile.skills if s.name]
            if not cand_names:
                return 0.0
            blob = (job.description or "").lower()
            matched = sum(1 for name in cand_names if name in blob)
            score = min(1.0, matched / len(cand_names))

        top_skills = {s.name.lower() for s in profile.skills if s.depth_score > cfg.skill_depth_threshold}
        if top_skills and overlap:
            matched_names = {
                m.get("uri", "").lower() for m in overlap.direct_matches
            }
            if matched_names:
                score = min(1.0, score * cfg.skill_depth_boost_multiplier)
        return float(max(0.0, min(1.0, score)))

    def _compute_experience_alignment(self, profile: CandidateProfile, job: Job) -> float:
        years = profile.total_years_experience
        level = (job.experience_level or "").lower()
        if not level:
            fields = extract_job_fields_rule(job, self._settings)
            level = fields.role_level
        if level == "entry":
            if years <= 3:
                return 1.0
            if years <= 6:
                return 0.7
            return 0.4
        if level == "mid":
            if 2 <= years <= 6:
                return 1.0
            if years <= 1:
                return 0.5
            return 0.7
        if level == "senior":
            if 4 <= years <= 10:
                return 1.0
            if years <= 3:
                return 0.5
            return 0.7
        if level in ("lead", "executive"):
            if years >= 7:
                return 1.0
            if years >= 4:
                return 0.6
            return 0.3
        return self._cfg.experience_unknown_score

    def _domain_score(self, candidate_domain: str, job_industry: str) -> float:
        cfg_scores = self._domain_map
        related = float(cfg_scores.get("related_score", 0.7))
        unrelated = float(cfg_scores.get("unrelated_score", 0.3))
        unknown = float(cfg_scores.get("unknown_score", 0.5))
        if not job_industry:
            return unknown
        if not candidate_domain:
            return unknown
        cd = candidate_domain.lower()
        ji = job_industry.lower()
        if cd == ji or cd in ji or ji in cd:
            return 1.0
        clusters: dict[str, list[str]] = cfg_scores.get("clusters") or {}
        for members in clusters.values():
            norm = [m.lower() for m in members]
            if cd in norm and ji in norm:
                return 1.0
            if cd in norm and any(ji in m or m in ji for m in norm):
                return related
            if ji in norm and any(cd in m or m in cd for m in norm):
                return related
        return unrelated

    def _compute_domain_relevance(self, profile: CandidateProfile, job: Job) -> float:
        if not profile.domains:
            return float(self._domain_map.get("unknown_score", 0.5))
        job_industry = job.industry or ""
        return max(self._domain_score(d, job_industry) for d in profile.domains)

    def _compute_role_shape_match(self, profile: CandidateProfile, job: Job) -> float:
        fields = extract_job_fields_rule(job, self._settings)
        archetype = (profile.role_archetype or "generalist").lower()
        job_role = fields.role_type.lower()
        compat: dict[str, Any] = self._role_map.get("compatibility") or {}
        archetype_row = compat.get(archetype) or compat.get("generalist") or {}
        score = float(archetype_row.get(job_role, self._role_map.get("default_score", 0.5)))
        if profile.career_trajectory == "career_changer":
            floor = float(
                self._role_map.get(
                    "career_changer_floor",
                    self._cfg.career_changer_role_shape_floor,
                ),
            )
            score = max(score, floor)
        return float(max(0.0, min(1.0, score)))

    def _compute_location_fit(
        self,
        profile: CandidateProfile,
        job: Job,
        preferences: MergedPreferences,
    ) -> float:
        remote = (job.remote_type or "").lower()
        locations = pref_list(preferences.locations)
        work_models = pref_list(preferences.work_models)

        if remote == "remote" and ("remote" in work_models or not work_models):
            return 1.0
        if locations and job.location:
            loc_lower = job.location.lower()
            if any(loc in loc_lower for loc in locations):
                return 1.0
        if remote == "hybrid":
            if "hybrid" in work_models and locations and job.location:
                if any(loc in job.location.lower() for loc in locations):
                    return 0.9
            return 0.5
        if remote == "onsite" and work_models == ["remote"]:
            return 0.0
        if not job.location:
            return 0.6
        if not locations:
            return 0.8
        return 0.5

    def _stage_distance(self, a: str, b: str) -> int:
        na, nb = _normalize_stage(a), _normalize_stage(b)
        if na not in _STAGE_ORDER or nb not in _STAGE_ORDER:
            return 99
        return abs(_STAGE_ORDER.index(na) - _STAGE_ORDER.index(nb))

    def _compute_company_stage_alignment(
        self,
        preferences: MergedPreferences,
        job: Job,
    ) -> float:
        preferred = pref_list(preferences.company_stages)
        job_stage = job.company_stage or ""
        if not preferred:
            return self._cfg.stage_no_preference_score
        if not job_stage:
            return self._cfg.stage_unknown_score
        norm_pref = {_normalize_stage(s) for s in preferred}
        norm_job = _normalize_stage(job_stage)
        if norm_job in norm_pref:
            return 1.0
        dist = min(self._stage_distance(norm_job, p) for p in norm_pref)
        if dist == 1:
            return self._cfg.stage_adjacent_score
        if dist == 2:
            return self._cfg.stage_two_step_score
        return self._cfg.stage_far_score

    def _freshness_boost(self, job: Job, utility: float) -> float:
        if not job.posted_date:
            return utility
        now = datetime.now(timezone.utc)
        posted = job.posted_date
        if posted.tzinfo is None:
            posted = posted.replace(tzinfo=timezone.utc)
        age = now - posted
        if age <= timedelta(hours=48):
            return min(1.0, utility + self._cfg.freshness_48h_boost)
        if age <= timedelta(days=7):
            return min(1.0, utility + self._cfg.freshness_7d_boost)
        return utility

    def _inject_diversity(self, ranked: list[RankedJob]) -> list[RankedJob]:
        cfg = self._cfg
        top = ranked[: cfg.diversity_top_n]
        if len(top) < cfg.diversity_top_n:
            return ranked
        industries = {
            (getattr(r.job, "industry", None) or "unknown").lower() for r in top
        }
        stages = {
            (getattr(r.job, "company_stage", None) or "unknown").lower() for r in top
        }
        if (
            len(industries) >= cfg.diversity_min_industries
            and len(stages) >= cfg.diversity_min_stages
        ):
            return ranked

        pool = ranked[cfg.diversity_top_n : cfg.diversity_scan_until]
        inject: list[RankedJob] = []
        for candidate in pool:
            ind = (getattr(candidate.job, "industry", None) or "unknown").lower()
            st = (getattr(candidate.job, "company_stage", None) or "unknown").lower()
            if ind not in industries or st not in stages:
                inject.append(candidate)
                industries.add(ind)
                stages.add(st)
            if len(inject) >= cfg.diversity_inject_end - cfg.diversity_inject_start + 1:
                break

        if not inject:
            return ranked

        head = ranked[: cfg.diversity_keep_top]
        tail = ranked[cfg.diversity_inject_end :]
        merged = head + inject[: cfg.diversity_inject_end - cfg.diversity_inject_start] + tail
        for i, item in enumerate(merged, start=1):
            item.rank = i
        logger.info("Diversity injection swapped %s jobs into ranks 15-20", len(inject))
        return merged

    def _label_worth_exploring(self, ranked: list[RankedJob]) -> None:
        if not ranked:
            return
        cutoff = max(1, int(len(ranked) * (1.0 - self._cfg.worth_exploring_percentile)))
        for item in ranked[cutoff:]:
            sem = item.factor_scores.get("semantic_similarity", 0.0)
            weak = any(
                item.factor_scores.get(k, 1.0) < self._cfg.worth_exploring_factor_max
                for k in FACTOR_KEYS
                if k != "semantic_similarity"
            )
            if sem > self._cfg.worth_exploring_semantic_min and weak:
                item.feed_section = "worth_exploring"

    def rerank(
        self,
        profile: CandidateProfile,
        fused_results: list[FusedResult],
        top_k: int = 50,
        custom_weights: Optional[dict[str, float]] = None,
    ) -> list[RankedJob]:
        """Rerank fused jobs by weighted utility score."""
        if not fused_results:
            return []

        weights = self._normalize_weights(custom_weights or self._default_weights())
        job_ids = [UUID(r.job_id) for r in fused_results]
        jobs = {
            str(j.id): j
            for j in self._session.scalars(select(Job).where(Job.id.in_(job_ids)))
        }
        cand_uris = self._candidate_skill_uris(profile)
        preferences = profile.preferences

        scored: list[RankedJob] = []
        for fused in fused_results:
            job = jobs.get(fused.job_id)
            if job is None:
                continue
            overlap: Optional[SkillOverlap] = None
            job_uris = self._job_skill_uris(job)
            if cand_uris and job_uris:
                overlap = self._graph.get_skill_overlap(cand_uris, job_uris)

            factors = {
                "skill_fit": self._compute_skill_fit(profile, job, overlap),
                "experience_alignment": self._compute_experience_alignment(profile, job),
                "domain_relevance": self._compute_domain_relevance(profile, job),
                "role_shape_match": self._compute_role_shape_match(profile, job),
                "location_fit": self._compute_location_fit(profile, job, preferences),
                "company_stage_alignment": self._compute_company_stage_alignment(
                    preferences,
                    job,
                ),
                "semantic_similarity": float(fused.vector_score),
            }
            utility = sum(weights[k] * factors[k] for k in FACTOR_KEYS)
            utility = self._freshness_boost(job, utility)
            skill_display = build_skill_match_display(profile, job, overlap)
            scored.append(
                RankedJob(
                    job_id=fused.job_id,
                    job=job,
                    rank=0,
                    match_score=utility,
                    match_percentage=int(round(utility * 100)),
                    factor_scores=factors,
                    retrieval_scores={
                        "bm25": fused.bm25_score,
                        "vector": fused.vector_score,
                        "graph": fused.graph_score,
                        "fused": fused.fused_score,
                    },
                    vector_dimension_scores=fused.vector_dimension_scores,
                    graph_matched_skills=fused.graph_matched_skills,
                    skill_match_display=skill_display,
                ),
            )

        scored.sort(key=lambda r: r.match_score, reverse=True)
        scored = scored[:top_k]
        scored = self._inject_diversity(scored)
        self._label_worth_exploring(scored)
        for i, item in enumerate(scored, start=1):
            item.rank = i
        return scored
