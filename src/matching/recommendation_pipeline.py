"""Full recommendation pipeline: filter, retrieve, rerank, explain, persist."""

from __future__ import annotations

import json
import logging
import time
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from src.api.schemas.candidate import CandidateProfile
from src.db.candidate_repository import load_candidate_embeddings_vectors
from src.db.models import Candidate, Recommendation
from src.db.recommendation_repository import (
    bulk_insert_recommendations,
    delete_recommendations_for_candidate,
    load_cached_recommendations,
)
from src.db.sync_database import get_sync_session
from src.embeddings.schemas import CandidateEmbeddings
from src.matching.explainer import Explainer
from src.matching.hard_filters import HardFilter
from src.matching.hybrid_pipeline import retrieve_hybrid_parallel
from src.matching.reranker import FACTOR_KEYS, Reranker
from src.matching.schemas import PipelineResult, PipelineStats, PipelineTiming, RankedJob

logger = logging.getLogger(__name__)


def _materialize_ranked_jobs(ranked: list[RankedJob]) -> list[RankedJob]:
    """Convert attached ORM Job rows to JobResponse while the session is active."""
    from src.api.schemas.job import JobResponse
    from src.db.models import Job

    for item in ranked:
        if isinstance(item.job, Job):
            item.job = JobResponse.model_validate(item.job)
    return ranked


def _normalize_utility_weights(raw: Optional[dict]) -> Optional[dict[str, float]]:
    if not raw:
        return None
    cleaned = {k: float(raw[k]) for k in FACTOR_KEYS if k in raw}
    if not cleaned:
        return None
    total = sum(cleaned.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in cleaned.items()}


def run_recommendation_pipeline(
    candidate_id: UUID,
    *,
    refresh: bool = False,
    session: Optional[Session] = None,
    settings: Optional[Settings] = None,
) -> PipelineResult:
    """Execute the end-to-end recommendation pipeline."""
    cfg = settings or get_settings()

    def _run(sess: Session) -> PipelineResult:
        if not refresh:
            cached = load_cached_recommendations(sess, candidate_id, settings=cfg)
            if cached:
                ranked = _materialize_ranked_jobs(_ranked_from_stored(cached))
                return PipelineResult(ranked_jobs=ranked, from_cache=True)

        candidate = sess.get(Candidate, candidate_id)
        if not candidate or not candidate.profile:
            raise ValueError("Candidate profile not found")

        profile = CandidateProfile.model_validate(candidate.profile)
        stats = PipelineStats()
        timing = PipelineTiming()
        t0 = time.perf_counter()

        hf = HardFilter(sess, cfg)
        funnel = hf.get_filter_funnel(profile.preferences)
        stats.filter_funnel = funnel
        allowed = hf.filter_jobs(profile.preferences)
        t1 = time.perf_counter()
        timing.hard_filter_ms = (t1 - t0) * 1000

        if funnel.final_count < cfg.hard_filter.min_results_warn:
            stats.warnings.append("preferences_too_restrictive")
        if funnel.final_count < cfg.hard_filter.pipeline_min_warn:
            logger.warning(
                "Only %s jobs pass hard filters for candidate %s",
                funnel.final_count,
                candidate_id,
            )

        if not allowed:
            timing.total_ms = (time.perf_counter() - t0) * 1000
            stats.timing_ms = timing
            return PipelineResult(ranked_jobs=[], stats=stats)

        fused, overlap = retrieve_hybrid_parallel(
            candidate_id=candidate_id,
            profile=profile,
            embeddings=_load_embeddings(candidate),
            allowed_job_ids=allowed,
            session=sess,
            settings=cfg,
        )
        t2 = time.perf_counter()
        timing.retrieval_ms = (t2 - t1) * 1000
        stats.retrieval_overlap = overlap

        weights = _normalize_utility_weights(candidate.utility_weights)
        reranker = Reranker(sess, cfg)
        ranked = reranker.rerank(
            profile,
            fused,
            top_k=cfg.recommendation.rerank_top_k,
            custom_weights=weights,
        )
        t3 = time.perf_counter()
        timing.rerank_ms = (t3 - t2) * 1000

        explain_top = cfg.recommendation.explain_top_k
        explainer = Explainer(cfg)
        explanations = explainer.explain_batch(profile, ranked[:explain_top], max_jobs=explain_top)
        for job_row, explanation in zip(ranked[:explain_top], explanations):
            job_row.explanation = explanation.model_dump_json()

        t4 = time.perf_counter()
        timing.explain_ms = (t4 - t3) * 1000

        store_k = min(cfg.recommendation.store_top_k, len(ranked))
        _persist_recommendations(sess, candidate_id, ranked[:store_k])

        timing.total_ms = (time.perf_counter() - t0) * 1000
        stats.timing_ms = timing
        logger.info(
            "Pipeline timing hard_filter=%.0fms retrieval=%.0fms rerank=%.0fms explain=%.0fms total=%.0fms",
            timing.hard_filter_ms,
            timing.retrieval_ms,
            timing.rerank_ms,
            timing.explain_ms,
            timing.total_ms,
        )
        return PipelineResult(
            ranked_jobs=_materialize_ranked_jobs(ranked[:store_k]),
            stats=stats,
        )

    if session is not None:
        return _run(session)
    with get_sync_session() as sess:
        return _run(sess)


def _load_embeddings(candidate: Candidate) -> Optional[CandidateEmbeddings]:
    vectors = load_candidate_embeddings_vectors(candidate)
    if not vectors:
        return None
    return CandidateEmbeddings(**vectors)


def _persist_recommendations(
    session: Session,
    candidate_id: UUID,
    ranked: list[RankedJob],
) -> None:
    delete_recommendations_for_candidate(session, candidate_id)
    rows: list[Recommendation] = []
    for item in ranked:
        factor_scores = dict(item.factor_scores)
        factor_scores["_feed_section"] = item.feed_section
        rows.append(
            Recommendation(
                candidate_id=candidate_id,
                job_id=UUID(item.job_id),
                match_score=item.match_score,
                factor_scores=factor_scores,
                retrieval_scores=item.retrieval_scores,
                explanation=item.explanation,
                rank=item.rank,
            ),
        )
    bulk_insert_recommendations(session, rows)


def _ranked_from_stored(rows: list[Recommendation]) -> list[RankedJob]:
    ranked: list[RankedJob] = []
    for row in rows:
        if row.job is None:
            continue
        factor_scores = dict(row.factor_scores or {})
        feed_section = str(factor_scores.pop("_feed_section", "strong_match"))
        ranked.append(
            RankedJob(
                job_id=str(row.job_id),
                job=row.job,
                rank=row.rank or 0,
                match_score=row.match_score,
                match_percentage=int(round(row.match_score * 100)),
                factor_scores=factor_scores,
                retrieval_scores=dict(row.retrieval_scores or {}),
                explanation=row.explanation,
                feed_section=feed_section,
            ),
        )
    return ranked
