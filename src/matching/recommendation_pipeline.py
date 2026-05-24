"""Full recommendation pipeline: filter, retrieve, rerank, explain, persist."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from src.api.schemas.candidate import CandidateProfile
from src.api.schemas.job import JobResponse
from src.cache.redis_client import get_redis_cache
from src.db.candidate_repository import load_candidate_embeddings_vectors
from src.db.models import Candidate, Job, Recommendation
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


def _recs_cache_key(candidate_id: UUID) -> str:
    return f"recs:{candidate_id}"


def _materialize_ranked_jobs(ranked: list[RankedJob]) -> list[RankedJob]:
    """Convert attached ORM Job rows to JobResponse while the session is active."""
    for item in ranked:
        if isinstance(item.job, Job):
            item.job = JobResponse.model_validate(item.job)
    return ranked


def _normalize_utility_weights(raw: Optional[dict]) -> Optional[dict[str, float]]:
    if not raw:
        return None
    payload = raw
    if isinstance(raw, dict) and "weights" in raw and isinstance(raw["weights"], dict):
        payload = raw["weights"]
    cleaned = {k: float(payload[k]) for k in FACTOR_KEYS if k in payload}
    if not cleaned:
        return None
    total = sum(cleaned.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in cleaned.items()}


def _serialize_ranked_for_redis(ranked: list[RankedJob]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in ranked:
        rows.append(
            {
                "job_id": item.job_id,
                "rank": item.rank,
                "match_score": item.match_score,
                "match_percentage": item.match_percentage,
                "factor_scores": item.factor_scores,
                "retrieval_scores": item.retrieval_scores,
                "graph_matched_skills": item.graph_matched_skills,
                "skill_match_display": item.skill_match_display,
                "feed_section": item.feed_section,
                "explanation": item.explanation,
            },
        )
    return rows


def _load_ranked_from_redis(
    session: Session,
    candidate_id: UUID,
    payload: list[dict[str, Any]],
    *,
    settings: Settings,
) -> list[RankedJob]:
    """Rehydrate ranked jobs from Redis payload and PostgreSQL job rows."""
    cache = get_redis_cache()
    job_ids = [UUID(row["job_id"]) for row in payload]
    jobs_by_id: dict[str, Job] = {}
    missing: list[UUID] = []

    for job_id in job_ids:
        cached = cache.get_json(f"job:{job_id}")
        if cached:
            jobs_by_id[str(job_id)] = JobResponse.model_validate(cached)  # type: ignore[assignment]
        else:
            missing.append(job_id)

    if missing:
        rows = session.scalars(select(Job).where(Job.id.in_(missing))).all()
        for job in rows:
            job_resp = JobResponse.model_validate(job)
            jobs_by_id[str(job.id)] = job_resp
            cache.set_json(
                f"job:{job.id}",
                job_resp.model_dump(mode="json"),
                settings.cache.job_row_ttl_seconds,
            )

    ranked: list[RankedJob] = []
    for row in payload:
        job = jobs_by_id.get(row["job_id"])
        if job is None:
            continue
        ranked.append(
            RankedJob(
                job_id=row["job_id"],
                job=job,
                rank=row.get("rank", 0),
                match_score=row["match_score"],
                match_percentage=row.get("match_percentage", int(round(row["match_score"] * 100))),
                factor_scores=row.get("factor_scores", {}),
                retrieval_scores=row.get("retrieval_scores", {}),
                graph_matched_skills=row.get("graph_matched_skills"),
                skill_match_display=row.get("skill_match_display"),
                feed_section=row.get("feed_section", "strong_match"),
                explanation=row.get("explanation"),
            ),
        )
    return ranked


def _log_pipeline_timings(
    candidate_id: UUID,
    timing: PipelineTiming,
    *,
    jobs_after_filter: int,
    results_returned: int,
    settings: Settings,
) -> None:
    if not settings.cache.log_pipeline_timings:
        return
    payload = {
        "candidate_id": str(candidate_id),
        "timings_ms": {
            "load_candidate": round(timing.load_candidate_ms, 1),
            "hard_filters": round(timing.hard_filters_ms, 1),
            "bm25_retrieval": round(timing.bm25_ms, 1),
            "vector_retrieval": round(timing.vector_ms, 1),
            "graph_retrieval": round(timing.graph_ms, 1),
            "hybrid_fusion": round(timing.fusion_ms, 1),
            "reranking": round(timing.rerank_ms, 1),
            "load_job_details": round(timing.load_job_details_ms, 1),
            "llm_explanations": round(timing.llm_explanations_ms, 1),
            "store_results": round(timing.store_results_ms, 1),
        },
        "total_ms": round(timing.total_ms, 1),
        "jobs_after_filter": jobs_after_filter,
        "results_returned": results_returned,
    }
    logger.info("pipeline_timings %s", json.dumps(payload))


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
        timing = PipelineTiming()
        pipeline_start = time.perf_counter()
        redis = get_redis_cache()

        if not refresh:
            redis_payload = redis.get_json(_recs_cache_key(candidate_id))
            if redis_payload:
                t_load = time.perf_counter()
                ranked = _load_ranked_from_redis(sess, candidate_id, redis_payload, settings=cfg)
                timing.load_job_details_ms = (time.perf_counter() - t_load) * 1000
                timing.total_ms = (time.perf_counter() - pipeline_start) * 1000
                if ranked:
                    _log_pipeline_timings(
                        candidate_id,
                        timing,
                        jobs_after_filter=0,
                        results_returned=len(ranked),
                        settings=cfg,
                    )
                    return PipelineResult(
                        ranked_jobs=_materialize_ranked_jobs(ranked),
                        from_cache=True,
                    )

            cached = load_cached_recommendations(sess, candidate_id, settings=cfg)
            if cached:
                ranked = _materialize_ranked_jobs(_ranked_from_stored(cached))
                redis.set_json(
                    _recs_cache_key(candidate_id),
                    _serialize_ranked_for_redis(ranked),
                    cfg.recommendation.cache_ttl_seconds,
                )
                timing.total_ms = (time.perf_counter() - pipeline_start) * 1000
                return PipelineResult(ranked_jobs=ranked, from_cache=True)

        t_load_cand = time.perf_counter()
        candidate = sess.get(Candidate, candidate_id)
        if not candidate or not candidate.profile:
            raise ValueError("Candidate profile not found")

        from src.feedback.service import apply_feedback_weights

        apply_feedback_weights(candidate_id, sess, settings=cfg)
        candidate = sess.get(Candidate, candidate_id)
        timing.load_candidate_ms = (time.perf_counter() - t_load_cand) * 1000

        profile = CandidateProfile.model_validate(candidate.profile)
        stats = PipelineStats()

        t_hf = time.perf_counter()
        hf = HardFilter(sess, cfg)
        funnel = hf.get_filter_funnel(profile.preferences)
        stats.filter_funnel = funnel
        allowed = hf.filter_jobs(profile.preferences)
        timing.hard_filters_ms = (time.perf_counter() - t_hf) * 1000

        if funnel.final_count < cfg.hard_filter.min_results_warn:
            stats.warnings.append("preferences_too_restrictive")
        if funnel.final_count < cfg.hard_filter.pipeline_min_warn:
            logger.warning(
                "Only %s jobs pass hard filters for candidate %s",
                funnel.final_count,
                candidate_id,
            )

        if not allowed:
            timing.total_ms = (time.perf_counter() - pipeline_start) * 1000
            stats.timing_ms = timing
            _log_pipeline_timings(
                candidate_id,
                timing,
                jobs_after_filter=0,
                results_returned=0,
                settings=cfg,
            )
            return PipelineResult(ranked_jobs=[], stats=stats)

        fused, overlap, hybrid_timing = retrieve_hybrid_parallel(
            candidate_id=candidate_id,
            profile=profile,
            embeddings=_load_embeddings(candidate),
            allowed_job_ids=allowed,
            session=sess,
            settings=cfg,
        )
        timing.bm25_ms = hybrid_timing.bm25_ms
        timing.vector_ms = hybrid_timing.vector_ms
        timing.graph_ms = hybrid_timing.graph_ms
        timing.fusion_ms = hybrid_timing.fusion_ms
        stats.retrieval_overlap = overlap

        t_rerank = time.perf_counter()
        weights = _normalize_utility_weights(candidate.utility_weights)
        reranker = Reranker(sess, cfg)
        ranked = reranker.rerank(
            profile,
            fused,
            top_k=cfg.recommendation.rerank_top_k,
            custom_weights=weights,
        )
        timing.rerank_ms = (time.perf_counter() - t_rerank) * 1000

        t_explain = time.perf_counter()
        explainer = Explainer(cfg)
        explain_top = cfg.recommendation.explain_top_k
        llm_top = min(cfg.explainer.explain_llm_top_k, explain_top)
        explanations = explainer.explain_batch(
            profile,
            ranked[:explain_top],
            max_jobs=explain_top,
            llm_max=llm_top,
        )
        for job_row, explanation in zip(ranked[:explain_top], explanations):
            job_row.explanation = explanation.model_dump_json()
        timing.llm_explanations_ms = (time.perf_counter() - t_explain) * 1000

        store_k = min(cfg.recommendation.store_top_k, len(ranked))
        t_store = time.perf_counter()
        _persist_recommendations(sess, candidate_id, ranked[:store_k])
        timing.store_results_ms = (time.perf_counter() - t_store) * 1000

        stored = ranked[:store_k]
        redis.set_json(
            _recs_cache_key(candidate_id),
            _serialize_ranked_for_redis(stored),
            cfg.recommendation.cache_ttl_seconds,
        )

        timing.total_ms = (time.perf_counter() - pipeline_start) * 1000
        stats.timing_ms = timing
        _log_pipeline_timings(
            candidate_id,
            timing,
            jobs_after_filter=funnel.final_count,
            results_returned=len(stored),
            settings=cfg,
        )
        logger.info(
            "Pipeline timing load=%.0fms hard_filter=%.0fms retrieval=%.0fms rerank=%.0fms explain=%.0fms total=%.0fms",
            timing.load_candidate_ms,
            timing.hard_filters_ms,
            timing.retrieval_ms,
            timing.rerank_ms,
            timing.llm_explanations_ms,
            timing.total_ms,
        )
        return PipelineResult(
            ranked_jobs=_materialize_ranked_jobs(stored),
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
        retrieval_scores = dict(item.retrieval_scores or {})
        if item.graph_matched_skills:
            retrieval_scores["graph_matched_skills"] = item.graph_matched_skills
        if item.skill_match_display:
            retrieval_scores["skill_match_display"] = item.skill_match_display
        rows.append(
            Recommendation(
                candidate_id=candidate_id,
                job_id=UUID(item.job_id),
                match_score=item.match_score,
                factor_scores=factor_scores,
                retrieval_scores=retrieval_scores,
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
        retrieval_scores = dict(row.retrieval_scores or {})
        graph_matched = retrieval_scores.pop("graph_matched_skills", None)
        skill_display = retrieval_scores.pop("skill_match_display", None)
        ranked.append(
            RankedJob(
                job_id=str(row.job_id),
                job=row.job,
                rank=row.rank or 0,
                match_score=row.match_score,
                match_percentage=int(round(row.match_score * 100)),
                factor_scores=factor_scores,
                retrieval_scores=retrieval_scores,
                graph_matched_skills=graph_matched,
                skill_match_display=skill_display,
                explanation=row.explanation,
                feed_section=feed_section,
            ),
        )
    return ranked
