"""Unit tests for the recommendation pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.matching.recommendation_pipeline import run_recommendation_pipeline
from src.matching.schemas import FilterFunnel, FusedResult, HybridTiming, RankedJob


def _funnel(final: int = 100) -> FilterFunnel:
    return FilterFunnel(
        total_jobs=1000,
        final_count=final,
        most_restrictive_filter="none",
    )


def _ranked(job_id: str, rank: int) -> RankedJob:
    return RankedJob(
        job_id=job_id,
        job=MagicMock(),
        rank=rank,
        match_score=0.8,
        match_percentage=80,
        factor_scores={"skill_fit": 0.8},
    )


@patch("src.matching.recommendation_pipeline.get_redis_cache")
@patch("src.matching.recommendation_pipeline.Explainer")
@patch("src.matching.recommendation_pipeline.Reranker")
@patch("src.matching.recommendation_pipeline.retrieve_hybrid_parallel")
@patch("src.matching.recommendation_pipeline.HardFilter")
@patch("src.matching.recommendation_pipeline.get_sync_session")
def test_pipeline_empty_allowed_set(
    mock_session_ctx,
    mock_hf_cls,
    mock_hybrid,
    mock_reranker_cls,
    mock_explainer_cls,
    mock_redis,
) -> None:
    mock_redis.return_value.get_json.return_value = None

    candidate_id = uuid4()
    session = MagicMock()
    mock_session_ctx.return_value.__enter__.return_value = session

    candidate = MagicMock()
    candidate.profile = {"name": "Test", "preferences": {}}
    candidate.utility_weights = None
    session.get.return_value = candidate

    hf = mock_hf_cls.return_value
    hf.get_filter_funnel.return_value = _funnel(final=0)
    hf.filter_jobs.return_value = set()

    result = run_recommendation_pipeline(candidate_id, refresh=True)

    assert result.ranked_jobs == []
    assert "preferences_too_restrictive" in result.stats.warnings
    mock_hybrid.assert_not_called()
    mock_reranker_cls.assert_not_called()
    mock_explainer_cls.assert_not_called()


@patch("src.feedback.service.apply_feedback_weights")
@patch("src.matching.recommendation_pipeline.get_redis_cache")
@patch("src.matching.recommendation_pipeline._load_embeddings", return_value=None)
@patch("src.matching.recommendation_pipeline.bulk_insert_recommendations")
@patch("src.matching.recommendation_pipeline.delete_recommendations_for_candidate")
@patch("src.matching.recommendation_pipeline.Explainer")
@patch("src.matching.recommendation_pipeline.Reranker")
@patch("src.matching.recommendation_pipeline.retrieve_hybrid_parallel")
@patch("src.matching.recommendation_pipeline.HardFilter")
@patch("src.matching.recommendation_pipeline.get_sync_session")
def test_pipeline_explains_top_20_only(
    mock_session_ctx,
    mock_hf_cls,
    mock_hybrid,
    mock_reranker_cls,
    mock_explainer_cls,
    mock_delete,
    mock_bulk,
    _mock_embeddings,
    mock_redis,
    _mock_feedback,
) -> None:
    candidate_id = uuid4()
    session = MagicMock()
    mock_session_ctx.return_value.__enter__.return_value = session

    candidate = MagicMock()
    candidate.profile = {
        "name": "Test",
        "preferences": {},
        "summary": "",
        "skills": [],
        "experience": [],
        "education": [],
        "domains": [],
        "role_archetype": "generalist",
        "career_trajectory": "lateral",
        "esco_linked_skills": [],
        "total_years_experience": 1.0,
    }
    candidate.utility_weights = None
    session.get.return_value = candidate

    hf = mock_hf_cls.return_value
    hf.get_filter_funnel.return_value = _funnel(final=200)
    hf.filter_jobs.return_value = {str(uuid4())}

    mock_redis.return_value.get_json.return_value = None
    mock_hybrid.return_value = (
        [FusedResult(job_id="j1", fused_score=0.9)],
        MagicMock(),
        HybridTiming(),
    )

    ranked = [_ranked(str(uuid4()), i) for i in range(1, 26)]
    mock_reranker_cls.return_value.rerank.return_value = ranked

    explanation = MagicMock()
    explanation.model_dump_json.return_value = '{"summary":"ok"}'
    mock_explainer_cls.return_value.explain_batch.return_value = [explanation] * 20

    with patch("src.matching.recommendation_pipeline.get_settings") as mock_settings:
        cfg = MagicMock()
        cfg.hard_filter.min_results_warn = 50
        cfg.hard_filter.pipeline_min_warn = 10
        cfg.recommendation.rerank_top_k = 50
        cfg.recommendation.store_top_k = 50
        cfg.recommendation.explain_top_k = 20
        cfg.recommendation.cache_ttl_seconds = 3600
        cfg.explainer.explain_llm_top_k = 20
        cfg.cache.job_row_ttl_seconds = 21600
        cfg.cache.log_pipeline_timings = False
        mock_settings.return_value = cfg

        result = run_recommendation_pipeline(candidate_id, refresh=True)

    assert len(result.ranked_jobs) == 25
    explained = [r for r in result.ranked_jobs if r.explanation]
    assert len(explained) == 20
    assert result.ranked_jobs[20].explanation is None
    mock_explainer_cls.return_value.explain_batch.assert_called_once()
    mock_bulk.assert_called_once()
