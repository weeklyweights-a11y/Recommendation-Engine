"""Degradation path tests (mocked external failures)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4


from src.api.schemas.candidate import CandidateProfile
from src.ingestion.schemas import (
    ExtractedExperience,
    ExtractedProfile,
    ExtractedSkill,
    InferredPreferences,
)
from src.matching.explainer import Explainer
from src.matching.schemas import RankedJob


def _profile() -> CandidateProfile:
    return CandidateProfile(
        name="Test",
        preferences={},
        summary="ML engineer",
        skills=[],
        experience=[],
        education=[],
        domains=["tech"],
        role_archetype="ml_engineer",
        career_trajectory="specialist",
        esco_linked_skills=[],
        total_years_experience=4.0,
    )


def _ranked() -> RankedJob:
    job = MagicMock()
    job.title = "ML Engineer"
    job.company = "Startup"
    job.description = "Build models"
    job.updated_at = None
    return RankedJob(
        job_id=str(uuid4()),
        job=job,
        rank=1,
        match_score=0.8,
        match_percentage=80,
        factor_scores={"skill_fit": 0.8, "semantic_similarity": 0.7},
    )


@patch.object(Explainer, "_call_gemini", side_effect=RuntimeError("LLM down"))
def test_llm_explain_uses_template_fallback(_mock_gemini) -> None:
    explainer = Explainer()
    result = explainer.explain_match(_profile(), _ranked())
    assert result.generated_by == "template_fallback"
    assert result.summary


@patch("src.ingestion.profile_builder.fetch_github_profile", side_effect=RuntimeError("GitHub down"))
@patch("src.ingestion.profile_builder.extract_profile")
@patch("src.ingestion.profile_builder.parse_resume")
def test_github_failure_profile_builds(
    mock_parse,
    mock_extract,
    _mock_github,
    tmp_path,
) -> None:
    from src.ingestion.profile_builder import build_profile

    mock_parse.return_value = "Python engineer resume " * 20
    mock_extract.return_value = ExtractedProfile(
        name="Test",
        skills=[ExtractedSkill(name="Python", category="other", proficiency="intermediate")],
        experience=[
            ExtractedExperience(
                company="Co",
                title="Eng",
                start_date="2020-01",
                end_date="present",
            ),
        ],
        education=[],
        domains=[],
        role_archetype="ml_engineer",
        career_trajectory="specialist",
        inferred_preferences=InferredPreferences(),
        summary="summary",
    )
    resume = tmp_path / "r.pdf"
    resume.write_bytes(b"%PDF-1.4 trailer<<>>\n%%EOF\n")
    import asyncio

    profile, _ = asyncio.run(build_profile(str(resume), github_username="someuser"))
    assert profile.github_summary is None


@patch("src.matching.recommendation_pipeline._load_embeddings", return_value=None)
@patch("src.feedback.service.apply_feedback_weights")
@patch("src.matching.recommendation_pipeline.get_redis_cache")
@patch("src.matching.recommendation_pipeline.retrieve_hybrid_parallel")
@patch("src.matching.recommendation_pipeline.HardFilter")
@patch("src.matching.recommendation_pipeline.get_sync_session")
def test_redis_down_pipeline_completes(
    mock_session_ctx,
    mock_hf_cls,
    mock_hybrid,
    mock_redis,
    _mock_feedback,
    _mock_embed_load,
) -> None:
    from src.matching.recommendation_pipeline import run_recommendation_pipeline
    from src.matching.schemas import FilterFunnel, FusedResult, HybridTiming

    mock_redis.return_value.get_json.return_value = None
    mock_redis.return_value.health_check.return_value = False

    candidate_id = uuid4()
    session = MagicMock()
    mock_session_ctx.return_value.__enter__.return_value = session
    candidate = MagicMock()
    candidate.profile = _profile().model_dump()
    candidate.utility_weights = None
    session.get.return_value = candidate

    hf = mock_hf_cls.return_value
    hf.get_filter_funnel.return_value = FilterFunnel(total_jobs=10, final_count=5)
    hf.filter_jobs.return_value = {str(uuid4())}
    mock_hybrid.return_value = ([FusedResult(job_id=str(uuid4()), fused_score=0.5)], MagicMock(), HybridTiming())

    with patch("src.matching.recommendation_pipeline.Reranker") as mock_rr, patch(
        "src.matching.recommendation_pipeline.Explainer",
    ) as mock_ex, patch("src.matching.recommendation_pipeline.get_settings") as mock_cfg:
        mock_rr.return_value.rerank.return_value = []
        mock_ex.return_value.explain_batch.return_value = []
        cfg = MagicMock()
        cfg.hard_filter.min_results_warn = 50
        cfg.hard_filter.pipeline_min_warn = 10
        cfg.recommendation.rerank_top_k = 50
        cfg.recommendation.store_top_k = 50
        cfg.recommendation.explain_top_k = 20
        cfg.recommendation.cache_ttl_seconds = 3600
        cfg.explainer.explain_llm_top_k = 10
        cfg.cache.job_row_ttl_seconds = 21600
        mock_cfg.return_value = cfg
        result = run_recommendation_pipeline(candidate_id, refresh=True)
    assert result.ranked_jobs == []
