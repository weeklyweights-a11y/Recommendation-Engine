"""Tests for Explainer."""

from unittest.mock import MagicMock, patch

import pytest

from src.api.schemas.candidate import CandidateProfile
from src.db.models import Job
from src.matching.explainer import Explainer
from src.matching.schemas import RankedJob


def _ranked(job_id: str = "job-1") -> RankedJob:
    job = Job(
        title="ML Engineer",
        company="Acme",
        description="Python ML",
    )
    return RankedJob(
        job_id=job_id,
        job=job,
        rank=1,
        match_score=0.8,
        match_percentage=80,
        factor_scores={
            "skill_fit": 0.85,
            "experience_alignment": 0.7,
            "domain_relevance": 0.9,
            "role_shape_match": 0.6,
            "location_fit": 1.0,
            "company_stage_alignment": 0.7,
            "semantic_similarity": 0.75,
        },
    )


MOCK_LLM_TEXT = """SUMMARY: This is a strong match based on skills and domain fit.
REASONS:
- Skill fit is high
- Domain aligns well
- Semantic similarity supports the match
GAPS: None significant"""


@patch("src.matching.explainer.genai.Client")
def test_explain_match_parses_response(mock_client_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.text = MOCK_LLM_TEXT
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    with patch.object(Explainer, "_get_client", return_value=mock_client):
        explainer = Explainer()
        explainer._settings.llm.google_ai_api_key = "test-key"
        result = explainer.explain_match(CandidateProfile(), _ranked())

    assert result.generated_by == "llm"
    assert "strong match" in result.summary.lower()
    assert len(result.reasons) >= 1


@patch("src.matching.explainer.genai.Client")
def test_template_fallback_on_failure(mock_client_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.models.generate_content.side_effect = RuntimeError("api down")

    with patch.object(Explainer, "_get_client", return_value=mock_client):
        explainer = Explainer()
        explainer._settings.llm.google_ai_api_key = "test-key"
        result = explainer.explain_match(CandidateProfile(), _ranked())

    assert result.generated_by == "template_fallback"
    assert "80%" in result.summary


@patch("src.matching.explainer.genai.Client")
def test_cache_avoids_second_llm_call(mock_client_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.text = MOCK_LLM_TEXT
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    profile = CandidateProfile()
    job = _ranked()
    with patch.object(Explainer, "_get_client", return_value=mock_client):
        explainer = Explainer()
        explainer._settings.llm.google_ai_api_key = "test-key"
        explainer.explain_match(profile, job)
        explainer.explain_match(profile, job)

    assert mock_client.models.generate_content.call_count == 1


def test_system_prompt_constrains_scoring() -> None:
    from src.matching.explainer_prompts import EXPLAINER_SYSTEM_PROMPT

    assert "must NOT argue" in EXPLAINER_SYSTEM_PROMPT
    assert "narrating the scores" in EXPLAINER_SYSTEM_PROMPT
