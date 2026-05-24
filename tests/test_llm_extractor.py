"""Tests for LLM structured profile extraction."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.exceptions import ExtractionFailedError
from src.ingestion.llm_extractor import (
    _strip_json_response,
    extract_profile_with_usage,
)
from src.ingestion.schemas import ExtractedProfile


SAMPLE_PROFILE = {
    "name": "Ada Lovelace",
    "email": "ada@example.com",
    "phone": None,
    "location": "London",
    "skills": [
        {
            "name": "Python",
            "category": "programming_language",
            "proficiency": "expert",
            "years_used": 5,
            "context": "ML pipelines",
        },
    ],
    "experience": [
        {
            "company": "Analytical Engines Inc",
            "title": "Engineer",
            "start_date": "2020-01",
            "end_date": "present",
            "duration_months": 48,
            "description": "Built data systems",
            "domain": "saas",
            "company_size_estimate": "51-200",
            "company_stage_estimate": "growth",
            "role_type": "ic",
            "key_achievements": ["Shipped platform v2"],
        },
    ],
    "education": [],
    "total_years_experience": 5,
    "domains": ["saas"],
    "role_archetype": "platform_engineer",
    "career_trajectory": "ascending",
    "inferred_preferences": {
        "preferred_company_stage": "growth",
        "preferred_team_size": "51-200",
        "preferred_work_style": "fast-paced",
        "likely_looking_for": "Senior platform role",
    },
    "summary": "Platform engineer with strong Python experience.",
}


def test_strip_json_response_removes_fences() -> None:
    """Fence stripping returns raw JSON."""
    raw = '```json\n{"name": "Test"}\n```'
    assert json.loads(_strip_json_response(raw))["name"] == "Test"


@patch("src.ingestion.llm_extractor.genai.Client")
def test_extract_profile_parses_mock_response(mock_client_cls: MagicMock) -> None:
    """Mock Gemini response is parsed into ExtractedProfile."""
    mock_response = MagicMock()
    mock_response.text = json.dumps(SAMPLE_PROFILE)
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=100,
        candidates_token_count=200,
        total_token_count=300,
    )
    mock_client_cls.return_value.models.generate_content.return_value = mock_response

    result = extract_profile_with_usage(
        "Resume text " * 20,
        github_summary="GitHub: Python repos",
        client=mock_client_cls.return_value,
    )
    assert isinstance(result.profile, ExtractedProfile)
    assert result.profile.name == "Ada Lovelace"
    assert result.token_usage.total_tokens == 300

    call_args = mock_client_cls.return_value.models.generate_content.call_args
    assert "GitHub: Python repos" in call_args.kwargs["contents"]


@patch("src.ingestion.llm_extractor.genai.Client")
def test_retry_on_invalid_json(mock_client_cls: MagicMock) -> None:
    """Second attempt succeeds after invalid JSON."""
    bad = MagicMock(text="not json", usage_metadata=MagicMock(prompt_token_count=1, candidates_token_count=0, total_token_count=1))
    good = MagicMock(
        text=json.dumps(SAMPLE_PROFILE),
        usage_metadata=MagicMock(prompt_token_count=10, candidates_token_count=20, total_token_count=30),
    )
    mock_client_cls.return_value.models.generate_content.side_effect = [bad, good]

    result = extract_profile_with_usage("Resume " * 30, client=mock_client_cls.return_value)
    assert result.profile.skills[0].name == "Python"


@patch("src.ingestion.llm_extractor.genai.Client")
def test_extraction_failure_after_retries(mock_client_cls: MagicMock) -> None:
    """Raises ExtractionFailedError when both attempts fail."""
    mock_client_cls.return_value.models.generate_content.return_value = MagicMock(
        text="broken",
        usage_metadata=MagicMock(prompt_token_count=1, candidates_token_count=0, total_token_count=1),
    )
    with pytest.raises(ExtractionFailedError):
        extract_profile_with_usage("Resume " * 30, client=mock_client_cls.return_value)


@patch("src.ingestion.llm_extractor.genai.Client")
def test_validation_rejects_empty_experience(mock_client_cls: MagicMock) -> None:
    """Incomplete payloads without experience are rejected."""
    payload = dict(SAMPLE_PROFILE)
    payload["experience"] = []
    mock_client_cls.return_value.models.generate_content.return_value = MagicMock(
        text=json.dumps(payload),
        usage_metadata=MagicMock(prompt_token_count=1, candidates_token_count=1, total_token_count=2),
    )
    with pytest.raises(ExtractionFailedError):
        extract_profile_with_usage("Resume " * 30, client=mock_client_cls.return_value)
