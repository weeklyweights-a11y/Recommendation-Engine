"""Tests for unified profile builder."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.schemas.candidate import CandidatePreferences
from src.ingestion.exceptions import ExtractionFailedError
from src.ingestion.profile_builder import _assemble_profile, build_profile
from src.ingestion.schemas import (
    ActivityMetrics,
    ExtractedProfile,
    ExtractedSkill,
    GitHubProfile,
    InferredPreferences,
)
from src.knowledge_graph.schemas import LinkedSkill
from tests.test_llm_extractor import SAMPLE_PROFILE


def _sample_extracted() -> ExtractedProfile:
    return ExtractedProfile.model_validate(SAMPLE_PROFILE)


def _sample_github() -> GitHubProfile:
    return GitHubProfile(
        username="octocat",
        inferred_skills=["Python", "Docker"],
        languages_distribution={"Python": 0.8, "Go": 0.2},
        activity_metrics=ActivityMetrics(total_repos=5, repos_last_6_months=3),
        overall_assessment="active_builder",
    )


@pytest.mark.asyncio
@patch("src.ingestion.profile_builder.link_skills")
@patch("src.ingestion.profile_builder.extract_profile")
@patch("src.ingestion.profile_builder.fetch_github_profile", new_callable=AsyncMock)
@patch("src.ingestion.profile_builder.parse_resume")
@patch("src.ingestion.profile_builder.validate_resume_file")
async def test_full_pipeline_merge(
    mock_validate: MagicMock,
    mock_parse: MagicMock,
    mock_github: AsyncMock,
    mock_extract: MagicMock,
    mock_link: MagicMock,
) -> None:
    """Profile merges resume, GitHub, and ESCO links."""
    mock_parse.return_value = "resume text " * 20
    mock_github.return_value = _sample_github()
    mock_extract.return_value = _sample_extracted()
    mock_link.return_value = [
        LinkedSkill(
            esco_uri="http://esco/skill/python",
            esco_label="Python",
            match_type="exact",
            confidence=0.95,
        ),
    ]

    profile, _, github, _ = await _assemble_profile(
        "resume.pdf",
        github_username="octocat",
        preferences=CandidatePreferences(target_roles=["Staff Engineer"]),
    )

    assert profile.name == "Ada Lovelace"
    assert profile.github_summary is not None
    assert profile.github_summary.username == "octocat"
    assert github is not None
    assert profile.skills[0].esco_uri is not None
    assert profile.skills[0].esco_match_confidence == pytest.approx(0.95)
    assert profile.preferences.target_roles.source == "explicit"
    assert profile.preferences.target_roles.value == ["Staff Engineer"]
    assert profile.esco_linked_skills


@pytest.mark.asyncio
@patch("src.ingestion.profile_builder.link_skills")
@patch("src.ingestion.profile_builder.extract_profile")
@patch("src.ingestion.profile_builder.fetch_github_profile", new_callable=AsyncMock)
@patch("src.ingestion.profile_builder.parse_resume")
@patch("src.ingestion.profile_builder.validate_resume_file")
async def test_github_unavailable_continues(
    mock_validate: MagicMock,
    mock_parse: MagicMock,
    mock_github: AsyncMock,
    mock_extract: MagicMock,
    mock_link: MagicMock,
) -> None:
    """GitHub failures do not block profile construction."""
    from src.ingestion.exceptions import GitHubUserNotFoundError

    mock_parse.return_value = "resume text " * 20
    mock_github.side_effect = GitHubUserNotFoundError("missing")
    mock_extract.return_value = _sample_extracted()
    mock_link.return_value = [None]

    profile, _, github, _ = await _assemble_profile("resume.pdf", github_username="missing")
    assert github is None
    assert profile.github_summary is None
    assert profile.skills


@pytest.mark.asyncio
@patch("src.ingestion.profile_builder.link_skills")
@patch("src.ingestion.profile_builder.extract_fallback_profile")
@patch("src.ingestion.profile_builder.extract_profile")
@patch("src.ingestion.profile_builder.parse_resume")
@patch("src.ingestion.profile_builder.validate_resume_file")
async def test_llm_fallback_path(
    mock_validate: MagicMock,
    mock_parse: MagicMock,
    mock_extract: MagicMock,
    mock_fallback: MagicMock,
    mock_link: MagicMock,
) -> None:
    """LLM failure triggers rule-based fallback."""
    mock_parse.return_value = "resume text " * 20
    mock_extract.side_effect = ExtractionFailedError("failed")
    mock_fallback.return_value = _sample_extracted()
    mock_link.return_value = [None]

    profile, _, _, _ = await _assemble_profile("resume.pdf")
    mock_fallback.assert_called_once()
    assert profile.name == "Ada Lovelace"


@pytest.mark.asyncio
@patch("src.ingestion.profile_builder._assemble_profile", new_callable=AsyncMock)
async def test_build_profile_delegates(mock_assemble: AsyncMock) -> None:
    """build_profile returns assembled profile."""
    expected = _sample_extracted()
    mock_assemble.return_value = (
        MagicMock(name="profile"),
        "text",
        None,
        "resume.pdf",
    )
    result = await build_profile("resume.pdf")
    assert result is mock_assemble.return_value[0]


def test_skill_depth_prefers_resume_base() -> None:
    """Resume-only skills start from resume base depth."""
    from config.settings import get_settings
    from src.ingestion.profile_builder import _compute_depth_score

    skill = ExtractedSkill(name="Rust", category="programming_language", proficiency="expert")
    score, sources = _compute_depth_score(
        skill,
        on_resume=True,
        github_ctx={"inferred": set(), "languages": set(), "recent_languages": set(), "production_languages": set(), "language_repo_counts": {}},
        settings=get_settings(),
    )
    assert score >= 0.3
    assert sources == ["resume"]
