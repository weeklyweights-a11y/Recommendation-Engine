"""End-to-end Phase 2 pipeline tests (mocked external APIs)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fitz
import numpy as np
import pytest

from src.api.schemas.candidate import CandidatePreferences
from src.embeddings.encoder import EMBEDDING_DIM
from src.ingestion.profile_builder import build_profile
from src.ingestion.schemas import ActivityMetrics, GitHubProfile
from src.knowledge_graph.schemas import LinkedSkill
from tests.test_llm_extractor import SAMPLE_PROFILE


@pytest.fixture
def resume_pdf(tmp_path: Path) -> Path:
    """Minimal resume PDF fixture."""
    pdf_path = tmp_path / "resume.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Senior Python Engineer at Acme Corp. Built ML pipelines and APIs for five years. "
        "Skills: Python, FastAPI, PostgreSQL, Docker, Kubernetes.",
    )
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.mark.integration
@pytest.mark.asyncio
@patch("src.embeddings.candidate_embedder.embed_candidate")
@patch("src.ingestion.profile_builder.link_skills")
@patch("src.ingestion.profile_builder.extract_profile")
@patch("src.ingestion.profile_builder.fetch_github_profile", new_callable=AsyncMock)
async def test_build_profile_pipeline_mocked(
    mock_github: AsyncMock,
    mock_extract: MagicMock,
    mock_link: MagicMock,
    mock_embed: MagicMock,
    resume_pdf: Path,
) -> None:
    """build_profile returns profile and four 384-dim embeddings."""
    from src.ingestion.schemas import ExtractedProfile

    mock_extract.return_value = ExtractedProfile.model_validate(SAMPLE_PROFILE)
    mock_github.return_value = GitHubProfile(
        username="octocat",
        inferred_skills=["Python"],
        languages_distribution={"Python": 1.0},
        activity_metrics=ActivityMetrics(total_repos=2),
        overall_assessment="active_builder",
    )
    mock_link.return_value = [
        LinkedSkill(
            esco_uri="http://esco/python",
            esco_label="Python",
            match_type="exact",
            confidence=0.9,
        ),
    ]
    skill = np.ones(EMBEDDING_DIM, dtype=np.float32)
    domain = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    domain[1] = 1.0
    role = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    role[2] = 1.0
    environment = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    environment[3] = 1.0
    mock_embed.return_value = MagicMock(
        skill=skill,
        domain=domain,
        role=role,
        environment=environment,
    )

    profile, embeddings = await build_profile(
        str(resume_pdf),
        github_username="octocat",
        preferences=CandidatePreferences(target_roles=["Staff Engineer"]),
    )

    assert profile.name == "Ada Lovelace"
    assert len(profile.skills) >= 1
    assert embeddings.skill.shape == (EMBEDDING_DIM,)
    assert embeddings.domain.shape == (EMBEDDING_DIM,)
    assert not np.array_equal(embeddings.skill, embeddings.domain)
