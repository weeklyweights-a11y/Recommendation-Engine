"""API tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_db_session
from src.api.main import create_app
from src.db.models import Candidate
from src.matching.schemas import FilterFunnel, PipelineResult, PipelineStats, RankedJob


def _minimal_profile() -> dict:
    return {
        "name": "Test User",
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


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.integration
def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "services" in body


def test_get_recommendations_missing_candidate(client: TestClient) -> None:
    candidate_id = uuid4()
    session = AsyncMock()
    session.get = AsyncMock(return_value=None)

    async def override_db():
        yield session

    client.app.dependency_overrides[get_db_session] = override_db
    try:
        response = client.get(f"/api/v1/recommendations/{candidate_id}")
    finally:
        client.app.dependency_overrides.clear()
    assert response.status_code == 404


@patch("src.api.routes.recommendations._ranked_from_stored")
@patch("src.api.routes.recommendations.run_recommendation_pipeline")
@patch("src.api.routes.recommendations.load_cached_recommendations")
def test_get_recommendations_returns_payload(
    mock_cache,
    mock_pipeline,
    mock_ranked_from_stored,
    client: TestClient,
) -> None:
    candidate_id = uuid4()
    job_id = uuid4()

    class FakeJob:
        id = job_id
        title = "Engineer"
        company = "Co"
        location = "Remote"
        description = "Build"
        remote_type = "remote"
        salary_min = None
        salary_max = None
        sponsorship_available = False
        company_size = "51-200"
        company_stage = "growth"
        industry = "tech"
        source_url = "https://example.com"
        posted_date = datetime.now(timezone.utc)
        skills_extracted = []
        is_embedded = True
        created_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)

    ranked = RankedJob(
        job_id=str(job_id),
        job=FakeJob(),
        rank=1,
        match_score=0.85,
        match_percentage=85,
        factor_scores={"skill_fit": 0.9},
        feed_section="strong_match",
    )
    mock_cache.return_value = []
    mock_pipeline.return_value = PipelineResult(
        ranked_jobs=[ranked],
        stats=PipelineStats(
            filter_funnel=FilterFunnel(total_jobs=100, final_count=50),
            warnings=[],
        ),
    )
    mock_ranked_from_stored.return_value = [ranked]

    candidate = Candidate()
    candidate.id = candidate_id
    candidate.profile = _minimal_profile()

    session = AsyncMock()
    session.get = AsyncMock(return_value=candidate)

    async def override_db():
        yield session

    client.app.dependency_overrides[get_db_session] = override_db
    try:
        response = client.get(f"/api/v1/recommendations/{candidate_id}")
    finally:
        client.app.dependency_overrides.clear()

    mock_ranked_from_stored.assert_not_called()
    assert response.status_code == 200
    data = response.json()
    assert data["pagination"]["total"] == 1
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["match_percentage"] == 85


@patch("src.api.routes.recommendations._ranked_from_stored")
@patch("src.api.routes.recommendations.run_recommendation_pipeline")
@patch("src.api.routes.recommendations.load_cached_recommendations")
def test_get_recommendations_uses_cache_before_session_closes(
    mock_cache,
    mock_pipeline,
    mock_ranked_from_stored,
    client: TestClient,
) -> None:
    """Cached rows are mapped to RankedJob while the sync session is still open."""
    candidate_id = uuid4()
    job_id = uuid4()

    class FakeJob:
        id = job_id
        title = "Engineer"
        company = "Co"
        location = "Remote"
        description = "Build"
        remote_type = "remote"
        salary_min = None
        salary_max = None
        sponsorship_available = False
        company_size = "51-200"
        company_stage = "growth"
        industry = "tech"
        source_url = "https://example.com"
        posted_date = datetime.now(timezone.utc)
        skills_extracted = []
        is_embedded = True
        created_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)

    ranked = RankedJob(
        job_id=str(job_id),
        job=FakeJob(),
        rank=1,
        match_score=0.9,
        match_percentage=90,
        factor_scores={"skill_fit": 0.9},
        feed_section="strong_match",
    )
    mock_cache.return_value = [object()]
    mock_ranked_from_stored.return_value = [ranked]

    candidate = Candidate()
    candidate.id = candidate_id
    candidate.profile = _minimal_profile()

    session = AsyncMock()
    session.get = AsyncMock(return_value=candidate)

    async def override_db():
        yield session

    client.app.dependency_overrides[get_db_session] = override_db
    try:
        response = client.get(f"/api/v1/recommendations/{candidate_id}")
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 200
    mock_ranked_from_stored.assert_called_once()
    mock_pipeline.assert_not_called()
