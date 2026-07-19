"""End-to-end integration tests for the recommendation pipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_db_session
from src.api.main import create_app
from src.db.models import Candidate
from src.embeddings.schemas import CandidateEmbeddings
from src.matching.reranker import FACTOR_KEYS
from src.matching.schemas import FilterFunnel, PipelineResult, PipelineStats, PipelineTiming, RankedJob
from tests.conftest import integration_services_available
from tests.integration_fixtures import seed_integration_jobs

import numpy as np


def _full_profile(**overrides: object) -> dict:
    base = {
        "name": "Alex Test",
        "email": "alex@example.com",
        "preferences": {
            "work_model": {"value": ["remote"]},
            "company_stage": {"value": ["seed", "series-a"]},
            "salary": {"min": 120000},
            "target_roles": {"value": ["ML Engineer"]},
        },
        "summary": "Python ML engineer",
        "skills": [{"name": "Python", "depth_score": 0.9}, {"name": "PyTorch", "depth_score": 0.85}],
        "experience": [
            {
                "company": "Startup",
                "title": "ML Engineer",
                "start_date": "2020-01",
                "end_date": "present",
            },
        ],
        "education": [],
        "domains": ["tech"],
        "role_archetype": "ml_engineer",
        "career_trajectory": "specialist",
        "esco_linked_skills": [
            {
                "original_name": "Python",
                "esco_uri": "http://esco/skill/python",
                "esco_label": "Python",
                "match_type": "exact",
                "confidence": 1.0,
            },
        ],
        "total_years_experience": 4.0,
    }
    base.update(overrides)
    return base


def _mock_embeddings() -> CandidateEmbeddings:
    vec = np.ones(384, dtype=np.float32)
    return CandidateEmbeddings(skill=vec, domain=vec, role=vec, environment=vec)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.integration
def test_health_endpoint_integration(client: TestClient, integration_services_ok: bool) -> None:
    if not integration_services_ok:
        pytest.skip("Docker services not reachable")
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"healthy", "degraded", "unhealthy"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_invalid_file_type(api_client) -> None:
    if not integration_services_available():
        pytest.skip("Docker services not reachable")
    files = {"resume": ("resume.txt", b"not a resume", "text/plain")}
    response = await api_client.post("/api/v1/candidates", files=files)
    assert response.status_code == 422
    assert "PDF or DOCX" in response.json()["detail"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_empty_file(api_client) -> None:
    if not integration_services_available():
        pytest.skip("Docker services not reachable")
    files = {"resume": ("resume.pdf", b"", "application/pdf")}
    response = await api_client.post("/api/v1/candidates", files=files)
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_recommendations_unknown_candidate(client: TestClient) -> None:
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


@patch("src.api.routes.recommendations.run_recommendation_pipeline")
@patch("src.api.routes.recommendations.load_cached_recommendations")
def test_recommendations_happy_path_mocked(
    mock_cache,
    mock_pipeline,
    client: TestClient,
) -> None:
    candidate_id = uuid4()
    job_id = uuid4()
    factors = {k: 0.5 for k in FACTOR_KEYS}
    from datetime import datetime, timezone

    from src.api.schemas.job import JobResponse

    job_resp = JobResponse(
        id=job_id,
        title="ML Engineer",
        company="Startup",
        location="Remote",
        description="Python",
        remote_type="remote",
        salary_min=130000,
        salary_max=180000,
        sponsorship_available=False,
        company_size="11-50",
        company_stage="seed",
        industry="tech",
        source_url="https://example.com",
        posted_date=datetime.now(timezone.utc),
        skills_extracted=[],
        is_embedded=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    ranked = RankedJob(
        job_id=str(job_id),
        job=job_resp,
        rank=1,
        match_score=0.88,
        match_percentage=88,
        factor_scores=factors,
        feed_section="strong_match",
        explanation=json.dumps({"summary": "Strong fit", "reasons": ["Skills align"]}),
    )
    mock_cache.return_value = []
    mock_pipeline.return_value = PipelineResult(
        ranked_jobs=[ranked],
        stats=PipelineStats(
            filter_funnel=FilterFunnel(total_jobs=100, final_count=80),
            timing_ms=PipelineTiming(total_ms=100),
        ),
    )

    candidate = Candidate()
    candidate.id = candidate_id
    candidate.profile = _full_profile()

    session = AsyncMock()
    session.get = AsyncMock(return_value=candidate)

    async def override_db():
        yield session

    client.app.dependency_overrides[get_db_session] = override_db
    try:
        response = client.get(f"/api/v1/recommendations/{candidate_id}?refresh=true")
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    rec = body["recommendations"][0]
    assert 0 <= rec["match_score"] <= 1
    assert len(rec["factor_scores"]) == len(FACTOR_KEYS)
    assert rec["explanation"]


@pytest.mark.integration
def test_seed_jobs_fixture(integration_services_ok: bool) -> None:
    if not integration_services_ok:
        pytest.skip("Docker services not reachable")
    from src.db.sync_database import get_sync_session

    with get_sync_session() as session:
        ids = seed_integration_jobs(session, count=10)
        session.commit()
        assert len(ids) == 10


@patch("src.cache.invalidation.invalidate_candidate_recommendation_cache")
@patch("src.feedback.service.apply_feedback_weights")
def test_cache_invalidation_on_feedback(mock_weights, mock_inv, client: TestClient) -> None:
    candidate_id = uuid4()
    job_id = uuid4()

    session = AsyncMock()
    candidate = MagicMock()
    candidate.id = candidate_id
    job = MagicMock()
    job.id = job_id
    job.source_url = None
    session.get = AsyncMock(side_effect=lambda model, pk: candidate if pk == candidate_id else job)
    session.add = MagicMock()
    session.commit = AsyncMock()
    from datetime import datetime, timezone

    feedback_id = uuid4()
    created = datetime.now(timezone.utc)

    async def _refresh(row: object) -> object:
        row.id = feedback_id
        row.candidate_id = candidate_id
        row.job_id = job_id
        row.action = "saved"
        row.created_at = created
        return row

    session.refresh = AsyncMock(side_effect=_refresh)

    async def override_db():
        yield session

    with patch("src.db.sync_database.get_sync_session") as mock_sync:
        mock_sync.return_value.__enter__.return_value = MagicMock()
        client.app.dependency_overrides[get_db_session] = override_db
        try:
            response = client.post(
                "/api/v1/feedback",
                json={
                    "candidate_id": str(candidate_id),
                    "job_id": str(job_id),
                    "action": "saved",
                },
            )
        finally:
            client.app.dependency_overrides.clear()

    assert response.status_code == 201
    mock_inv.assert_called_once_with(candidate_id)
    mock_weights.assert_called_once()
