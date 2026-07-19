"""Tests for Reranker."""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy import JSON, create_engine
from sqlalchemy.orm import sessionmaker

from src.api.schemas.candidate import CandidateProfile, MergedPreferences
from src.matching.graph_retriever import GraphRetriever
from src.matching.reranker import Reranker
from src.matching.schemas import FusedResult, SkillOverlap
from src.db.models import Base, Job


for _col in Job.__table__.columns:
    if "JSON" in type(_col.type).__name__:
        _col.type = JSON()


@pytest.fixture
def rerank_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[Job.__table__])
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    job = Job(
        id=uuid.uuid4(),
        title="Senior ML Engineer",
        company="Acme",
        description="Python PyTorch machine learning senior role",
        location="Remote",
        remote_type="remote",
        experience_level="senior",
        industry="fintech",
        company_stage="seed",
        is_embedded=True,
        posted_date=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.add(job)
    session.commit()
    yield session, job
    session.close()


def test_semantic_passthrough(rerank_session) -> None:
    session, job = rerank_session
    graph = MagicMock(spec=GraphRetriever)
    graph.get_skill_overlap.return_value = SkillOverlap()
    reranker = Reranker(session, graph_retriever=graph)
    profile = CandidateProfile(
        total_years_experience=5.0,
        role_archetype="platform_engineer",
        domains=["fintech"],
        preferences=MergedPreferences(),
    )
    fused = [
        FusedResult(job_id=str(job.id), fused_score=0.8, vector_score=0.75),
    ]
    ranked = reranker.rerank(profile, fused, top_k=5)
    assert len(ranked) == 1
    assert ranked[0].factor_scores["semantic_similarity"] == pytest.approx(0.75)


def test_career_changer_role_floor(rerank_session) -> None:
    session, job = rerank_session
    graph = MagicMock(spec=GraphRetriever)
    graph.get_skill_overlap.return_value = SkillOverlap()
    reranker = Reranker(session, graph_retriever=graph)
    profile = CandidateProfile(
        career_trajectory="career_changer",
        role_archetype="research_scientist",
        total_years_experience=2.0,
        preferences=MergedPreferences(),
    )
    fused = [FusedResult(job_id=str(job.id), fused_score=0.5, vector_score=0.5)]
    ranked = reranker.rerank(profile, fused, top_k=5)
    assert ranked[0].factor_scores["role_shape_match"] >= 0.6


def test_utility_weights_change_order(rerank_session) -> None:
    session, job = rerank_session
    job2 = Job(
        id=uuid.uuid4(),
        title="Junior Analyst",
        company="Other",
        description="entry level",
        remote_type="onsite",
        experience_level="entry",
        industry="healthcare",
        is_embedded=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.add(job2)
    session.commit()

    graph = MagicMock(spec=GraphRetriever)
    graph.get_skill_overlap.return_value = SkillOverlap()
    reranker = Reranker(session, graph_retriever=graph)
    profile = CandidateProfile(
        total_years_experience=1.0,
        domains=["healthcare"],
        preferences=MergedPreferences(),
    )
    fused = [
        FusedResult(job_id=str(job.id), fused_score=0.9, vector_score=0.9),
        FusedResult(job_id=str(job2.id), fused_score=0.8, vector_score=0.4),
    ]
    reranker.rerank(profile, fused, top_k=2)  # verify no error
    custom = reranker.rerank(
        profile,
        fused,
        top_k=2,
        custom_weights={
            "skill_fit": 0.0,
            "experience_alignment": 0.0,
            "domain_relevance": 1.0,
            "role_shape_match": 0.0,
            "location_fit": 0.0,
            "company_stage_alignment": 0.0,
            "semantic_similarity": 0.0,
        },
    )
    assert custom[0].job_id == str(job2.id)


def test_scores_in_unit_interval(rerank_session) -> None:
    session, job = rerank_session
    graph = MagicMock(spec=GraphRetriever)
    graph.get_skill_overlap.return_value = SkillOverlap(
        direct_matches=[{"uri": "x"}],
    )
    reranker = Reranker(session, graph_retriever=graph)
    profile = CandidateProfile(
        skills=[],
        esco_linked_skills=[],
        total_years_experience=4.0,
        preferences=MergedPreferences(),
    )
    fused = [FusedResult(job_id=str(job.id), fused_score=0.6, vector_score=0.6)]
    ranked = reranker.rerank(profile, fused, top_k=1)
    for val in ranked[0].factor_scores.values():
        assert 0.0 <= val <= 1.0
    assert 0.0 <= ranked[0].match_score <= 1.0
