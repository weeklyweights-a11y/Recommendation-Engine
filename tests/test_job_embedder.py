"""Tests for job field extraction and embedding text construction."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings.encoder import EMBEDDING_DIM
from src.embeddings.job_embedder import (
    _build_domain_text,
    _build_role_text,
    _build_skill_text,
    extract_job_fields,
    link_job_skills,
)
from src.embeddings.job_field_extractor import extract_job_fields_rule
from src.embeddings.schemas import JobFields, LinkedJobSkill


def _mock_job(**kwargs):
    job = MagicMock()
    job.title = kwargs.get("title", "Senior Python Engineer")
    job.company = kwargs.get("company", "Acme Corp")
    job.description = kwargs.get(
        "description",
        "We need Python, PyTorch, and AWS. Senior engineer role. Fast-paced fintech startup.",
    )
    job.industry = kwargs.get("industry", "fintech")
    job.company_stage = kwargs.get("company_stage", "series_b")
    job.company_size = kwargs.get("company_size", "201-500")
    job.remote_type = kwargs.get("remote_type", "hybrid")
    return job


def test_rule_extractor_finds_skills_and_level():
    job = _mock_job()
    fields = extract_job_fields_rule(job)
    assert "Python" in fields.required_skills
    assert fields.role_level == "senior"
    assert fields.responsibilities_summary == job.description[:500]


def test_rule_responsibilities_truncated_to_500_chars():
    long_desc = "x" * 800
    job = _mock_job(description=long_desc)
    fields = extract_job_fields_rule(job)
    assert len(fields.responsibilities_summary) == 500


def test_domain_text_includes_industry_keywords():
    fields = JobFields(
        domain="fintech",
        company_description="startup",
        industry_keywords_from_description="fintech, Python",
    )
    text = _build_domain_text(fields, "")
    assert "fintech" in text
    assert "Python" in text


def test_link_job_skills_degrades_on_failure():
    with patch("src.embeddings.job_embedder.link_skill", side_effect=RuntimeError("neo4j down")):
        linked = link_job_skills(["Python"])
    assert len(linked) == 1
    assert linked[0].name == "Python"
    assert linked[0].esco_uri is None


@patch("src.embeddings.job_embedder.get_encoder")
def test_embed_job_record_mock_encoder(mock_get_encoder):
    mock_encoder = MagicMock()
    mock_encoder.encode_batch.return_value = np.random.randn(4, EMBEDDING_DIM).astype(np.float32)
    mock_get_encoder.return_value = mock_encoder
    from src.embeddings.job_embedder import embed_job_record

    job = _mock_job()
    embeddings, skills_json = embed_job_record(job, use_llm=False)
    assert embeddings.skill.shape == (EMBEDDING_DIM,)
    assert "skills" in skills_json
