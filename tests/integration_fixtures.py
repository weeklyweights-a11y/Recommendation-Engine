"""Helpers for integration tests (job seeding and resume fixtures)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.db.models import Job
from src.embeddings.encoder import serialize_embedding
import numpy as np

EMBEDDING_DIM = 384


def make_job(
    *,
    title: str,
    company: str,
    remote_type: str = "remote",
    company_stage: str = "seed",
    industry: str = "tech",
    salary_min: int | None = 120000,
    skills: list[dict] | None = None,
) -> Job:
    """Create an in-memory Job row (not yet persisted)."""
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    blob = serialize_embedding(vec)
    return Job(
        id=uuid.uuid4(),
        title=title,
        company=company,
        description=f"{title} at {company}. Python ML role.",
        location="Remote",
        salary_min=salary_min,
        salary_max=(salary_min + 50000) if salary_min else None,
        remote_type=remote_type,
        company_stage=company_stage,
        industry=industry,
        experience_level="mid",
        skills_extracted=skills or [{"name": "Python", "esco_uri": "http://data.europa.eu/esco/skill/python"}],
        source_url=f"https://example.com/jobs/{uuid.uuid4()}",
        posted_date=datetime.now(timezone.utc),
        embedding_skill=blob,
        embedding_domain=blob,
        embedding_role=blob,
        embedding_environment=blob,
        is_embedded=True,
    )


def seed_integration_jobs(session: Session, count: int = 100) -> list[uuid.UUID]:
    """Insert varied jobs for integration tests."""
    stages = ["seed", "series-a", "series-b", "enterprise", "growth"]
    remotes = ["remote", "hybrid", "onsite"]
    industries = ["tech", "fintech", "healthcare", "saas"]
    ids: list[uuid.UUID] = []
    for i in range(count):
        job = make_job(
            title=f"Engineer {i}",
            company=f"Company {i % 20}",
            remote_type=remotes[i % len(remotes)],
            company_stage=stages[i % len(stages)],
            industry=industries[i % len(industries)],
            salary_min=80000 + (i * 1000),
        )
        session.add(job)
        ids.append(job.id)
    session.flush()
    return ids
