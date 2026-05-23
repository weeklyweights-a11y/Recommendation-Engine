"""SQLAlchemy ORM models."""

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Job(Base):
    """Job listing stored for matching."""

    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_company_title", "company", "title"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    company: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[Optional[str]] = mapped_column(String(512), index=True)
    salary_min: Mapped[Optional[int]] = mapped_column(Integer)
    salary_max: Mapped[Optional[int]] = mapped_column(Integer)
    currency: Mapped[Optional[str]] = mapped_column(String(16), default="USD")
    remote_type: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    sponsorship_available: Mapped[Optional[bool]] = mapped_column(Boolean)
    company_size: Mapped[Optional[str]] = mapped_column(String(32))
    company_stage: Mapped[Optional[str]] = mapped_column(String(32))
    industry: Mapped[Optional[str]] = mapped_column(String(256), index=True)
    experience_level: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    skills_extracted: Mapped[Optional[list[Any]]] = mapped_column(JSONB)
    source_url: Mapped[Optional[str]] = mapped_column(String(2048))
    source_platform: Mapped[Optional[str]] = mapped_column(String(128))
    posted_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), index=True)
    embedding_skill: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    embedding_domain: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    embedding_role: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    embedding_environment: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    is_embedded: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class Candidate(Base):
    """Candidate profile for personalized recommendations."""

    __tablename__ = "candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[Optional[str]] = mapped_column(String(256))
    email: Mapped[Optional[str]] = mapped_column(String(320), unique=True)
    resume_text: Mapped[Optional[str]] = mapped_column(Text)
    resume_filename: Mapped[Optional[str]] = mapped_column(String(512))
    github_username: Mapped[Optional[str]] = mapped_column(String(256))
    github_data: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    profile: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    preferences: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    embedding_skill: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    embedding_domain: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    embedding_role: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    embedding_environment: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    utility_weights: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    recommendations: Mapped[list["Recommendation"]] = relationship(back_populates="candidate")
    feedback_entries: Mapped[list["Feedback"]] = relationship(back_populates="candidate")


class Recommendation(Base):
    """Stored recommendation for a candidate-job pair."""

    __tablename__ = "recommendations"
    __table_args__ = (
        UniqueConstraint("candidate_id", "job_id", name="uq_recommendations_candidate_job"),
        Index("ix_recommendations_candidate_rank", "candidate_id", "rank"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidates.id"),
        nullable=False,
        index=True,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id"),
        nullable=False,
        index=True,
    )
    match_score: Mapped[float] = mapped_column(Float, nullable=False)
    factor_scores: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    retrieval_scores: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    explanation: Mapped[Optional[str]] = mapped_column(Text)
    rank: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    candidate: Mapped["Candidate"] = relationship(back_populates="recommendations")
    job: Mapped["Job"] = relationship()


class Feedback(Base):
    """User feedback on a job recommendation."""

    __tablename__ = "feedback"
    __table_args__ = (
        UniqueConstraint(
            "candidate_id",
            "job_id",
            "action",
            name="uq_feedback_candidate_job_action",
        ),
        Index("ix_feedback_candidate_action", "candidate_id", "action"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidates.id"),
        nullable=False,
        index=True,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id"),
        nullable=False,
        index=True,
    )
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    candidate: Mapped["Candidate"] = relationship(back_populates="feedback_entries")
    job: Mapped["Job"] = relationship()
