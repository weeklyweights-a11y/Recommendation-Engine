"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-05-23

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial tables."""
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("company", sa.String(length=512), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("location", sa.String(length=512), nullable=True),
        sa.Column("salary_min", sa.Integer(), nullable=True),
        sa.Column("salary_max", sa.Integer(), nullable=True),
        sa.Column("currency", sa.String(length=16), nullable=True),
        sa.Column("remote_type", sa.String(length=32), nullable=True),
        sa.Column("sponsorship_available", sa.Boolean(), nullable=True),
        sa.Column("company_size", sa.String(length=32), nullable=True),
        sa.Column("company_stage", sa.String(length=32), nullable=True),
        sa.Column("industry", sa.String(length=256), nullable=True),
        sa.Column("experience_level", sa.String(length=32), nullable=True),
        sa.Column("skills_extracted", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("source_url", sa.String(length=2048), nullable=True),
        sa.Column("source_platform", sa.String(length=128), nullable=True),
        sa.Column("posted_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("embedding_skill", sa.LargeBinary(), nullable=True),
        sa.Column("embedding_domain", sa.LargeBinary(), nullable=True),
        sa.Column("embedding_role", sa.LargeBinary(), nullable=True),
        sa.Column("embedding_environment", sa.LargeBinary(), nullable=True),
        sa.Column("is_embedded", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_jobs_company"), "jobs", ["company"], unique=False)
    op.create_index(op.f("ix_jobs_experience_level"), "jobs", ["experience_level"], unique=False)
    op.create_index(op.f("ix_jobs_industry"), "jobs", ["industry"], unique=False)
    op.create_index(op.f("ix_jobs_is_embedded"), "jobs", ["is_embedded"], unique=False)
    op.create_index(op.f("ix_jobs_location"), "jobs", ["location"], unique=False)
    op.create_index(op.f("ix_jobs_posted_date"), "jobs", ["posted_date"], unique=False)
    op.create_index(op.f("ix_jobs_remote_type"), "jobs", ["remote_type"], unique=False)
    op.create_index(op.f("ix_jobs_title"), "jobs", ["title"], unique=False)
    op.create_index("ix_jobs_company_title", "jobs", ["company", "title"], unique=False)

    op.create_table(
        "candidates",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=True),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column("resume_text", sa.Text(), nullable=True),
        sa.Column("resume_filename", sa.String(length=512), nullable=True),
        sa.Column("github_username", sa.String(length=256), nullable=True),
        sa.Column("github_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("profile", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("preferences", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("embedding_skill", sa.LargeBinary(), nullable=True),
        sa.Column("embedding_domain", sa.LargeBinary(), nullable=True),
        sa.Column("embedding_role", sa.LargeBinary(), nullable=True),
        sa.Column("embedding_environment", sa.LargeBinary(), nullable=True),
        sa.Column("utility_weights", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )

    op.create_table(
        "recommendations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("match_score", sa.Float(), nullable=False),
        sa.Column("factor_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("retrieval_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["candidate_id"], ["candidates.id"]),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("candidate_id", "job_id", name="uq_recommendations_candidate_job"),
    )
    op.create_index(
        op.f("ix_recommendations_candidate_id"),
        "recommendations",
        ["candidate_id"],
        unique=False,
    )
    op.create_index(op.f("ix_recommendations_job_id"), "recommendations", ["job_id"], unique=False)
    op.create_index(
        "ix_recommendations_candidate_rank",
        "recommendations",
        ["candidate_id", "rank"],
        unique=False,
    )

    op.create_table(
        "feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["candidate_id"], ["candidates.id"]),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "candidate_id",
            "job_id",
            "action",
            name="uq_feedback_candidate_job_action",
        ),
    )
    op.create_index(op.f("ix_feedback_candidate_id"), "feedback", ["candidate_id"], unique=False)
    op.create_index(op.f("ix_feedback_job_id"), "feedback", ["job_id"], unique=False)
    op.create_index(
        "ix_feedback_candidate_action",
        "feedback",
        ["candidate_id", "action"],
        unique=False,
    )


def downgrade() -> None:
    """Drop initial tables."""
    op.drop_index("ix_feedback_candidate_action", table_name="feedback")
    op.drop_index(op.f("ix_feedback_job_id"), table_name="feedback")
    op.drop_index(op.f("ix_feedback_candidate_id"), table_name="feedback")
    op.drop_table("feedback")
    op.drop_index("ix_recommendations_candidate_rank", table_name="recommendations")
    op.drop_index(op.f("ix_recommendations_job_id"), table_name="recommendations")
    op.drop_index(op.f("ix_recommendations_candidate_id"), table_name="recommendations")
    op.drop_table("recommendations")
    op.drop_table("candidates")
    op.drop_index("ix_jobs_company_title", table_name="jobs")
    op.drop_index(op.f("ix_jobs_title"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_remote_type"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_posted_date"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_location"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_is_embedded"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_industry"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_experience_level"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_company"), table_name="jobs")
    op.drop_table("jobs")
