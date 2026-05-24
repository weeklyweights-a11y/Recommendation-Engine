"""Persistence helpers for candidate profiles."""

from __future__ import annotations

from typing import Any, Optional

from src.embeddings.schemas import CandidateEmbeddings  # noqa: TC001 — used at runtime
from uuid import UUID

from sqlalchemy.orm import Session

from src.api.schemas.candidate import CandidateProfile, MergedPreferences
from src.db.models import Candidate


def upsert_candidate_profile(
    session: Session,
    profile: CandidateProfile,
    *,
    resume_text: str,
    resume_filename: str,
    github_username: Optional[str] = None,
    github_data: Optional[dict[str, Any]] = None,
    embeddings: Optional[CandidateEmbeddings] = None,
    candidate_id: Optional[UUID] = None,
) -> Candidate:
    """Insert or update a candidate row from a built profile."""
    candidate: Candidate | None = None
    if candidate_id is not None:
        candidate = session.get(Candidate, candidate_id)
    elif profile.email:
        candidate = session.query(Candidate).filter(Candidate.email == profile.email).one_or_none()

    if candidate is None:
        candidate = Candidate()
        session.add(candidate)

    candidate.name = profile.name
    candidate.email = profile.email
    candidate.resume_text = resume_text
    candidate.resume_filename = resume_filename
    candidate.github_username = github_username
    candidate.github_data = github_data
    candidate.profile = profile.model_dump(mode="json")
    candidate.preferences = profile.preferences.model_dump(mode="json")

    if embeddings is not None:
        from src.embeddings.encoder import serialize_embedding

        candidate.embedding_skill = serialize_embedding(embeddings.skill)
        candidate.embedding_domain = serialize_embedding(embeddings.domain)
        candidate.embedding_role = serialize_embedding(embeddings.role)
        candidate.embedding_environment = serialize_embedding(embeddings.environment)

    session.flush()
    session.refresh(candidate)
    return candidate


def load_merged_preferences(raw: Optional[dict[str, Any]]) -> MergedPreferences:
    """Deserialize merged preferences from JSONB."""
    if not raw:
        return MergedPreferences()
    return MergedPreferences.model_validate(raw)


def load_profile_embeddings(candidate: Candidate) -> Optional[dict[str, Any]]:
    """Deserialize stored embedding vectors for a candidate."""
    return load_candidate_embeddings_vectors(candidate)


def load_candidate_embeddings_vectors(candidate: Candidate) -> Optional[dict[str, Any]]:
    """Deserialize stored embedding vectors for a candidate."""
    from src.embeddings.encoder import deserialize_embedding

    columns = (
        candidate.embedding_skill,
        candidate.embedding_domain,
        candidate.embedding_role,
        candidate.embedding_environment,
    )
    if not all(columns):
        return None
    return {
        "skill": deserialize_embedding(candidate.embedding_skill),  # type: ignore[arg-type]
        "domain": deserialize_embedding(candidate.embedding_domain),  # type: ignore[arg-type]
        "role": deserialize_embedding(candidate.embedding_role),  # type: ignore[arg-type]
        "environment": deserialize_embedding(candidate.embedding_environment),  # type: ignore[arg-type]
    }


def load_candidate_embeddings(candidate: Candidate) -> Optional[CandidateEmbeddings]:
    """Load CandidateEmbeddings from ORM binary columns."""
    vectors = load_candidate_embeddings_vectors(candidate)
    if vectors is None:
        return None
    return CandidateEmbeddings(**vectors)
