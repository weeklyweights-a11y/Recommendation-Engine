"""Persistence helpers for stored recommendations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, joinedload

from config.settings import Settings, get_settings
from src.db.models import Recommendation


def load_cached_recommendations(
    session: Session,
    candidate_id: UUID,
    *,
    settings: Optional[Settings] = None,
) -> list[Recommendation]:
    """Return fresh cached recommendations if within TTL."""
    cfg = settings or get_settings()
    ttl = cfg.recommendation.cache_ttl_seconds
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl)
    stmt = (
        select(Recommendation)
        .options(joinedload(Recommendation.job))
        .where(Recommendation.candidate_id == candidate_id)
        .where(Recommendation.created_at >= cutoff)
        .order_by(Recommendation.rank.asc())
    )
    return list(session.scalars(stmt).all())


def delete_recommendations_for_candidate(session: Session, candidate_id: UUID) -> None:
    """Remove all stored recommendations for a candidate."""
    session.execute(delete(Recommendation).where(Recommendation.candidate_id == candidate_id))


def bulk_insert_recommendations(
    session: Session,
    rows: list[Recommendation],
) -> None:
    """Insert recommendation rows."""
    session.add_all(rows)
    session.flush()
