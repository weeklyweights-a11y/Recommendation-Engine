"""Cache invalidation helpers."""

from __future__ import annotations

import logging
from uuid import UUID

from src.cache.redis_client import get_redis_cache
from src.db.recommendation_repository import delete_recommendations_for_candidate
from src.db.sync_database import get_sync_session

logger = logging.getLogger(__name__)


def invalidate_candidate_recommendation_cache(candidate_id: UUID) -> None:
    """Clear Redis and PostgreSQL recommendation caches for a candidate."""
    cache = get_redis_cache()
    cache.delete_pattern(f"recs:{candidate_id}*")
    cache.delete_pattern(f"explain:{candidate_id}:*")
    with get_sync_session() as session:
        delete_recommendations_for_candidate(session, candidate_id)
    logger.info("Invalidated recommendation cache for candidate %s", candidate_id)
