"""Health check routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings
from src.api.dependencies import get_db_session, get_embedding_encoder, get_faiss_manager, get_neo4j, get_settings_dep
from src.db.models import Job
from src.matching.bm25_retriever import BM25Retriever

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(
    settings: Settings = Depends(get_settings_dep),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Report connectivity for infrastructure services."""
    services: dict[str, Any] = {}

    try:
        await session.execute(text("SELECT 1"))
        embedded = await session.scalar(
            select(func.count()).select_from(Job).where(Job.is_embedded.is_(True)),
        )
        services["postgresql"] = {"status": "connected", "embedded_jobs": embedded}
        pg_ok = True
    except Exception as exc:
        services["postgresql"] = {"status": "disconnected", "error": str(exc)}
        pg_ok = False

    try:
        neo4j = get_neo4j(settings)
        rows = neo4j.run_query("MATCH (s:Skill) RETURN count(s) AS c")
        count = rows[0].get("c", 0) if rows else 0
        services["neo4j"] = {"status": "connected", "skill_count": count}
        neo4j_ok = True
    except Exception as exc:
        services["neo4j"] = {"status": "disconnected", "error": str(exc)}
        neo4j_ok = False

    try:
        bm25 = BM25Retriever(settings)
        es_ok = bm25.health_check()
        services["elasticsearch"] = {
            "status": "connected" if es_ok else "disconnected",
        }
    except Exception as exc:
        services["elasticsearch"] = {"status": "disconnected", "error": str(exc)}
        es_ok = False

    try:
        faiss = get_faiss_manager(settings)
        index_stats = faiss.get_index_stats()
        faiss_ok = any(s.get("exists") for s in index_stats.values())
        services["faiss"] = {
            "status": "loaded" if faiss_ok else "not_loaded",
            "indexes": index_stats,
        }
    except Exception as exc:
        services["faiss"] = {"status": "not_loaded", "error": str(exc)}
        faiss_ok = False

    try:
        import redis

        client = redis.from_url(settings.redis.redis_url)
        client.ping()
        services["redis"] = {"status": "connected"}
        redis_ok = True
    except Exception as exc:
        services["redis"] = {"status": "disconnected", "error": str(exc)}
        redis_ok = False

    try:
        encoder = get_embedding_encoder(settings)
        encoder._load_model()
        services["embedding_model"] = {"status": "loaded"}
        model_ok = True
    except Exception as exc:
        services["embedding_model"] = {"status": "not_loaded", "error": str(exc)}
        model_ok = False

    if not pg_ok:
        overall = "unhealthy"
    elif all((neo4j_ok, es_ok, faiss_ok, redis_ok, model_ok)):
        overall = "healthy"
    else:
        overall = "degraded"

    return {"status": overall, "services": services}
