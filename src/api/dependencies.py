"""FastAPI dependency injection helpers."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings, get_settings
from src.db.database import get_db
from src.embeddings.encoder import EmbeddingEncoder, get_encoder
from src.embeddings.faiss_manager import FAISSManager
from src.embeddings.faiss_manager import DIMENSIONS
from src.knowledge_graph.neo4j_client import Neo4jClient

_faiss_manager: Optional[FAISSManager] = None
_neo4j_client: Optional[Neo4jClient] = None


def get_settings_dep() -> Settings:
    """Return application settings."""
    return get_settings()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield async database session."""
    async for session in get_db():
        yield session


def get_faiss_manager(settings: Settings | None = None) -> FAISSManager:
    """Return shared FAISS manager instance."""
    global _faiss_manager
    if _faiss_manager is None:
        mgr = FAISSManager(settings or get_settings())
        for dimension in DIMENSIONS:
            try:
                mgr._ensure_loaded(dimension)
            except Exception:
                pass
        _faiss_manager = mgr
    return _faiss_manager


def get_embedding_encoder(settings: Settings | None = None) -> EmbeddingEncoder:
    """Return shared embedding encoder."""
    return get_encoder(settings)


def get_neo4j(settings: Settings | None = None) -> Neo4jClient:
    """Return shared Neo4j client."""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client
