"""Synchronous database session for CLI scripts."""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings

_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def get_sync_engine():
    """Return sync SQLAlchemy engine for batch loaders."""
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.database.alembic_database_url, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    return _engine


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    """Yield a sync ORM session."""
    get_sync_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
