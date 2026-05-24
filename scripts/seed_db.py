"""Run database migrations and print table counts."""

import logging
import os
import subprocess
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))

import scripts._bootstrap  # noqa: F401

from config.logging import setup_logging
from config.settings import get_settings
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


def run_migrations() -> None:
    """Execute alembic upgrade head."""
    settings = get_settings()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["ALEMBIC_DATABASE_URL"] = settings.database.alembic_database_url
    env["DATABASE_URL"] = settings.database.database_url
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        logger.error("Alembic failed: %s", result.stderr)
        raise RuntimeError("Migration failed")
    logger.info("Migrations applied successfully")


def log_table_counts() -> None:
    """Log row counts for core tables."""
    settings = get_settings()
    engine = create_engine(settings.database.alembic_database_url)
    tables = ("jobs", "candidates", "recommendations", "feedback")
    with engine.connect() as conn:
        for table in tables:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            logger.info("Table %s: %s rows", table, count)


def main() -> None:
    """Run migrations and log counts."""
    settings = get_settings()
    setup_logging(settings.app.log_level)
    run_migrations()
    log_table_counts()


if __name__ == "__main__":
    main()
