"""CLI to build and optionally persist a candidate profile."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))
import scripts._bootstrap  # noqa: F401

from config.logging import setup_logging
from config.settings import get_settings
from src.api.schemas.candidate import CandidatePreferences
from src.db.sync_database import get_sync_session
from src.ingestion.profile_builder import build_and_save_profile, build_profile

logger = logging.getLogger(__name__)


async def _run(args: argparse.Namespace) -> None:
    preferences = None
    if args.target_role:
        preferences = CandidatePreferences(target_roles=[args.target_role])

    if args.save:
        with get_sync_session() as session:
            profile, embeddings = await build_and_save_profile(
                args.resume,
                github_username=args.github,
                preferences=preferences,
                session=session,
            )
        logger.info("Saved candidate profile for %s", profile.email or profile.name)
        logger.info(
            "Embeddings stored — skill norm=%.3f domain norm=%.3f",
            float(embeddings.skill @ embeddings.skill),
            float(embeddings.domain @ embeddings.domain),
        )
    else:
        profile, embeddings = await build_profile(
            args.resume,
            github_username=args.github,
            preferences=preferences,
        )
        logger.info("Built profile for %s with %s skills", profile.name, len(profile.skills))
        logger.info("Embedding shapes: skill=%s", embeddings.skill.shape)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Build candidate profile from resume")
    parser.add_argument("--resume", required=True, help="Path to resume PDF or DOCX")
    parser.add_argument("--github", default=None, help="GitHub username")
    parser.add_argument("--target-role", default=None, help="Explicit target role preference")
    parser.add_argument("--save", action="store_true", help="Persist profile to PostgreSQL")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.app.log_level)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
