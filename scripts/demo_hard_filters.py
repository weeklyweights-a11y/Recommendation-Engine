"""Demo hard filters for a candidate email or default preferences."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))
import scripts._bootstrap  # noqa: F401

from config.logging import setup_logging
from src.api.schemas.candidate import CandidateProfile, MergedPreferences, PreferenceField
from src.db.sync_database import get_sync_session
from src.matching.hard_filters import HardFilter

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Demo hard job filters")
    parser.add_argument("--email", help="Candidate email in DB")
    args = parser.parse_args()

    with get_sync_session() as session:
        hf = HardFilter(session)
        if args.email:
            from sqlalchemy import select

            from src.db.models import Candidate

            candidate = session.scalar(select(Candidate).where(Candidate.email == args.email))
            if not candidate or not candidate.profile:
                raise SystemExit("Candidate not found or missing profile")
            profile = CandidateProfile.model_validate(candidate.profile)
            preferences = profile.preferences
        else:
            preferences = MergedPreferences(
                work_models=PreferenceField(value=["remote"], source="explicit"),
                salary_min=PreferenceField(value=100000, source="explicit"),
            )

        funnel = hf.get_filter_funnel(preferences)
        allowed = hf.filter_jobs(preferences)
        logger.info("Allowed jobs: %s", len(allowed))
        print(json.dumps(funnel.model_dump(), indent=2))


if __name__ == "__main__":
    main()
