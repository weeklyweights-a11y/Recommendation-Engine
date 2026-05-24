"""Demo hybrid fusion + rerank for a candidate."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))
import scripts._bootstrap  # noqa: F401

from config.logging import setup_logging
from config.settings import get_settings
from src.db.sync_database import get_sync_session
from src.matching.hybrid_pipeline import retrieve_hybrid
from src.matching.reranker import Reranker

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Demo rerank after hybrid retrieval")
    parser.add_argument("--email", required=True)
    parser.add_argument("--display", type=int, default=20)
    args = parser.parse_args()

    settings = get_settings()
    with get_sync_session() as session:
        fused = retrieve_hybrid(email=args.email, session=session, settings=settings)
        from sqlalchemy import select

        from src.db.models import Candidate

        candidate = session.scalar(select(Candidate).where(Candidate.email == args.email))
        if not candidate or not candidate.profile:
            raise SystemExit("Candidate missing")
        from src.api.schemas.candidate import CandidateProfile

        profile = CandidateProfile.model_validate(candidate.profile)
        reranker = Reranker(session, settings=settings)
        ranked = reranker.rerank(profile, fused, top_k=50)
        logger.info("Reranked %s jobs (fused input %s)", len(ranked), len(fused))
        for item in ranked[: args.display]:
            logger.info(
                "%s. %s @ %s score=%.3f skill=%.2f sem=%.2f section=%s",
                item.rank,
                getattr(item.job, "title", ""),
                getattr(item.job, "company", ""),
                item.match_score,
                item.factor_scores.get("skill_fit", 0),
                item.factor_scores.get("semantic_similarity", 0),
                item.feed_section,
            )


if __name__ == "__main__":
    main()
