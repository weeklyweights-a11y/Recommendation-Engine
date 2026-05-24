"""Run the full recommendation pipeline for a candidate."""

from __future__ import annotations

import argparse
import logging
import sys
from uuid import UUID

from config.settings import get_settings
from src.db.sync_database import get_sync_session
from src.matching.recommendation_pipeline import run_recommendation_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run recommendation pipeline demo")
    parser.add_argument("--candidate-id", required=True, help="Candidate UUID")
    parser.add_argument("--refresh", action="store_true", help="Bypass cache")
    args = parser.parse_args()

    candidate_id = UUID(args.candidate_id)
    settings = get_settings()

    with get_sync_session() as session:
        result = run_recommendation_pipeline(
            candidate_id,
            refresh=args.refresh,
            session=session,
            settings=settings,
        )

    logger.info("From cache: %s", result.from_cache)
    if result.stats.filter_funnel:
        funnel = result.stats.filter_funnel
        logger.info(
            "Filter funnel: initial=%s final=%s restrictive=%s",
            funnel.initial_count,
            funnel.final_count,
            funnel.most_restrictive_filter,
        )
    if result.stats.warnings:
        logger.info("Warnings: %s", result.stats.warnings)
    if result.stats.timing_ms:
        t = result.stats.timing_ms
        logger.info(
            "Timing ms: filter=%.0f retrieval=%.0f rerank=%.0f explain=%.0f total=%.0f",
            t.hard_filter_ms,
            t.retrieval_ms,
            t.rerank_ms,
            t.explain_ms,
            t.total_ms,
        )

    for item in result.ranked_jobs[:10]:
        logger.info(
            "rank=%s score=%.3f section=%s job=%s",
            item.rank,
            item.match_score,
            item.feed_section,
            item.job_id,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
