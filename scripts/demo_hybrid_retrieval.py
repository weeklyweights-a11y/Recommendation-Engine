"""Demo hybrid retrieval for a candidate in PostgreSQL."""

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
from src.matching.hybrid_pipeline import retrieve_hybrid

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Demo hybrid job retrieval")
    parser.add_argument("--email", required=True, help="Candidate email in DB")
    parser.add_argument("--display", type=int, default=20, help="Rows to print")
    args = parser.parse_args()

    settings = get_settings()
    results = retrieve_hybrid(email=args.email, settings=settings)
    logger.info("Retrieved %s fused results (top_k=%s)", len(results), settings.retrieval.hybrid_top_k)

    for i, item in enumerate(results[: args.display], start=1):
        logger.info(
            "%s. job_id=%s fused=%.3f bm25=%.3f vector=%.3f graph=%.3f sources=%s",
            i,
            item.job_id,
            item.fused_score,
            item.bm25_score,
            item.vector_score,
            item.graph_score,
            item.sources,
        )


if __name__ == "__main__":
    main()
