"""End-to-end smoke test: health check, cached recommendations, optional refresh."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))
import scripts._bootstrap  # noqa: F401

import httpx


logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="API E2E smoke test")
    parser.add_argument("--candidate-id", required=True, help="Candidate UUID")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--refresh", action="store_true", help="Force pipeline refresh")
    parser.add_argument("--skip-server", action="store_true", help="Only hit health if server already up")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    base = args.base_url.rstrip("/")
    candidate_id = args.candidate_id

    with httpx.Client(timeout=600.0) as client:
        health = client.get(f"{base}/api/v1/health")
        health.raise_for_status()
        body = health.json()
        logger.info("Health status=%s", body.get("status"))
        if body.get("status") == "unhealthy":
            logger.error("Health unhealthy: %s", body.get("services"))
            return 1

        params = {"refresh": "true"} if args.refresh else {}
        rec = client.get(f"{base}/api/v1/recommendations/{candidate_id}", params=params)
        rec.raise_for_status()
        data = rec.json()
        total = data.get("pagination", {}).get("total", 0)
        warnings = (data.get("pipeline_stats") or {}).get("warnings", [])
        logger.info("Recommendations total=%s warnings=%s", total, warnings)
        if total == 0:
            logger.error("No recommendations returned")
            return 1
        first = data["recommendations"][0]
        logger.info(
            "Top match rank=%s score=%s section=%s job=%s",
            first.get("rank"),
            first.get("match_score"),
            first.get("feed_section"),
            (first.get("job") or {}).get("title"),
        )
        print(json.dumps({"total": total, "warnings": warnings, "top": first}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
