"""Configurable job listing scraper."""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import httpx
import yaml
from bs4 import BeautifulSoup
from sqlalchemy import select
from tenacity import retry, stop_after_attempt, wait_exponential

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))
import scripts._bootstrap  # noqa: F401

from config.logging import setup_logging
from config.settings import get_settings
from src.db.models import Job
from src.db.sync_database import get_sync_session

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict[str, Any]:
    """Load scraper YAML configuration."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def clean_text(text: Optional[str]) -> str:
    """Normalize extracted HTML text."""
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


def extract_field(soup: BeautifulSoup, selector: str, base_url: str, attr: Optional[str]) -> str:
    """Extract a single field using CSS selector."""
    element = soup.select_one(selector)
    if not element:
        return ""
    if attr:
        value = element.get(attr, "")
        if attr == "href" and value:
            return urljoin(base_url, value)
        return str(value)
    return clean_text(element.get_text())


def infer_remote_type(location: str, description: str) -> Optional[str]:
    """Infer remote type from text."""
    combined = f"{location} {description}".lower()
    if "remote" in combined:
        return "remote"
    if "hybrid" in combined:
        return "hybrid"
    if "on-site" in combined or "onsite" in combined:
        return "onsite"
    return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_page(client: httpx.Client, url: str) -> str:
    """Fetch page HTML with retries."""
    response = client.get(url, timeout=30.0)
    response.raise_for_status()
    return response.text


def parse_listing(
    html: str,
    source: dict[str, Any],
    page_url: str,
) -> list[dict[str, Any]]:
    """Parse job listings from a page."""
    soup = BeautifulSoup(html, "lxml")
    selectors = source["selectors"]
    cards = soup.select(selectors["job_card"])
    listings: list[dict[str, Any]] = []
    for card in cards:
        card_soup = BeautifulSoup(str(card), "lxml")
        title = extract_field(card_soup, selectors["title"], page_url, None)
        if not title:
            continue
        description = extract_field(card_soup, selectors["description"], page_url, None)
        company = extract_field(card_soup, selectors.get("company", ""), page_url, None)
        location = extract_field(card_soup, selectors.get("location", ""), page_url, None)
        link = extract_field(card_soup, selectors.get("link", "a"), page_url, "href")
        listings.append(
            {
                "title": title,
                "company": company or "Unknown",
                "description": description or title,
                "location": location or None,
                "remote_type": infer_remote_type(location, description),
                "source_url": link or page_url,
                "source_platform": source["name"],
                "posted_date": datetime.utcnow(),
            },
        )
    return listings


def job_exists(session, company: str, title: str, source_url: Optional[str]) -> bool:
    """Check if job already exists."""
    stmt = select(Job.id).where(
        Job.company == company,
        Job.title == title,
        Job.source_url == source_url,
    )
    return session.execute(stmt).first() is not None


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Append records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")


def insert_jobs(session, records: list[dict[str, Any]]) -> tuple[int, int]:
    """Insert jobs into PostgreSQL."""
    inserted = 0
    skipped = 0
    for record in records:
        if job_exists(session, record["company"], record["title"], record.get("source_url")):
            skipped += 1
            continue
        session.add(Job(**record))
        inserted += 1
    return inserted, skipped


def scrape_source(
    source: dict[str, Any],
    client: httpx.Client,
    limit: Optional[int],
    output_path: Optional[Path],
    insert_db: bool,
    dry_run: bool,
) -> dict[str, int]:
    """Scrape a single configured source."""
    stats = {"pages": 0, "found": 0, "inserted": 0, "skipped": 0, "errors": 0}
    if not source.get("enabled", True):
        return stats

    base_url = source["base_url"].rstrip("/")
    listing_path = source.get("listing_path", "")
    start_url = urljoin(base_url + "/", listing_path.lstrip("/"))
    delay = float(source.get("rate_limit_seconds", 2.0))
    max_pages = int(source.get("pagination", {}).get("max_pages", 10))

    all_records: list[dict[str, Any]] = []

    for page_num in range(1, max_pages + 1):
        if limit and stats["found"] >= limit:
            break
        page_url = start_url
        pagination = source.get("pagination", {})
        if pagination.get("type") == "query_param" and page_num > 1:
            param = pagination.get("param", "page")
            sep = "&" if "?" in start_url else "?"
            page_url = f"{start_url}{sep}{param}={page_num}"

        try:
            html = fetch_page(client, page_url)
            records = parse_listing(html, source, page_url)
            stats["pages"] += 1
            stats["found"] += len(records)
            if limit:
                records = records[: max(0, limit - len(all_records))]
            all_records.extend(records)
            logger.info("Page %s: %s listings from %s", page_num, len(records), page_url)
        except Exception as exc:
            stats["errors"] += 1
            logger.error("Failed page %s: %s", page_url, exc)
        time.sleep(delay)

    if dry_run:
        logger.info("Dry run: would process %s listings", len(all_records))
        return stats

    if output_path and all_records:
        write_jsonl(output_path, all_records)

    if insert_db and all_records:
        with get_sync_session() as session:
            inserted, skipped = insert_jobs(session, all_records)
            stats["inserted"] = inserted
            stats["skipped"] = skipped

    return stats


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Scrape job listings")
    parser.add_argument("--source", help="Run only this source name")
    parser.add_argument("--limit", type=int, help="Max listings per source")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", help="JSONL output path")
    parser.add_argument("--insert-db", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.app.log_level)
    config = load_config(settings.paths.scraper_config_path)

    mode = settings.scraper.scraper_default_mode
    output_path = Path(args.output) if args.output else Path(settings.paths.scraper_output_path)
    insert_db = args.insert_db or mode in ("db", "both")
    write_jsonl_file = args.output or mode in ("jsonl", "both")

    sources = config.get("sources", [])
    if args.source:
        sources = [s for s in sources if s.get("name") == args.source]

    if not sources:
        logger.warning("No enabled sources in scraper config")
        return

    totals = {"pages": 0, "found": 0, "inserted": 0, "skipped": 0, "errors": 0}
    start = time.time()

    with httpx.Client(headers={"User-Agent": config.get("defaults", {}).get("user_agent", "")}) as client:
        for source in sources:
            stats = scrape_source(
                source,
                client,
                args.limit,
                output_path if write_jsonl_file else None,
                insert_db,
                args.dry_run,
            )
            for key in totals:
                totals[key] += stats[key]

    elapsed = time.time() - start
    logger.info(
        "Scrape complete: pages=%s found=%s inserted=%s skipped=%s errors=%s elapsed=%.1fs",
        totals["pages"],
        totals["found"],
        totals["inserted"],
        totals["skipped"],
        totals["errors"],
        elapsed,
    )


if __name__ == "__main__":
    main()
