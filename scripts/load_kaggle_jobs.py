"""Load LinkedIn job postings Kaggle dataset into PostgreSQL."""

import argparse
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import select

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))
import scripts._bootstrap  # noqa: F401

from config.logging import setup_logging
from config.settings import get_settings
from src.db.models import Job
from src.db.sync_database import get_sync_session

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = ".kaggle_loader_checkpoint"


def parse_salary(value: Any) -> tuple[Optional[int], Optional[int]]:
    """Parse salary fields from numeric or string values."""
    if pd.isna(value):
        return None, None
    try:
        amount = int(float(value))
    except (TypeError, ValueError):
        return None, None
    if amount < 0 or amount > 10_000_000:
        return None, None
    return amount, amount


def parse_salary_row(row: pd.Series) -> tuple[Optional[int], Optional[int]]:
    """Extract min/max salary with med_salary fallback."""
    min_sal, _ = parse_salary(row.get("min_salary"))
    max_sal, _ = parse_salary(row.get("max_salary"))
    med_sal, _ = parse_salary(row.get("med_salary"))
    if min_sal is None and max_sal is None and med_sal is not None:
        return med_sal, med_sal
    if min_sal is None and med_sal is not None:
        min_sal = med_sal
    if max_sal is None and med_sal is not None:
        max_sal = med_sal
    return min_sal, max_sal


def infer_remote_type(row: pd.Series) -> Optional[str]:
    """Infer remote type from location and description."""
    location = str(row.get("location", "") or "").lower()
    description = str(row.get("description", "") or "").lower()
    if row.get("remote_allowed") in (True, "True", "true", 1):
        return "remote"
    if "remote" in location or "remote" in description:
        return "remote"
    if "hybrid" in description:
        return "hybrid"
    if "on-site" in description or "onsite" in description:
        return "onsite"
    return None


def infer_experience_level(title: str) -> Optional[str]:
    """Infer experience level from job title."""
    title_lower = title.lower()
    if any(k in title_lower for k in ("junior", "entry", "graduate")):
        return "entry"
    if any(k in title_lower for k in ("senior", "sr.", "sr ")):
        return "senior"
    if any(k in title_lower for k in ("lead", "principal", "staff")):
        return "lead"
    if "director" in title_lower or "executive" in title_lower:
        return "executive"
    if "mid" in title_lower:
        return "mid"
    return None


def clean_description(text: str, max_len: int = 50000) -> str:
    """Clean job description text."""
    if not text or pd.isna(text):
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_len]


def load_companies(path: Path) -> dict[Any, str]:
    """Load company_id to name mapping."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "company_id" not in df.columns:
        return {}
    name_col = "name" if "name" in df.columns else "company"
    return dict(zip(df["company_id"], df[name_col], strict=False))


def row_to_job(row: pd.Series, companies: dict[Any, str]) -> Optional[dict[str, Any]]:
    """Map a dataframe row to Job fields."""
    title = str(row.get("title", "") or "").strip()
    description = clean_description(row.get("description", ""))
    if not title or not description:
        return None

    company = str(row.get("company", "") or row.get("company_name", "") or "").strip()
    if company.lower() == "nan":
        company = ""
    if not company and "company_id" in row.index:
        company = str(companies.get(row.get("company_id"), "") or "")

    salary_min, salary_max = parse_salary_row(row)
    posted = row.get("original_listed_time") or row.get("listed_time")
    posted_date = None
    if posted is not None and not pd.isna(posted):
        try:
            ts = float(posted)
            if ts > 1e12:
                posted_date = pd.to_datetime(ts, unit="ms", utc=True).to_pydatetime()
            elif ts > 1e9:
                posted_date = pd.to_datetime(ts, unit="s", utc=True).to_pydatetime()
            else:
                posted_date = pd.to_datetime(posted, utc=True).to_pydatetime()
        except (TypeError, ValueError):
            posted_date = pd.to_datetime(posted, utc=True, errors="coerce")
            if pd.isna(posted_date):
                posted_date = None
            else:
                posted_date = posted_date.to_pydatetime()

    skills_raw = row.get("skills_desc")
    skills_extracted = None
    if skills_raw is not None and not pd.isna(skills_raw):
        skills_extracted = [str(skills_raw)]

    return {
        "title": title,
        "company": company or "Unknown",
        "description": description,
        "location": str(row.get("location", "") or "") or None,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "remote_type": infer_remote_type(row),
        "experience_level": infer_experience_level(title),
        "skills_extracted": skills_extracted,
        "source_url": str(row.get("job_posting_url", "") or "") or None,
        "source_platform": "kaggle-linkedin",
        "posted_date": posted_date,
    }


def dedupe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by company+title keeping latest posting."""
    if "posted_date" not in df.columns:
        df = df.copy()
        df["posted_date"] = pd.to_datetime(
            df.get("original_listed_time", df.get("listed_time")),
            errors="coerce",
            utc=True,
        )
    df = df.sort_values("posted_date", ascending=False, na_position="last")
    return df.drop_duplicates(subset=["company", "title"], keep="first")


def existing_pairs(session) -> set[tuple[str, str]]:
    """Load existing company+title pairs from database."""
    rows = session.execute(select(Job.company, Job.title)).all()
    return {(r[0], r[1]) for r in rows}


def save_checkpoint(path: Path, offset: int) -> None:
    """Save loader checkpoint offset."""
    path.write_text(str(offset), encoding="utf-8")


def load_checkpoint(path: Path) -> int:
    """Load checkpoint offset if present."""
    if not path.exists():
        return 0
    return int(path.read_text(encoding="utf-8").strip() or "0")


def sanitize_job_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize DataFrame-derived values for ORM insert."""
    clean = dict(record)
    for key in ("salary_min", "salary_max"):
        value = clean.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            clean[key] = None
        else:
            clean[key] = int(value)
    company = clean.get("company")
    if company is None or (isinstance(company, float) and pd.isna(company)):
        clean["company"] = "Unknown"
    elif str(company).strip().lower() == "nan":
        clean["company"] = "Unknown"
    return clean


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Load Kaggle LinkedIn jobs CSV")
    parser.add_argument("--file-path", help="Directory or CSV path")
    parser.add_argument("--batch-size", type=int, help="Insert batch size")
    parser.add_argument("--limit", type=int, help="Max rows to process")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.app.log_level)
    batch_size = args.batch_size or settings.retrieval.kaggle_batch_size

    base = Path(args.file_path or settings.paths.kaggle_jobs_path)
    csv_path = base / "job_postings.csv" if base.is_dir() else base
    if not csv_path.exists():
        logger.error("CSV not found: %s — download Kaggle dataset first", csv_path)
        sys.exit(1)

    companies_path = csv_path.parent / "company_details" / "companies.csv"
    if not companies_path.exists() and csv_path.parent.name != "sample_jobs":
        alt = csv_path.parent.parent / "company_details" / "companies.csv"
        companies_path = alt if alt.exists() else companies_path

    checkpoint_path = base / CHECKPOINT_NAME if base.is_dir() else csv_path.parent / CHECKPOINT_NAME
    start_offset = load_checkpoint(checkpoint_path)

    logger.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    total_read = len(df)
    logger.info("Read %s rows", total_read)

    companies = load_companies(companies_path)
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        mapped = row_to_job(row, companies)
        if mapped:
            records.append(mapped)

    logger.info("After cleaning: %s rows", len(records))
    deduped = dedupe_dataframe(pd.DataFrame(records))
    records = deduped.to_dict(orient="records")
    logger.info("After dedup: %s rows", len(records))

    if args.limit:
        records = records[: args.limit]

    if start_offset:
        records = records[start_offset:]
        logger.info("Resuming from offset %s", start_offset)

    if args.dry_run:
        logger.info("Dry run: would insert %s rows", len(records))
        return

    inserted = 0
    skipped = 0
    errors = 0
    start = time.time()

    with get_sync_session() as session:
        existing = existing_pairs(session)
        batch: list[Job] = []
        for idx, record in enumerate(records):
            key = (record["company"], record["title"])
            if key in existing:
                skipped += 1
                continue
            batch.append(Job(**sanitize_job_record(record)))
            existing.add(key)
            if len(batch) >= batch_size:
                try:
                    session.add_all(batch)
                    session.flush()
                    inserted += len(batch)
                    batch = []
                    save_checkpoint(checkpoint_path, start_offset + idx + 1)
                except Exception as exc:
                    errors += len(batch)
                    logger.error("Batch insert failed: %s", exc)
                    session.rollback()
                    batch = []
        if batch:
            session.add_all(batch)
            session.flush()
            inserted += len(batch)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    elapsed = time.time() - start
    logger.info(
        "Load complete: inserted=%s skipped=%s errors=%s elapsed=%.1fs",
        inserted,
        skipped,
        errors,
        elapsed,
    )


if __name__ == "__main__":
    main()
