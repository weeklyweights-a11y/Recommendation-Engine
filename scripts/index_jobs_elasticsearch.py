"""Bulk index PostgreSQL jobs into Elasticsearch for BM25 search."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sqlalchemy import func, select
from tqdm import tqdm

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))
import scripts._bootstrap  # noqa: F401

from config.logging import setup_logging
from config.settings import get_settings
from src.db.models import Job
from src.db.sync_database import get_sync_session
from src.matching.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "job_id": {"type": "keyword"},
            "title": {"type": "text"},
            "description": {"type": "text"},
            "company": {"type": "text"},
            "location": {"type": "keyword"},
            "remote_type": {"type": "keyword"},
            "experience_level": {"type": "keyword"},
            "industry": {"type": "keyword"},
            "skills_extracted": {"type": "text"},
            "posted_date": {"type": "date"},
        },
    },
}


def _skills_text(job: Job) -> str:
    """Build searchable skills text from job fields."""
    if job.skills_extracted:
        if isinstance(job.skills_extracted, list):
            return " ".join(str(s) for s in job.skills_extracted)
        return str(job.skills_extracted)
    return (job.description or "")[:2000]


def job_to_doc(job: Job) -> dict[str, Any]:
    """Map ORM job to Elasticsearch document."""
    return {
        "job_id": str(job.id),
        "title": job.title,
        "description": job.description,
        "company": job.company,
        "location": job.location,
        "remote_type": job.remote_type,
        "experience_level": job.experience_level,
        "industry": job.industry,
        "skills_extracted": _skills_text(job),
        "posted_date": job.posted_date.isoformat() if job.posted_date else None,
    }


def load_job_documents(limit: int | None) -> list[tuple[str, dict[str, Any]]]:
    """Load job documents from PostgreSQL while session is active."""
    with get_sync_session() as session:
        stmt = select(Job).order_by(Job.created_at)
        if limit:
            stmt = stmt.limit(limit)
        return [(str(job.id), job_to_doc(job)) for job in session.scalars(stmt)]


def recreate_index(client: Elasticsearch, index_name: str) -> None:
    """Delete and recreate index with mapping."""
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name, mappings=INDEX_MAPPING["mappings"])


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Index jobs into Elasticsearch")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.app.log_level)
    index_name = settings.elasticsearch.es_index_name
    batch_size = settings.retrieval.es_index_batch_size

    client = Elasticsearch(settings.elasticsearch.elasticsearch_url)
    if not client.ping():
        logger.error("Elasticsearch is not reachable")
        sys.exit(1)

    with get_sync_session() as session:
        pg_count = session.scalar(select(func.count()).select_from(Job)) or 0

    if args.recreate:
        logger.info("Recreating index %s", index_name)
        recreate_index(client, index_name)

    job_docs = load_job_documents(args.limit)
    logger.info("Indexing %s jobs (postgres total: %s)", len(job_docs), pg_count)

    if job_docs:
        def actions() -> Iterator[dict[str, Any]]:
            for job_id, source in tqdm(job_docs, desc="Indexing jobs"):
                yield {
                    "_index": index_name,
                    "_id": job_id,
                    "_source": source,
                }

        success, _ = bulk(client, actions(), chunk_size=batch_size, request_timeout=120)
        logger.info("Indexed %s documents", success)

    es_count = client.count(index=index_name)["count"]
    logger.info("Elasticsearch document count: %s (postgres: %s)", es_count, pg_count)

    retriever = BM25Retriever(settings)
    sample = retriever.retrieve("machine learning engineer", top_k=5)
    for item in sample:
        logger.info("Sample hit: %s @ %s (score=%.3f)", item.title, item.company, item.score)


if __name__ == "__main__":
    main()
