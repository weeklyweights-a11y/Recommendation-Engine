"""Batch-embed jobs in PostgreSQL and build FAISS indexes."""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

from sqlalchemy import func, select
from tqdm import tqdm

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))
import scripts._bootstrap  # noqa: F401

from config.logging import setup_logging
from config.settings import get_settings
from src.db.models import Job
from src.db.sync_database import get_sync_session
from src.embeddings.encoder import serialize_embedding
from src.embeddings.faiss_manager import DIMENSIONS, FAISSManager
from src.embeddings.job_embedder import embed_job_record

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed jobs and build FAISS indexes")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--llm-batch-size", type=int, default=None)
    parser.add_argument("--use-llm", action="store_true", help="Use Gemini Flash for all jobs")
    parser.add_argument("--llm-sample", type=int, default=None, help="Random N jobs use LLM")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild-index", action="store_true", help="Only rebuild FAISS from PG")
    parser.add_argument(
        "--dimension",
        choices=list(DIMENSIONS),
        default=None,
        help="Re-embed a single dimension only",
    )
    return parser.parse_args()


def _should_use_llm(job_id: str, args: argparse.Namespace, llm_sample_ids: set[str]) -> bool:
    if args.use_llm:
        return True
    return str(job_id) in llm_sample_ids


def _embed_one_job(
    job: Job,
    *,
    use_llm: bool,
    dimension: Optional[str],
    settings,
) -> None:
    """Embed a single job and update ORM fields in memory."""
    if dimension:
        from src.embeddings.job_embedder import build_skills_extracted_for_job, embed_job

        fields, linked, skills_json = build_skills_extracted_for_job(
            job, use_llm=use_llm, settings=settings,
        )
        embeddings = embed_job(fields, job.description or "", linked, settings=settings)
        setattr(job, f"embedding_{dimension}", serialize_embedding(getattr(embeddings, dimension)))
        job.skills_extracted = skills_json
        return

    embeddings, skills_json = embed_job_record(job, use_llm=use_llm, settings=settings)
    job.embedding_skill = serialize_embedding(embeddings.skill)
    job.embedding_domain = serialize_embedding(embeddings.domain)
    job.embedding_role = serialize_embedding(embeddings.role)
    job.embedding_environment = serialize_embedding(embeddings.environment)
    job.skills_extracted = skills_json


def main() -> None:
    setup_logging()
    args = _parse_args()
    settings = get_settings()
    batch_size = args.batch_size or settings.job_embedding.job_embed_batch_size

    if args.rebuild_index:
        with get_sync_session() as session:
            manager = FAISSManager(settings)
            stats = manager.build_indexes(session)
            logger.info("FAISS rebuild complete: %s", stats)
        return

    llm_sample_ids: set[str] = set()
    if args.llm_sample and not args.use_llm:
        with get_sync_session() as session:
            ids = list(
                session.scalars(
                    select(Job.id).where(Job.is_embedded.is_(False)).limit(args.llm_sample * 3),
                ),
            )
            random.shuffle(ids)
            llm_sample_ids = {str(i) for i in ids[: args.llm_sample]}
            logger.info("LLM sample size: %s jobs", len(llm_sample_ids))

    with get_sync_session() as session:
        total_pending = session.scalar(
            select(func.count()).select_from(Job).where(Job.is_embedded.is_(False)),
        ) or 0
        if args.limit:
            total_pending = min(total_pending, args.limit)
        logger.info("Jobs to embed: %s", total_pending)

        processed = 0
        start = time.perf_counter()
        pbar = tqdm(total=total_pending, desc="Embedding jobs")

        while True:
            if args.limit and processed >= args.limit:
                break
            fetch_size = batch_size
            if args.limit:
                fetch_size = min(batch_size, args.limit - processed)

            jobs = list(
                session.scalars(
                    select(Job)
                    .where(Job.is_embedded.is_(False))
                    .order_by(Job.created_at)
                    .limit(fetch_size),
                ),
            )
            if not jobs:
                break

            batch_start = time.perf_counter()
            for job in jobs:
                use_llm = _should_use_llm(str(job.id), args, llm_sample_ids)
                _embed_one_job(job, use_llm=use_llm, dimension=args.dimension, settings=settings)
                if not args.dimension:
                    job.is_embedded = True
                processed += 1
                pbar.update(1)

            session.commit()
            elapsed = time.perf_counter() - batch_start
            rate = len(jobs) / elapsed if elapsed > 0 else 0
            logger.info(
                "Embedded batch %s jobs, %.2f jobs/sec",
                len(jobs),
                rate,
            )

        pbar.close()
        wall = time.perf_counter() - start
        logger.info(
            "Embedding complete: %s jobs in %.1fs (avg %.3fs/job)",
            processed,
            wall,
            wall / processed if processed else 0,
        )

        if not args.dimension:
            manager = FAISSManager(settings)
            stats = manager.build_indexes(session)
            logger.info("FAISS index stats: %s", stats)
            try:
                from src.matching.graph_retriever import GraphRetriever

                GraphRetriever(settings).rebuild(session)
            except Exception as exc:
                logger.warning("Graph reverse index rebuild skipped: %s", exc)


if __name__ == "__main__":
    main()
