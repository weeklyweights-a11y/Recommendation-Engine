"""End-to-end hybrid retrieval orchestrator."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from src.api.schemas.candidate import CandidateProfile
from src.db.candidate_repository import load_candidate_embeddings_vectors
from src.db.models import Candidate
from src.db.sync_database import get_sync_session
from src.embeddings.faiss_manager import FAISSManager
from src.embeddings.schemas import CandidateEmbeddings
from src.matching.bm25_retriever import BM25Retriever
from src.matching.graph_retriever import GraphRetriever
from src.matching.hybrid_fuser import HybridFuser
from src.matching.schemas import FusedResult, HybridTiming, RetrievalStats
from src.matching.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


def _build_bm25_query(profile: CandidateProfile) -> str:
    """Construct BM25 query text from candidate profile."""
    parts: list[str] = []
    if profile.summary:
        parts.append(profile.summary[:500])
    if profile.role_archetype:
        parts.append(profile.role_archetype)
    if profile.domains:
        parts.append(" ".join(profile.domains))
    target_roles = profile.preferences.target_roles.value
    if target_roles:
        parts.append(" ".join(target_roles))
    top_skills = sorted(profile.skills, key=lambda s: s.depth_score, reverse=True)[:15]
    parts.extend(s.name for s in top_skills if s.name)
    return " ".join(parts)


def _filter_allowed(results: list, allowed_job_ids: Optional[set[str]]) -> list:
    if allowed_job_ids is None:
        return results
    return [r for r in results if r.job_id in allowed_job_ids]


def retrieve_hybrid(
    *,
    candidate_id: Optional[UUID] = None,
    email: Optional[str] = None,
    profile: Optional[CandidateProfile] = None,
    embeddings: Optional[CandidateEmbeddings] = None,
    top_k: Optional[int] = None,
    allowed_job_ids: Optional[set[str]] = None,
    session: Optional[Session] = None,
    settings: Optional[Settings] = None,
) -> list[FusedResult]:
    """Run BM25 + vector + graph retrieval and fuse results."""
    cfg = settings or get_settings()
    k = top_k if top_k is not None else cfg.retrieval.hybrid_top_k

    def _run(sess: Session) -> list[FusedResult]:
        resolved_profile = profile
        resolved_embeddings = embeddings
        candidate: Optional[Candidate] = None
        if candidate_id is not None:
            candidate = sess.get(Candidate, candidate_id)
        elif email:
            candidate = sess.scalar(select(Candidate).where(Candidate.email == email))

        if resolved_profile is None and candidate and candidate.profile:
            resolved_profile = CandidateProfile.model_validate(candidate.profile)

        if resolved_profile is None:
            raise ValueError("Candidate profile required for hybrid retrieval")

        if resolved_embeddings is None and candidate:
            vectors = load_candidate_embeddings_vectors(candidate)
            if vectors:
                resolved_embeddings = CandidateEmbeddings(**vectors)

        if resolved_embeddings is None:
            raise ValueError("Candidate embeddings required for hybrid retrieval")

        bm25 = BM25Retriever(cfg)
        faiss_manager = FAISSManager(cfg)
        vector_retriever = VectorRetriever(faiss_manager, cfg)
        graph_retriever = GraphRetriever(cfg)
        fuser = HybridFuser(cfg)

        query_text = _build_bm25_query(resolved_profile)

        try:
            bm25_results = bm25.retrieve(query_text, top_k=k)
        except Exception as exc:
            logger.error("BM25 retrieval failed: %s", exc)
            bm25_results = []
        bm25_results = _filter_allowed(bm25_results, allowed_job_ids)

        try:
            vector_results = vector_retriever.retrieve(
                resolved_embeddings,
                top_k=k,
                allowed_job_ids=allowed_job_ids,
            )
        except Exception as exc:
            logger.error("Vector retrieval failed: %s", exc)
            vector_results = []

        try:
            graph_results = graph_retriever.retrieve(
                resolved_profile.esco_linked_skills,
                top_k=k,
                allowed_job_ids=allowed_job_ids,
                session=sess,
                profile=resolved_profile,
            )
        except Exception as exc:
            logger.error("Graph retrieval failed: %s", exc)
            graph_results = []

        return fuser.fuse(bm25_results, vector_results, graph_results, top_k=k)

    if session is not None:
        return _run(session)
    with get_sync_session() as sess:
        return _run(sess)


def retrieve_hybrid_parallel(
    *,
    candidate_id: Optional[UUID] = None,
    email: Optional[str] = None,
    profile: Optional[CandidateProfile] = None,
    embeddings: Optional[CandidateEmbeddings] = None,
    top_k: Optional[int] = None,
    allowed_job_ids: Optional[set[str]] = None,
    session: Optional[Session] = None,
    settings: Optional[Settings] = None,
) -> tuple[list[FusedResult], RetrievalStats, HybridTiming]:
    """Run BM25, vector, and graph retrieval in parallel, then fuse."""
    cfg = settings or get_settings()
    k = top_k if top_k is not None else cfg.retrieval.hybrid_top_k

    def _run(sess: Session) -> tuple[list[FusedResult], RetrievalStats, HybridTiming]:
        resolved_profile = profile
        resolved_embeddings = embeddings
        candidate: Optional[Candidate] = None
        if candidate_id is not None:
            candidate = sess.get(Candidate, candidate_id)
        elif email:
            candidate = sess.scalar(select(Candidate).where(Candidate.email == email))

        if resolved_profile is None and candidate and candidate.profile:
            resolved_profile = CandidateProfile.model_validate(candidate.profile)
        if resolved_profile is None:
            raise ValueError("Candidate profile required for hybrid retrieval")
        if resolved_embeddings is None and candidate:
            vectors = load_candidate_embeddings_vectors(candidate)
            if vectors:
                resolved_embeddings = CandidateEmbeddings(**vectors)
        if resolved_embeddings is None:
            raise ValueError("Candidate embeddings required for hybrid retrieval")

        bm25 = BM25Retriever(cfg)
        faiss_manager = FAISSManager(cfg)
        vector_retriever = VectorRetriever(faiss_manager, cfg)
        graph_retriever = GraphRetriever(cfg)
        fuser = HybridFuser(cfg)
        query_text = _build_bm25_query(resolved_profile)

        def _bm25() -> tuple[list, float]:
            t0 = time.perf_counter()
            try:
                results = bm25.retrieve(
                    query_text,
                    top_k=k,
                    allowed_job_ids=allowed_job_ids,
                )
                return _filter_allowed(results, allowed_job_ids), (time.perf_counter() - t0) * 1000
            except Exception as exc:
                logger.error("BM25 retrieval failed: %s", exc)
                return [], (time.perf_counter() - t0) * 1000

        def _vector() -> tuple[list, float]:
            t0 = time.perf_counter()
            try:
                results = vector_retriever.retrieve(
                    resolved_embeddings,
                    top_k=k,
                    allowed_job_ids=allowed_job_ids,
                )
                return results, (time.perf_counter() - t0) * 1000
            except Exception as exc:
                logger.error("Vector retrieval failed: %s", exc)
                return [], (time.perf_counter() - t0) * 1000

        def _graph() -> tuple[list, float]:
            t0 = time.perf_counter()
            try:
                results = graph_retriever.retrieve(
                    resolved_profile.esco_linked_skills,
                    top_k=k,
                    allowed_job_ids=allowed_job_ids,
                    session=sess,
                    profile=resolved_profile,
                )
                return results, (time.perf_counter() - t0) * 1000
            except Exception as exc:
                logger.error("Graph retrieval failed: %s", exc)
                return [], (time.perf_counter() - t0) * 1000

        with ThreadPoolExecutor(max_workers=3) as pool:
            f_bm25 = pool.submit(_bm25)
            f_vector = pool.submit(_vector)
            f_graph = pool.submit(_graph)
            bm25_results, bm25_ms = f_bm25.result()
            vector_results, vector_ms = f_vector.result()
            graph_results, graph_ms = f_graph.result()

        t_fuse = time.perf_counter()
        stats = fuser.analyze_retrieval_diversity(bm25_results, vector_results, graph_results)
        fused = fuser.fuse(bm25_results, vector_results, graph_results, top_k=k)
        fusion_ms = (time.perf_counter() - t_fuse) * 1000
        hybrid_timing = HybridTiming(
            bm25_ms=bm25_ms,
            vector_ms=vector_ms,
            graph_ms=graph_ms,
            fusion_ms=fusion_ms,
        )
        return fused, stats, hybrid_timing

    if session is not None:
        return _run(session)
    with get_sync_session() as sess:
        return _run(sess)


async def retrieve_hybrid_parallel_async(
    **kwargs: object,
) -> tuple[list[FusedResult], RetrievalStats, HybridTiming]:
    """Async wrapper executing parallel hybrid retrieval in a thread pool."""
    return await asyncio.to_thread(retrieve_hybrid_parallel, **kwargs)
