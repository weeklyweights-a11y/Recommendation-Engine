"""FAISS multi-vector job retrieval."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from config.settings import Settings, get_settings
from src.api.schemas.recommendation import ScoredJob
from src.embeddings.faiss_manager import DIMENSIONS, FAISSManager
from src.embeddings.schemas import CandidateEmbeddings

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Retrieve jobs by searching four FAISS embedding dimensions."""

    def __init__(
        self,
        faiss_manager: FAISSManager,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize with FAISS manager and settings."""
        self._faiss = faiss_manager
        self._settings = settings or get_settings()

    def _default_weights(self) -> dict[str, float]:
        r = self._settings.retrieval
        return {
            "skill": r.vector_skill_weight,
            "domain": r.vector_domain_weight,
            "role": r.vector_role_weight,
            "environment": r.vector_environment_weight,
        }

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            return {d: 1.0 / len(DIMENSIONS) for d in DIMENSIONS}
        return {k: v / total for k, v in weights.items()}

    def _is_zero_vector(self, vector: np.ndarray) -> bool:
        eps = self._settings.retrieval.vector_zero_norm_epsilon
        return float(np.linalg.norm(vector)) < eps

    def retrieve(
        self,
        candidate_embeddings: CandidateEmbeddings,
        top_k: Optional[int] = None,
        dimension_weights: Optional[dict[str, float]] = None,
        allowed_job_ids: Optional[set[str]] = None,
    ) -> list[ScoredJob]:
        """Search all dimensions and return fused ranked ScoredJob list."""
        k = top_k if top_k is not None else self._settings.retrieval.hybrid_top_k
        weights = self._normalize_weights(dimension_weights or self._default_weights())
        overfetch = self._settings.retrieval.vector_overfetch_multiplier
        search_k = k * overfetch

        vectors = {
            "skill": candidate_embeddings.skill,
            "domain": candidate_embeddings.domain,
            "role": candidate_embeddings.role,
            "environment": candidate_embeddings.environment,
        }

        active_weights: dict[str, float] = {}
        per_dim_scores: dict[str, dict[str, float]] = {}
        dimensions_searched = 0

        for dimension in DIMENSIONS:
            vector = vectors[dimension]
            if self._is_zero_vector(vector):
                logger.warning("Skipping zero vector for dimension %s", dimension)
                continue
            try:
                hits = self._faiss.search(dimension, vector, top_k=search_k)
            except Exception as exc:
                logger.error("FAISS search failed for %s: %s", dimension, exc)
                continue
            if not hits:
                continue
            dimensions_searched += 1
            active_weights[dimension] = weights[dimension]
            max_score = max(score for _, score in hits) or 1.0
            per_dim_scores[dimension] = {
                job_id: (score / max_score if max_score > 0 else 0.0)
                for job_id, score in hits
            }

        if dimensions_searched == 0:
            raise RuntimeError("All FAISS dimensions failed or were zero vectors")

        aw_total = sum(active_weights.values())
        active_weights = {d: w / aw_total for d, w in active_weights.items()}

        fused: dict[str, float] = {}
        dim_breakdown: dict[str, dict[str, float]] = {d: {} for d in DIMENSIONS}
        for dimension, scores in per_dim_scores.items():
            w = active_weights.get(dimension, 0.0)
            for job_id, norm_score in scores.items():
                if allowed_job_ids is not None and job_id not in allowed_job_ids:
                    continue
                fused[job_id] = fused.get(job_id, 0.0) + w * norm_score
                dim_breakdown[dimension][job_id] = norm_score

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        results: list[ScoredJob] = []
        for job_id, score in ranked:
            dim_scores = {
                d: dim_breakdown[d].get(job_id, 0.0)
                for d in DIMENSIONS
                if dim_breakdown[d]
            }
            results.append(
                ScoredJob(
                    job_id=job_id,
                    score=score,
                    source="vector",
                    dimension_scores=dim_scores,
                ),
            )
        return results

    def retrieve_single_dimension(
        self,
        candidate_embedding: np.ndarray,
        dimension: str,
        top_k: int = 100,
    ) -> list[ScoredJob]:
        """Search a single FAISS dimension."""
        hits = self._faiss.search(dimension, candidate_embedding, top_k=top_k)
        max_score = max((s for _, s in hits), default=1.0) or 1.0
        return [
            ScoredJob(
                job_id=job_id,
                score=score / max_score,
                source=f"vector_{dimension}",
                dimension_scores={dimension: score / max_score},
            )
            for job_id, score in hits
        ]
