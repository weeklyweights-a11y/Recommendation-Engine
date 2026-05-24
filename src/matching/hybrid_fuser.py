"""Fuse BM25, vector, and graph retrieval results."""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import Settings, get_settings
from src.api.schemas.recommendation import ScoredJob
from src.matching.schemas import FusedResult, RetrievalStats

logger = logging.getLogger(__name__)


class HybridFuser:
    """Combine ranked lists from three retrieval sources."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Load fusion weights and strategy from settings."""
        self._settings = settings or get_settings()

    def _default_weights(self) -> dict[str, float]:
        r = self._settings.retrieval
        return {
            "bm25": r.fusion_bm25_weight,
            "vector": r.fusion_vector_weight,
            "graph": r.fusion_graph_weight,
        }

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning("Fusion weights sum to %.3f, normalizing", total)
        if total <= 0:
            return {"bm25": 1 / 3, "vector": 1 / 3, "graph": 1 / 3}
        return {k: v / total for k, v in weights.items()}

    def _redistribute_weights(
        self,
        weights: dict[str, float],
        present: set[str],
    ) -> dict[str, float]:
        if not present:
            return weights
        if len(present) == 1:
            source = next(iter(present))
            return {k: (1.0 if k == source else 0.0) for k in weights}
        sub = {k: weights[k] for k in present}
        total = sum(sub.values())
        return {k: (sub[k] / total if k in present else 0.0) for k in weights}

    def analyze_retrieval_diversity(
        self,
        bm25_results: list[ScoredJob],
        vector_results: list[ScoredJob],
        graph_results: list[ScoredJob],
    ) -> RetrievalStats:
        """Compute overlap statistics across retrieval sources."""
        bm25_ids = {r.job_id for r in bm25_results}
        vector_ids = {r.job_id for r in vector_results}
        graph_ids = {r.job_id for r in graph_results}
        all_ids = bm25_ids | vector_ids | graph_ids

        all_three = len(bm25_ids & vector_ids & graph_ids)
        in_two = 0
        in_one = 0
        for job_id in all_ids:
            count = sum(
                job_id in source
                for source in (bm25_ids, vector_ids, graph_ids)
            )
            if count == 2:
                in_two += 1
            elif count == 1:
                in_one += 1

        return RetrievalStats(
            all_three=all_three,
            exactly_two=in_two,
            only_one=in_one,
            total_unique=len(all_ids),
        )

    def fuse(
        self,
        bm25_results: list[ScoredJob],
        vector_results: list[ScoredJob],
        graph_results: list[ScoredJob],
        top_k: Optional[int] = None,
        weights: Optional[dict[str, float]] = None,
        strategy: Optional[str] = None,
    ) -> list[FusedResult]:
        """Fuse three result lists into a single ranked list."""
        k = top_k if top_k is not None else self._settings.retrieval.hybrid_top_k
        w = self._normalize_weights(weights or self._default_weights())
        strat = strategy or self._settings.retrieval.fusion_strategy

        sources_present: set[str] = set()
        if bm25_results:
            sources_present.add("bm25")
        if vector_results:
            sources_present.add("vector")
        if graph_results:
            sources_present.add("graph")

        if not sources_present:
            return []

        if len(sources_present) == 1:
            only = next(iter(sources_present))
            logger.info("Single source fusion passthrough: %s", only)
            results_map = {
                "bm25": bm25_results,
                "vector": vector_results,
                "graph": graph_results,
            }
            return self._passthrough(results_map[only], only, k)

        w = self._redistribute_weights(w, sources_present)
        missing = {"bm25", "vector", "graph"} - sources_present
        if missing:
            logger.info("Missing retrieval sources: %s; redistributed weights", missing)

        if strat == "weighted_sum":
            fused = self._fuse_weighted_sum(bm25_results, vector_results, graph_results, w)
        else:
            fused = self._fuse_rrf(bm25_results, vector_results, graph_results, w)

        stats = self.analyze_retrieval_diversity(bm25_results, vector_results, graph_results)
        logger.info(
            "Fusion diversity — all_three=%s exactly_two=%s only_one=%s unique=%s",
            stats.all_three,
            stats.exactly_two,
            stats.only_one,
            stats.total_unique,
        )
        return fused[:k]

    def _passthrough(self, results: list[ScoredJob], source: str, top_k: int) -> list[FusedResult]:
        return [
            FusedResult(
                job_id=r.job_id,
                fused_score=r.score,
                bm25_score=r.score if source == "bm25" else 0.0,
                vector_score=r.score if source == "vector" else 0.0,
                graph_score=r.score if source == "graph" else 0.0,
                sources=[source],
                vector_dimension_scores=r.dimension_scores,
                graph_matched_skills=r.matched_skills,
            )
            for r in results[:top_k]
        ]

    def _score_maps(
        self,
        bm25: list[ScoredJob],
        vector: list[ScoredJob],
        graph: list[ScoredJob],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, Any]]:
        bm25_map = {r.job_id: r.score for r in bm25}
        vector_map = {r.job_id: r.score for r in vector}
        graph_map = {r.job_id: r.score for r in graph}
        meta: dict[str, Any] = {}
        for r in vector:
            if r.dimension_scores:
                meta.setdefault(r.job_id, {})["vector_dimension_scores"] = r.dimension_scores
        for r in graph:
            if r.matched_skills:
                meta.setdefault(r.job_id, {})["graph_matched_skills"] = r.matched_skills
        return bm25_map, vector_map, graph_map, meta

    def _fuse_weighted_sum(
        self,
        bm25: list[ScoredJob],
        vector: list[ScoredJob],
        graph: list[ScoredJob],
        weights: dict[str, float],
    ) -> list[FusedResult]:
        bm25_map, vector_map, graph_map, meta = self._score_maps(bm25, vector, graph)
        all_ids = set(bm25_map) | set(vector_map) | set(graph_map)
        fused: list[FusedResult] = []
        for job_id in all_ids:
            s_bm25 = bm25_map.get(job_id, 0.0)
            s_vec = vector_map.get(job_id, 0.0)
            s_graph = graph_map.get(job_id, 0.0)
            score = (
                weights["bm25"] * s_bm25
                + weights["vector"] * s_vec
                + weights["graph"] * s_graph
            )
            sources = []
            if job_id in bm25_map:
                sources.append("bm25")
            if job_id in vector_map:
                sources.append("vector")
            if job_id in graph_map:
                sources.append("graph")
            extra = meta.get(job_id, {})
            fused.append(
                FusedResult(
                    job_id=job_id,
                    fused_score=score,
                    bm25_score=s_bm25,
                    vector_score=s_vec,
                    graph_score=s_graph,
                    sources=sources,
                    vector_dimension_scores=extra.get("vector_dimension_scores"),
                    graph_matched_skills=extra.get("graph_matched_skills"),
                ),
            )
        fused.sort(key=lambda x: x.fused_score, reverse=True)
        max_score = fused[0].fused_score if fused else 1.0
        if max_score > 0:
            for item in fused:
                item.fused_score = item.fused_score / max_score
        return fused

    def _fuse_rrf(
        self,
        bm25: list[ScoredJob],
        vector: list[ScoredJob],
        graph: list[ScoredJob],
        weights: dict[str, float],
    ) -> list[FusedResult]:
        k_const = self._settings.retrieval.rrf_k
        bm25_map, vector_map, graph_map, meta = self._score_maps(bm25, vector, graph)
        all_ids = set(bm25_map) | set(vector_map) | set(graph_map)

        def rrf_rank(results: list[ScoredJob], weight: float) -> dict[str, float]:
            scores: dict[str, float] = {}
            for rank, item in enumerate(results, start=1):
                scores[item.job_id] = weight * (1.0 / (k_const + rank))
            return scores

        rrf_scores: dict[str, float] = {}
        for job_id in all_ids:
            rrf_scores[job_id] = (
                rrf_rank(bm25, weights["bm25"]).get(job_id, 0.0)
                + rrf_rank(vector, weights["vector"]).get(job_id, 0.0)
                + rrf_rank(graph, weights["graph"]).get(job_id, 0.0)
            )

        fused: list[FusedResult] = []
        for job_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            sources = []
            if job_id in bm25_map:
                sources.append("bm25")
            if job_id in vector_map:
                sources.append("vector")
            if job_id in graph_map:
                sources.append("graph")
            extra = meta.get(job_id, {})
            fused.append(
                FusedResult(
                    job_id=job_id,
                    fused_score=score,
                    bm25_score=bm25_map.get(job_id, 0.0),
                    vector_score=vector_map.get(job_id, 0.0),
                    graph_score=graph_map.get(job_id, 0.0),
                    sources=sources,
                    vector_dimension_scores=extra.get("vector_dimension_scores"),
                    graph_matched_skills=extra.get("graph_matched_skills"),
                ),
            )
        max_score = fused[0].fused_score if fused else 1.0
        if max_score > 0:
            for item in fused:
                item.fused_score = item.fused_score / max_score
        return fused
