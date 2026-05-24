"""Tests for HybridFuser."""

from src.api.schemas.recommendation import ScoredJob
from src.matching.hybrid_fuser import HybridFuser


def _job(job_id: str, score: float, source: str) -> ScoredJob:
    return ScoredJob(job_id=job_id, score=score, source=source)


def test_weighted_sum_job_in_all_sources_ranks_first():
    fuser = HybridFuser()
    bm25 = [_job("a", 0.9, "bm25"), _job("b", 0.8, "bm25")]
    vector = [_job("a", 0.85, "vector"), _job("d", 0.7, "vector")]
    graph = [_job("a", 0.8, "graph"), _job("c", 0.6, "graph")]
    fused = fuser.fuse(bm25, vector, graph, top_k=10, strategy="weighted_sum")
    assert fused[0].job_id == "a"
    assert "bm25" in fused[0].sources


def test_rrf_fusion():
    fuser = HybridFuser()
    bm25 = [_job("a", 1.0, "bm25")]
    vector = [_job("b", 1.0, "vector")]
    graph = [_job("c", 1.0, "graph")]
    fused = fuser.fuse(bm25, vector, graph, top_k=3, strategy="rrf")
    assert len(fused) == 3


def test_missing_bm25_redistributes():
    fuser = HybridFuser()
    vector = [_job("b", 0.9, "vector")]
    graph = [_job("c", 0.8, "graph")]
    fused = fuser.fuse([], vector, graph, top_k=5)
    assert len(fused) > 0


def test_all_empty_returns_empty():
    fuser = HybridFuser()
    assert fuser.fuse([], [], [], top_k=5) == []


def test_diversity_stats():
    fuser = HybridFuser()
    bm25 = [_job("a", 1.0, "bm25"), _job("b", 1.0, "bm25")]
    vector = [_job("a", 1.0, "vector"), _job("d", 1.0, "vector")]
    graph = [_job("a", 1.0, "graph")]
    stats = fuser.analyze_retrieval_diversity(bm25, vector, graph)
    assert stats.all_three == 1
    assert stats.total_unique == 3
