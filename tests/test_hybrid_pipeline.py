"""Integration-style tests for hybrid pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.api.schemas.candidate import CandidateProfile
from src.api.schemas.recommendation import ScoredJob
from src.embeddings.encoder import EMBEDDING_DIM
from src.embeddings.schemas import CandidateEmbeddings
from src.matching.schemas import FusedResult


@pytest.fixture
def sample_profile():
    return CandidateProfile(
        name="Test User",
        email="test@example.com",
        summary="Machine learning engineer with Python experience",
        skills=[],
        esco_linked_skills=[],
    )


@pytest.fixture
def sample_embeddings():
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    return CandidateEmbeddings(skill=vec, domain=vec, role=vec, environment=vec)


def test_retrieve_hybrid_degrades_when_bm25_fails(sample_profile, sample_embeddings):
    with patch("src.matching.hybrid_pipeline.BM25Retriever") as mock_bm25_cls:
        mock_bm25 = MagicMock()
        mock_bm25.retrieve.side_effect = RuntimeError("ES down")
        mock_bm25_cls.return_value = mock_bm25

        with patch("src.matching.hybrid_pipeline.VectorRetriever") as mock_vr_cls:
            mock_vr = MagicMock()
            mock_vr.retrieve.return_value = [_job("j1", 0.9)]
            mock_vr_cls.return_value = mock_vr

            with patch("src.matching.hybrid_pipeline.GraphRetriever") as mock_gr_cls:
                mock_gr = MagicMock()
                mock_gr.retrieve.return_value = []
                mock_gr_cls.return_value = mock_gr

                with patch("src.matching.hybrid_pipeline.HybridFuser") as mock_fuser_cls:
                    mock_fuser = MagicMock()
                    mock_fuser.fuse.return_value = [
                        FusedResult(job_id="j1", fused_score=0.9, sources=["vector"]),
                    ]
                    mock_fuser_cls.return_value = mock_fuser

                    from src.matching.hybrid_pipeline import retrieve_hybrid

                    results = retrieve_hybrid(
                        profile=sample_profile,
                        embeddings=sample_embeddings,
                        session=MagicMock(),
                    )
                    assert len(results) == 1
                    mock_fuser.fuse.assert_called_once()
                    assert mock_fuser.fuse.call_args[0][0] == []


def _job(job_id: str, score: float) -> ScoredJob:
    return ScoredJob(job_id=job_id, score=score, source="vector")
