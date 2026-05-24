"""Tests for VectorRetriever."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest

from config.settings import Settings
from src.embeddings.encoder import EMBEDDING_DIM
from src.embeddings.faiss_manager import FAISSManager
from src.embeddings.schemas import CandidateEmbeddings
from src.matching.vector_retriever import VectorRetriever


def _build_index(tmp_path: Path, job_ids: list[str], matrix: np.ndarray) -> FAISSManager:
    settings = Settings()
    settings.embedding.faiss_index_path = str(tmp_path)
    for dim in ("skill", "domain", "role", "environment"):
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(matrix.astype(np.float32))
        faiss.write_index(index, str(tmp_path / f"{dim}_index.faiss"))
        np.save(tmp_path / f"{dim}_ids.npy", np.array(job_ids, dtype=object))
    return FAISSManager(settings)


def test_vector_retriever_ranks_by_similarity():
    n = 10
    ids = [f"j{i}" for i in range(n)]
    matrix = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    matrix[3] = 1.0
    faiss.normalize_L2(matrix)

    with tempfile.TemporaryDirectory() as tmp:
        manager = _build_index(Path(tmp), ids, matrix)
        retriever = VectorRetriever(manager)
        query = CandidateEmbeddings(
            skill=matrix[3].copy(),
            domain=matrix[3].copy(),
            role=matrix[3].copy(),
            environment=matrix[3].copy(),
        )
        results = retriever.retrieve(query, top_k=3)
        assert results[0].job_id == "j3"
        assert all(0 <= r.score <= 1 for r in results)


def test_zero_vector_dimension_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        n = 5
        ids = [f"j{i}" for i in range(n)]
        matrix = np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
        faiss.normalize_L2(matrix)
        manager = _build_index(Path(tmp), ids, matrix)
        retriever = VectorRetriever(manager)
        query = CandidateEmbeddings(
            skill=matrix[0],
            domain=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            role=matrix[0],
            environment=matrix[0],
        )
        results = retriever.retrieve(query, top_k=3)
        assert len(results) > 0


def test_allowed_job_ids_filter():
    with tempfile.TemporaryDirectory() as tmp:
        n = 5
        ids = [f"j{i}" for i in range(n)]
        matrix = np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
        faiss.normalize_L2(matrix)
        manager = _build_index(Path(tmp), ids, matrix)
        retriever = VectorRetriever(manager)
        query = CandidateEmbeddings(
            skill=matrix[0],
            domain=matrix[0],
            role=matrix[0],
            environment=matrix[0],
        )
        allowed = {"j0"}
        results = retriever.retrieve(query, top_k=10, allowed_job_ids=allowed)
        assert all(r.job_id in allowed for r in results)
