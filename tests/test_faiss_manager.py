"""Tests for FAISSManager."""

import tempfile
from pathlib import Path

import faiss
import numpy as np

from config.settings import Settings
from src.embeddings.encoder import EMBEDDING_DIM
from src.embeddings.faiss_manager import FAISSManager


def _settings_with_tmp_dir(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.embedding.faiss_index_path = str(tmp_path)
    return settings


def test_faiss_search_returns_ordered_results():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings_with_tmp_dir(Path(tmp))
        manager = FAISSManager(settings)
        dim_dir = Path(tmp)

        n = 20
        ids = [f"job-{i}" for i in range(n)]
        matrix = np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(matrix)
        faiss.write_index(index, str(dim_dir / "skill_index.faiss"))
        np.save(dim_dir / "skill_ids.npy", np.array(ids, dtype=object))

        query = matrix[0:1]
        results = manager.search("skill", query[0], top_k=5)
        assert len(results) > 0
        assert results[0][0] == "job-0"
        assert results[0][1] >= results[-1][1]
