"""FAISS index build, load, and search for job embeddings."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from src.db.models import Job
from src.embeddings.encoder import EMBEDDING_DIM, deserialize_embedding

logger = logging.getLogger(__name__)

DIMENSIONS = ("skill", "domain", "role", "environment")
_COLUMN_MAP = {
    "skill": Job.embedding_skill,
    "domain": Job.embedding_domain,
    "role": Job.embedding_role,
    "environment": Job.embedding_environment,
}


class FAISSManager:
    """Manage four per-dimension FAISS indexes with lazy loading."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize paths; indexes load on first search."""
        self._settings = settings or get_settings()
        self._index_dir = Path(self._settings.embedding.faiss_index_path)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._indexes: dict[str, faiss.Index] = {}
        self._id_maps: dict[str, np.ndarray] = {}
        self._index_meta: dict[str, dict[str, Any]] = {}

    def _index_path(self, dimension: str) -> Path:
        return self._index_dir / f"{dimension}_index.faiss"

    def _ids_path(self, dimension: str) -> Path:
        return self._index_dir / f"{dimension}_ids.npy"

    def _select_index_type(self, n_vectors: int) -> str:
        cfg = self._settings.job_embedding
        if cfg.faiss_index_type:
            return cfg.faiss_index_type
        if n_vectors < cfg.faiss_flat_ip_max_jobs:
            return "flat"
        if n_vectors <= 500_000:
            return "ivf"
        return "ivfpq"

    def _create_index(self, n_vectors: int) -> tuple[faiss.Index, str]:
        """Create a FAISS index appropriate for vector count."""
        index_type = self._select_index_type(n_vectors)
        if index_type == "flat":
            return faiss.IndexFlatIP(EMBEDDING_DIM), "IndexFlatIP"
        nlist = self._settings.job_embedding.faiss_ivf_nlist
        if nlist <= 0:
            nlist = max(int(math.sqrt(n_vectors)), 1)
        quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
        index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist, faiss.METRIC_INNER_PRODUCT)
        return index, f"IndexIVFFlat(nlist={nlist})"

    def build_indexes(self, session: Session) -> dict[str, dict[str, Any]]:
        """Build and persist all four indexes from PostgreSQL embeddings."""
        stats: dict[str, dict[str, Any]] = {}
        for dimension in DIMENSIONS:
            stats[dimension] = self._build_one_index(session, dimension)
        return stats

    def _build_one_index(self, session: Session, dimension: str) -> dict[str, Any]:
        """Build a single dimension index from embedded jobs."""
        column = _COLUMN_MAP[dimension]
        rows = session.execute(
            select(Job.id, column).where(Job.is_embedded.is_(True), column.isnot(None)),
        ).all()
        if not rows:
            logger.warning("No embeddings for dimension %s", dimension)
            return {"vector_count": 0}

        job_ids: list[str] = []
        vectors: list[np.ndarray] = []
        for job_id, blob in rows:
            if blob:
                job_ids.append(str(job_id))
                vectors.append(deserialize_embedding(blob))

        matrix = np.stack(vectors, axis=0).astype(np.float32)
        n_vectors = matrix.shape[0]
        index, type_name = self._create_index(n_vectors)
        if isinstance(index, faiss.IndexIVFFlat):
            index.train(matrix)
            index.nprobe = self._settings.job_embedding.faiss_nprobe
        index.add(matrix)

        faiss.write_index(index, str(self._index_path(dimension)))
        np.save(self._ids_path(dimension), np.array(job_ids, dtype=object))

        self._indexes.pop(dimension, None)
        self._id_maps.pop(dimension, None)

        file_size = os.path.getsize(self._index_path(dimension))
        meta = {
            "vector_count": n_vectors,
            "index_type": type_name,
            "file_size_bytes": file_size,
        }
        self._index_meta[dimension] = meta
        logger.info("Built FAISS %s index: %s vectors, %s, %s bytes", dimension, n_vectors, type_name, file_size)
        return meta

    def _ensure_loaded(self, dimension: str) -> None:
        """Lazy-load index and id map for a dimension."""
        if dimension in self._indexes:
            return
        index_path = self._index_path(dimension)
        ids_path = self._ids_path(dimension)
        if not index_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"FAISS index not found for dimension {dimension}")
        index = faiss.read_index(str(index_path))
        if isinstance(index, faiss.IndexIVF):
            index.nprobe = self._settings.job_embedding.faiss_nprobe
        self._indexes[dimension] = index
        self._id_maps[dimension] = np.load(ids_path, allow_pickle=True)

    def reload_index(self, dimension: str) -> None:
        """Drop cached index so next search reloads from disk."""
        self._indexes.pop(dimension, None)
        self._id_maps.pop(dimension, None)
        self._ensure_loaded(dimension)

    def search(
        self,
        dimension: str,
        query_vector: np.ndarray,
        top_k: int = 500,
    ) -> list[tuple[str, float]]:
        """Search one dimension; return (job_id, score) pairs."""
        try:
            self._ensure_loaded(dimension)
            index = self._indexes[dimension]
            id_map = self._id_maps[dimension]
            query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
            scores, indices = index.search(query, min(top_k, index.ntotal))
            results: list[tuple[str, float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                job_id = str(id_map[int(idx)])
                results.append((job_id, float(score)))
            return results
        except Exception as exc:
            logger.exception("FAISS search failed for dimension %s: %s", dimension, exc)
            raise

    def search_multi(
        self,
        query_vectors: dict[str, np.ndarray],
        top_k: int = 500,
        weights: Optional[dict[str, float]] = None,
    ) -> list[tuple[str, float]]:
        """Search multiple dimensions and fuse with weighted scores."""
        if weights is None:
            r = self._settings.retrieval
            weights = {
                "skill": r.vector_skill_weight,
                "domain": r.vector_domain_weight,
                "role": r.vector_role_weight,
                "environment": r.vector_environment_weight,
            }
        total_w = sum(weights.get(d, 0.0) for d in query_vectors)
        if total_w <= 0:
            total_w = 1.0
        fused: dict[str, float] = {}
        for dimension, vector in query_vectors.items():
            w = weights.get(dimension, 0.0) / total_w
            if w <= 0:
                continue
            try:
                hits = self.search(dimension, vector, top_k=top_k)
                if not hits:
                    continue
                max_score = max(s for _, s in hits) or 1.0
                for job_id, score in hits:
                    norm = score / max_score if max_score > 0 else 0.0
                    fused[job_id] = fused.get(job_id, 0.0) + w * norm
            except Exception as exc:
                logger.warning("search_multi skip dimension %s: %s", dimension, exc)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def get_index_stats(self) -> dict[str, dict[str, Any]]:
        """Return stats for each dimension index."""
        stats: dict[str, dict[str, Any]] = {}
        for dimension in DIMENSIONS:
            path = self._index_path(dimension)
            if path.exists():
                stats[dimension] = {
                    "exists": True,
                    "file_size_bytes": os.path.getsize(path),
                    **self._index_meta.get(dimension, {}),
                }
                try:
                    self._ensure_loaded(dimension)
                    stats[dimension]["vector_count"] = self._indexes[dimension].ntotal
                except Exception:
                    pass
            else:
                stats[dimension] = {"exists": False}
        return stats
