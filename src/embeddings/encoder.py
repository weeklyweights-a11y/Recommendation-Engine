"""Sentence-transformer encoder with serialization helpers."""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
_encoder: Optional["EmbeddingEncoder"] = None


def get_encoder(settings: Settings | None = None) -> "EmbeddingEncoder":
    """Return the shared encoder singleton."""
    global _encoder
    if _encoder is None:
        _encoder = EmbeddingEncoder(settings=settings)
    return _encoder


def serialize_embedding(vector: np.ndarray) -> bytes:
    """Serialize a float32 embedding vector for PostgreSQL storage."""
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-dim vector, got {arr.shape[0]}")
    return arr.tobytes()


def deserialize_embedding(data: bytes) -> np.ndarray:
    """Deserialize bytes from PostgreSQL into a float32 vector."""
    arr = np.frombuffer(data, dtype=np.float32)
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-dim vector, got {arr.shape[0]}")
    return arr.copy()


class EmbeddingEncoder:
    """Lazy-loaded sentence-transformer wrapper."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None

    def _resolve_device(self) -> str:
        device_pref = self._settings.ingestion.embedding_device
        if device_pref == "cpu":
            return "cpu"
        if device_pref == "cuda":
            return "cuda"
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._settings.embedding.embedding_model,
                device=self._resolve_device(),
            )
        return self._model

    def _chunk_text(self, text: str) -> list[str]:
        limit = self._settings.ingestion.embedding_chunk_token_limit
        words = re.split(r"\s+", text.strip())
        if not words:
            return []
        chunks: list[str] = []
        current: list[str] = []
        for word in words:
            current.append(word)
            if len(current) >= limit:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string to a normalized 384-dim vector."""
        cleaned = (text or "").strip()
        if not cleaned:
            logger.warning("Empty text passed to encoder; returning zero vector")
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        model = self._load_model()
        chunks = self._chunk_text(cleaned)
        if not chunks:
            logger.warning("No encodable chunks; returning zero vector")
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        vectors = model.encode(chunks, normalize_embeddings=True)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            return matrix
        mean_vec = matrix.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        return mean_vec.astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts; empty inputs become zero vectors."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return np.stack([self.encode(text) for text in texts], axis=0)
