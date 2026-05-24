"""Embedding generation for candidates and jobs."""

from src.embeddings.candidate_embedder import embed_candidate
from src.embeddings.encoder import (
    EmbeddingEncoder,
    deserialize_embedding,
    get_encoder,
    serialize_embedding,
)
from src.embeddings.schemas import CandidateEmbeddings

__all__ = [
    "CandidateEmbeddings",
    "EmbeddingEncoder",
    "deserialize_embedding",
    "embed_candidate",
    "get_encoder",
    "serialize_embedding",
]
