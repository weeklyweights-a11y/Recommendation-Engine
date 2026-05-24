"""Embedding generation for candidates and jobs."""

from src.embeddings.candidate_embedder import embed_candidate
from src.embeddings.encoder import (
    EmbeddingEncoder,
    deserialize_embedding,
    get_encoder,
    serialize_embedding,
)
from src.embeddings.faiss_manager import FAISSManager
from src.embeddings.job_embedder import embed_job, embed_job_record, extract_job_fields
from src.embeddings.schemas import CandidateEmbeddings, JobEmbeddings, JobFields

__all__ = [
    "CandidateEmbeddings",
    "EmbeddingEncoder",
    "FAISSManager",
    "JobEmbeddings",
    "JobFields",
    "deserialize_embedding",
    "embed_candidate",
    "embed_job",
    "embed_job_record",
    "extract_job_fields",
    "get_encoder",
    "serialize_embedding",
]
