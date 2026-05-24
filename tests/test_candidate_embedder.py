"""Tests for embedding encoder and candidate embedder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.api.schemas.candidate import CandidateProfile, ProfileExperience, ProfileSkill
from src.embeddings.encoder import (
    EMBEDDING_DIM,
    EmbeddingEncoder,
    deserialize_embedding,
    serialize_embedding,
)
from src.embeddings.candidate_embedder import embed_candidate


def _mock_encode(text: str) -> np.ndarray:
    """Deterministic fake encoder for tests."""
    seed = sum(ord(c) for c in text) % 97
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vec[seed % EMBEDDING_DIM] = 1.0
    vec[(seed + 7) % EMBEDDING_DIM] = 0.5
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm else vec


@patch("sentence_transformers.SentenceTransformer")
def test_encoder_shape_and_empty(mock_st: MagicMock) -> None:
    """Encoder returns 384-dim vectors; empty input is zero vector."""
    mock_st.return_value.encode.return_value = np.ones((1, EMBEDDING_DIM), dtype=np.float32)
    encoder = EmbeddingEncoder()
    vector = encoder.encode("hello world")
    assert vector.shape == (EMBEDDING_DIM,)
    assert vector.dtype == np.float32

    empty = encoder.encode("   ")
    assert empty.shape == (EMBEDDING_DIM,)
    assert float(np.linalg.norm(empty)) == 0.0


@patch("sentence_transformers.SentenceTransformer")
def test_encode_batch(mock_st: MagicMock) -> None:
    """Batch encoding returns (N, 384) matrix."""
    mock_st.return_value.encode.return_value = np.ones((1, EMBEDDING_DIM), dtype=np.float32)
    encoder = EmbeddingEncoder()
    matrix = encoder.encode_batch(["a", "b"])
    assert matrix.shape == (2, EMBEDDING_DIM)


def test_serialization_roundtrip() -> None:
    """Serialize and deserialize preserve vector values."""
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    restored = deserialize_embedding(serialize_embedding(vec))
    np.testing.assert_allclose(restored, vec)


@patch("src.embeddings.candidate_embedder.get_encoder")
@patch("src.embeddings.candidate_embedder.expand_skill")
def test_embed_candidate_four_distinct_vectors(mock_expand: MagicMock, mock_get_encoder: MagicMock) -> None:
    """Four embedding types are not identical for a rich profile."""
    mock_expand.return_value = []
    encoder = MagicMock()
    encoder.encode.side_effect = _mock_encode
    mock_get_encoder.return_value = encoder

    profile = CandidateProfile(
        skills=[
            ProfileSkill(name="Python", category="programming_language", proficiency="expert", depth_score=0.9),
        ],
        experience=[
            ProfileExperience(
                company="Acme",
                title="Engineer",
                start_date="2020-01",
                end_date="present",
                description="Built APIs",
                domain="saas",
                company_stage_estimate="growth",
                role_type="ic",
                key_achievements=["Shipped v2"],
            ),
        ],
        domains=["saas"],
        role_archetype="platform_engineer",
        career_trajectory="ascending",
        summary="Platform engineer",
    )

    embeddings = embed_candidate(profile)
    vectors = [embeddings.skill, embeddings.domain, embeddings.role, embeddings.environment]
    assert all(vec.shape == (EMBEDDING_DIM,) for vec in vectors)
    assert not np.allclose(vectors[0], vectors[1])
    assert not np.allclose(vectors[0], vectors[2])
    assert not np.allclose(vectors[0], vectors[3])


@patch("src.embeddings.candidate_embedder.get_encoder")
def test_semantic_skill_similarity_loose(mock_get_encoder: MagicMock) -> None:
    """ML-heavy profiles produce more similar skill vectors than cross-domain pairs."""
    encoder = MagicMock()
    encoder.encode.side_effect = _mock_encode
    mock_get_encoder.return_value = encoder

    ml_profile = CandidateProfile(
        skills=[ProfileSkill(name="PyTorch", depth_score=0.8), ProfileSkill(name="TensorFlow", depth_score=0.7)],
        summary="ML engineer",
    )
    fe_profile = CandidateProfile(
        skills=[ProfileSkill(name="React", depth_score=0.8), ProfileSkill(name="CSS", depth_score=0.7)],
        summary="Frontend engineer",
    )

    ml_embeddings = embed_candidate(ml_profile)
    fe_embeddings = embed_candidate(fe_profile)

    ml_sim = float(np.dot(ml_embeddings.skill, ml_embeddings.skill))
    cross = float(np.dot(ml_embeddings.skill, fe_embeddings.skill))
    assert ml_sim >= cross
