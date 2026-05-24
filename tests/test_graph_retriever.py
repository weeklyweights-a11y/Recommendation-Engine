"""Tests for GraphRetriever."""

from unittest.mock import MagicMock, patch

import pytest

from src.api.schemas.candidate import ESCOLinkedSkill
from src.knowledge_graph.schemas import ExpandedSkill
from src.matching.graph_retriever import GraphRetriever


@pytest.fixture
def reverse_index():
    return {
        "uri:python": {"job-a", "job-b"},
        "uri:ml": {"job-b"},
    }


def test_direct_match_scores_highest():
    retriever = GraphRetriever()
    retriever._reverse_index = {"uri:python": {"job-a"}}
    retriever._job_skill_counts = {"job-a": 1}
    retriever._index_built = True

    with patch("src.matching.graph_retriever.expand_skill") as mock_expand:
        mock_expand.return_value = []
        results = retriever.retrieve(
            [ESCOLinkedSkill(original_name="Python", esco_uri="uri:python", esco_label="Python", match_type="exact", confidence=1.0)],
            top_k=5,
            session=MagicMock(),
        )
    assert len(results) == 1
    assert results[0].job_id == "job-a"
    assert results[0].score == pytest.approx(1.0, abs=0.01)


def test_one_hop_match():
    retriever = GraphRetriever()
    retriever._reverse_index = {"uri:ml": {"job-b"}}
    retriever._job_skill_counts = {"job-b": 2}
    retriever._index_built = True

    with patch("src.matching.graph_retriever.expand_skill") as mock_expand:
        mock_expand.return_value = [
            ExpandedSkill(uri="uri:ml", label="ML", weight=0.5, hop_distance=1, path=[]),
        ]
        results = retriever.retrieve(
            [ESCOLinkedSkill(original_name="Python", esco_uri="uri:python", esco_label="Python", match_type="exact", confidence=1.0)],
            top_k=5,
            session=MagicMock(),
        )
    assert any(r.job_id == "job-b" for r in results)


def test_empty_candidate_skills_returns_empty():
    retriever = GraphRetriever()
    retriever._index_built = True
    results = retriever.retrieve([], top_k=5, session=MagicMock())
    assert results == []
