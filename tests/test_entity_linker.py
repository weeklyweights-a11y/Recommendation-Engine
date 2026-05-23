"""Tests for ESCO entity linking."""

import pytest

from src.knowledge_graph.entity_linker import link_skill, link_skills


@pytest.mark.integration
def test_link_python_exact():
    """Python should match exactly."""
    result = link_skill("Python")
    assert result is not None
    assert result.match_type == "exact"
    assert result.confidence == 1.0


@pytest.mark.integration
def test_link_pytorch_fuzzy():
    """Pytorch typo should fuzzy-match."""
    result = link_skill("Pytorch")
    assert result is not None
    assert result.match_type in ("exact", "fuzzy")


@pytest.mark.integration
def test_link_container_orchestration_semantic():
    """Semantic match for related phrasing."""
    result = link_skill("container orchestration")
    assert result is not None
    assert result.match_type in ("fuzzy", "semantic")


@pytest.mark.integration
def test_link_garbage_returns_none():
    """Nonsense input should not link."""
    result = link_skill("xyzzyplugh999notaskill")
    assert result is None


@pytest.mark.integration
def test_link_skills_batch():
    """Batch linking returns aligned results."""
    results = link_skills(["Python", "Pytorch", "xyzzyplugh999notaskill"])
    assert len(results) == 3
    assert results[0] is not None
    assert results[2] is None


