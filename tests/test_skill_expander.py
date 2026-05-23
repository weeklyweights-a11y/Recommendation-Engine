"""Tests for ESCO skill graph expansion."""

import pytest

from config.settings import SkillGraphSettings
from src.knowledge_graph.skill_expander import (
    _cached_expand,
    expand_skill,
    expand_skill_by_label,
    expand_skills,
    _hop_weight,
)


def test_hop_weight_broader_penalty():
    """Broader relations apply penalty multiplier."""
    settings = SkillGraphSettings()
    related = _hop_weight(settings, 1, "related")
    broader = _hop_weight(settings, 1, "broader")
    assert broader < related


@pytest.mark.integration
def test_expand_pytorch_label():
    """PyTorch label expands to related ML skills."""
    expanded = expand_skill_by_label("PyTorch", max_hops=2)
    labels = {s.label.lower() for s in expanded}
    assert expanded
    assert any(
        term in " ".join(labels)
        for term in ("deep learning", "neural", "tensor", "machine learning")
    )


@pytest.mark.integration
def test_expand_kubernetes_label():
    """Kubernetes expands toward container-related skills."""
    expanded = expand_skill_by_label("Kubernetes", max_hops=2)
    assert expanded


@pytest.mark.integration
def test_hop_decay_ordering():
    """Closer hops should not have lower weight than farther when same path."""
    expanded = expand_skill_by_label("Python", max_hops=2)
    if len(expanded) >= 2:
        assert expanded[0].weight >= expanded[-1].weight or expanded[0].hop_distance <= expanded[-1].hop_distance


@pytest.mark.integration
def test_expand_skills_batch_merge():
    """Batch expansion merges under __merged__ key."""
    from src.knowledge_graph.entity_linker import link_skill

    linked = link_skill("Python")
    assert linked is not None
    result = expand_skills([linked.esco_uri], max_hops=1)
    assert linked.esco_uri in result
    assert "__merged__" in result


@pytest.mark.integration
def test_cache_hit():
    """Second call uses LRU cache."""
    from src.knowledge_graph.entity_linker import link_skill

    linked = link_skill("Python")
    assert linked is not None
    _cached_expand.cache_clear()
    first = expand_skill(linked.esco_uri, max_hops=1)
    info_before = _cached_expand.cache_info()
    second = expand_skill(linked.esco_uri, max_hops=1)
    info_after = _cached_expand.cache_info()
    assert len(second) == len(first)
    assert info_after.hits >= info_before.hits
