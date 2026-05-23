"""ESCO skill graph expansion with hop-decay weighting."""

import logging
from functools import lru_cache
from typing import Optional

from config.settings import get_settings
from src.knowledge_graph.entity_linker import link_skill
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.schemas import ExpandedSkill, OccupationSkill

logger = logging.getLogger(__name__)


def _hop_weight(settings, hop: int, rel_type: str) -> float:
    """Compute decay weight for a hop and relationship type."""
    if hop == 1:
        base = settings.skill_graph.skill_hop_decay_1
    elif hop == 2:
        base = settings.skill_graph.skill_hop_decay_2
    else:
        base = settings.skill_graph.skill_hop_decay_3
    if rel_type in ("BROADER_THAN", "broader"):
        return base * settings.skill_graph.skill_broader_penalty
    return base


@lru_cache(maxsize=10000)
def _cached_expand(skill_uri: str, max_hops: int) -> tuple[ExpandedSkill, ...]:
    """LRU-cached expansion keyed by uri and hops."""
    results = _expand_skill_impl(skill_uri, max_hops)
    return tuple(results)


def _expand_skill_impl(skill_uri: str, max_hops: int) -> list[ExpandedSkill]:
    """Traverse ESCO graph from a skill URI."""
    settings = get_settings()
    cypher = f"""
    MATCH (start:Skill {{uri: $uri}})
    MATCH path = (start)-[:RELATED_TO*1..{max_hops}]-(related:Skill)
    WHERE related.uri <> start.uri
    RETURN related.uri AS uri, related.label AS label,
           length(path) AS hop_distance,
           [rel IN relationships(path) | coalesce(rel.relation_type, 'related')] AS rel_types
  """
    with Neo4jClient() as client:
        rows = client.run_query(cypher, {"uri": skill_uri})

    best: dict[str, ExpandedSkill] = {}
    for row in rows:
        hop = int(row["hop_distance"])
        rel_types = row.get("rel_types") or ["RELATED_TO"]
        rel_type = str(rel_types[-1] if rel_types else "related").lower()
        weight = _hop_weight(settings, hop, rel_type)
        uri = str(row["uri"])
        label = str(row.get("label", ""))
        if uri in best and best[uri].weight >= weight:
            continue
        best[uri] = ExpandedSkill(
            uri=uri,
            label=label,
            weight=weight,
            hop_distance=hop,
            path=[skill_uri, uri],
        )
    return sorted(best.values(), key=lambda s: s.weight, reverse=True)


def expand_skill(skill_uri: str, max_hops: int = 2) -> list[ExpandedSkill]:
    """Expand a skill URI to related skills with decay weights."""
    settings = get_settings()
    hops = min(max_hops, settings.skill_graph.skill_expansion_max_hops)
    cached = _cached_expand(skill_uri, hops)
    return list(cached)


def expand_skill_by_label(label: str, max_hops: int = 2) -> list[ExpandedSkill]:
    """Link a label then expand its ESCO URI."""
    linked = link_skill(label)
    if not linked:
        return []
    return expand_skill(linked.esco_uri, max_hops=max_hops)


def expand_skills(skill_uris: list[str], max_hops: int = 2) -> dict[str, list[ExpandedSkill]]:
    """Batch expand skills; merge duplicate targets keeping max weight."""
    per_skill: dict[str, list[ExpandedSkill]] = {}
    merged: dict[str, ExpandedSkill] = {}
    for uri in skill_uris:
        expanded = expand_skill(uri, max_hops=max_hops)
        per_skill[uri] = expanded
        for item in expanded:
            if item.uri not in merged or merged[item.uri].weight < item.weight:
                merged[item.uri] = item
    per_skill["__merged__"] = list(merged.values())
    return per_skill


def get_skills_for_occupation(occupation_uri: str) -> list[OccupationSkill]:
    """Return skills required by an occupation."""
    cypher = """
    MATCH (o:Occupation {uri: $uri})-[r:REQUIRES_SKILL]->(s:Skill)
    RETURN s.uri AS uri, s.label AS label, r.relationship_type AS relationship_type
    """
    with Neo4jClient() as client:
        rows = client.run_query(cypher, {"uri": occupation_uri})
    return [
        OccupationSkill(
            uri=str(row["uri"]),
            label=str(row.get("label", "")),
            relationship_type=str(row.get("relationship_type", "essential")),
        )
        for row in rows
    ]
