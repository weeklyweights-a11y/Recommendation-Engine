"""Shared knowledge graph data models."""

from pydantic import BaseModel, Field


class ExpandedSkill(BaseModel):
    """A skill related to a seed skill via graph traversal."""

    uri: str
    label: str
    weight: float
    hop_distance: int
    path: list[str] = Field(default_factory=list)


class LinkedSkill(BaseModel):
    """A free-text skill linked to an ESCO node."""

    esco_uri: str
    esco_label: str
    match_type: str
    confidence: float


class OccupationSkill(BaseModel):
    """Skill required by an occupation."""

    uri: str
    label: str
    relationship_type: str
