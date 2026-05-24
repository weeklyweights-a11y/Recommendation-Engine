"""Tests for skill match display builder."""

from src.api.schemas.candidate import CandidateProfile, ProfileSkill
from src.db.models import Job
from src.matching.schemas import SkillOverlap
from src.matching.skill_match_display import build_skill_match_display


def _job_with_skills(skills: list[dict]) -> Job:
    return Job(
        title="Data Scientist",
        company="Acme",
        description="Role description",
        skills_extracted={"extraction_method": "rule", "skills": skills},
    )


def test_name_match_tensorflow_not_a_gap() -> None:
    profile = CandidateProfile(
        name="Test",
        skills=[
            ProfileSkill(name="Python", depth_score=0.8),
            ProfileSkill(name="TensorFlow", depth_score=0.7),
            ProfileSkill(name="Deep Learning", depth_score=0.6),
        ],
    )
    job = _job_with_skills(
        [
            {"name": "Python", "esco_uri": "http://data.europa.eu/esco/skill/python"},
            {"name": "TensorFlow", "esco_uri": "http://data.europa.eu/esco/skill/tensorflow"},
            {"name": "SQL", "esco_uri": "http://data.europa.eu/esco/skill/sql"},
        ],
    )
    overlap = SkillOverlap(
        direct_matches=[{"uri": "http://data.europa.eu/esco/skill/python", "hop": 0}],
        unmatched_job_skills=[
            "http://data.europa.eu/esco/skill/tensorflow",
            "http://data.europa.eu/esco/skill/sql",
        ],
    )
    result = build_skill_match_display(profile, job, overlap)
    matched_names = {m["skill"] for m in result["matched"]}
    gap_names = {g["skill"] for g in result["gaps"]}
    assert "Python" in matched_names
    assert "TensorFlow" in matched_names
    assert "TensorFlow" not in gap_names
    assert "SQL" in gap_names


def test_alias_ml_matches_profile() -> None:
    profile = CandidateProfile(
        name="Test",
        skills=[ProfileSkill(name="Machine Learning", depth_score=0.9)],
    )
    job = _job_with_skills([{"name": "ML"}])
    result = build_skill_match_display(profile, job, None)
    assert any(m["skill"] == "ML" for m in result["matched"])
    assert result["gaps"] == []
