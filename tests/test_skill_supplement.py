"""Tests for skill enrichment."""

from src.api.schemas.candidate import CandidateProfile, ProfileSkill
from src.ingestion.schemas import ExtractedProfile, ExtractedSkill, GitHubProfile
from src.ingestion.skill_supplement import enrich_extracted_skills


def _profile() -> ExtractedProfile:
    return ExtractedProfile(
        name="Dev",
        summary="Engineer",
        experience=[
            {
                "company": "Co",
                "title": "DS",
                "start_date": "2020-01",
                "end_date": "present",
            },
        ],
        skills=[ExtractedSkill(name="Python", category="programming_language", proficiency="expert")],
    )


def test_section_and_github_only_not_full_resume() -> None:
    resume = """
    Bhargavin189@gmail.com | 424-402-7408 | USA

    Technical Skills:
    PyTorch, TensorFlow, Kubernetes, LangGraph

    Experience:
    Driving an 8% increase in automated categorization at Intuit.
    """
    github = GitHubProfile(
        username="dev",
        inferred_skills=["Go"],
        languages_distribution={"TypeScript": 0.5},
    )
    enriched = enrich_extracted_skills(_profile(), resume, github)
    names = {s.name.lower() for s in enriched.skills}
    assert "python" in names
    assert "pytorch" in names
    assert "langgraph" in names
    assert "go" in names
    assert "bhargavin189@gmail.com" not in names
    assert not any("driving an" in n for n in names)


def test_split_compound_langchain_langgraph() -> None:
    profile = ExtractedProfile(
        name="Dev",
        summary="Engineer",
        experience=[
            {
                "company": "Co",
                "title": "DS",
                "start_date": "2020-01",
                "end_date": "present",
            },
        ],
        skills=[
            ExtractedSkill(
                name="LangChain / LangGraph",
                category="framework",
                proficiency="expert",
            ),
        ],
    )
    enriched = enrich_extracted_skills(profile, "Skills: LangChain / LangGraph", None)
    names = {s.name.lower() for s in enriched.skills}
    assert "langchain" in names
    assert "langgraph" in names
