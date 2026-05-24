"""Tests for skill name validation."""

from src.ingestion.skill_filters import filter_skill_names, is_plausible_skill


def test_rejects_email_and_prose() -> None:
    assert not is_plausible_skill("Bhargavin189@gmail.com")
    assert not is_plausible_skill("424-402-7408")
    assert not is_plausible_skill("Driving an 8% increase in automated")
    assert not is_plausible_skill("Professional Experience")
    assert is_plausible_skill("PyTorch")
    assert is_plausible_skill("LangChain")


def test_filter_skill_names() -> None:
    names = ["Python", "bhargavi@test.com", "SQL", "Summary"]
    assert filter_skill_names(names) == ["Python", "SQL"]
