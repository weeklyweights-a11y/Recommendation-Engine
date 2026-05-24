"""Helpers for reading MergedPreferences in matching pipelines."""

from __future__ import annotations

from typing import Optional, TypeVar

from src.api.schemas.candidate import MergedPreferences, PreferenceField

T = TypeVar("T")

ALL_WORK_MODELS = frozenset({"remote", "hybrid", "onsite"})
ALL_COMPANY_SIZES = frozenset(
    {"1-10", "11-50", "51-200", "201-1000", "1000+", "10-50", "50-200", "200-1000"},
)
ALL_COMPANY_STAGES = frozenset(
    {
        "pre-seed",
        "seed",
        "series-a",
        "series-b",
        "growth",
        "enterprise",
        "pre_seed",
        "series_a",
        "series_b",
    },
)


def pref_value(field: PreferenceField[T]) -> Optional[T]:
    """Return preference value when the candidate set it (non-default empty)."""
    if field.value is None:
        return None
    if isinstance(field.value, list) and len(field.value) == 0:
        return None
    if field.source == "default":
        return None
    return field.value


def pref_list(field: PreferenceField[list[str]]) -> list[str]:
    """Return a normalized lowercase list or empty if unset."""
    raw = pref_value(field)
    if not raw:
        return []
    return [str(item).strip().lower() for item in raw if str(item).strip()]


def pref_bool(field: PreferenceField[bool]) -> Optional[bool]:
    """Return bool preference when explicitly or inferentially set."""
    if field.value is None:
        return None
    if field.source == "default":
        return None
    return bool(field.value)


def pref_int(field: PreferenceField[int]) -> Optional[int]:
    """Return int preference when set."""
    if field.value is None:
        return None
    if field.source == "default":
        return None
    return int(field.value)


def should_apply_work_model_filter(work_models: list[str]) -> bool:
    """Skip work-model filter when unset or all options selected."""
    if not work_models:
        return False
    normalized = {m.lower() for m in work_models}
    if normalized >= ALL_WORK_MODELS:
        return False
    return True


def should_apply_company_size_filter(sizes: list[str]) -> bool:
    """Skip company size filter when unset."""
    return bool(sizes)


def should_apply_company_stage_filter(stages: list[str]) -> bool:
    """Skip company stage filter when unset."""
    return bool(stages)
