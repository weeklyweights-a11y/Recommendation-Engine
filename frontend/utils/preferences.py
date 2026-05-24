"""Normalize UI preference values for API and hard filters."""

from __future__ import annotations

_STAGE_ALIASES: dict[str, str] = {
    "pre seed": "pre-seed",
    "pre_seed": "pre-seed",
    "series a": "series-a",
    "series_a": "series-a",
    "series b": "series-b",
    "series_b": "series-b",
    "growth stage": "growth",
    "growth": "growth",
}

_WORK_MODEL_ALIASES: dict[str, str] = {
    "on-site": "onsite",
    "onsite": "onsite",
    "remote": "remote",
    "hybrid": "hybrid",
}


def normalize_stage(label: str) -> str:
    """Map display label to hard-filter stage token."""
    key = label.strip().lower()
    if key in _STAGE_ALIASES:
        return _STAGE_ALIASES[key]
    return key.replace(" ", "-").replace("_", "-")


def normalize_work_model(label: str) -> str:
    """Map display label to work model token."""
    key = label.strip().lower()
    return _WORK_MODEL_ALIASES.get(key, key.replace(" ", ""))


def normalize_job_type(label: str) -> str:
    """Map job type label to lowercase API value."""
    return label.strip().lower().replace("-", " ").replace("  ", " ")


def normalize_company_size(label: str) -> str:
    """Normalize company size bucket."""
    return label.strip()


def normalize_list(values: list[str], normalizer) -> list[str]:
    """Apply normalizer to each non-empty value."""
    out: list[str] = []
    for item in values:
        cleaned = str(item).strip()
        if cleaned:
            out.append(normalizer(cleaned))
    return out


def split_locations(text: str) -> list[str]:
    """Split comma-separated locations."""
    return [part.strip() for part in text.split(",") if part.strip()]


def split_roles(text: str) -> list[str]:
    """Split comma-separated role titles."""
    return [part.strip() for part in text.split(",") if part.strip()]


def preferences_to_api_payload(raw: dict) -> dict:
    """Convert form dict to CandidatePreferences-compatible JSON."""
    return {
        "job_types": normalize_list(raw.get("job_types") or [], normalize_job_type),
        "work_models": normalize_list(raw.get("work_models") or [], normalize_work_model),
        "locations": raw.get("locations") or [],
        "visa_sponsorship_needed": raw.get("visa_sponsorship_needed"),
        "salary_min": raw.get("salary_min"),
        "salary_max": raw.get("salary_max"),
        "company_stages": normalize_list(raw.get("company_stages") or [], normalize_stage),
        "company_sizes": normalize_list(raw.get("company_sizes") or [], normalize_company_size),
        "target_roles": raw.get("target_roles") or [],
        "industries_target": raw.get("industries_target") or [],
        "industries_avoid": raw.get("industries_avoid") or [],
        "priority_ranking": raw.get("priority_ranking") or [],
    }
