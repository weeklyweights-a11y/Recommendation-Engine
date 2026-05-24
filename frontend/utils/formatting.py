"""Display formatting helpers for the Streamlit UI."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from frontend.config import get_frontend_settings

_FACTOR_LABELS: dict[str, str] = {
    "skill_fit": "Skill Fit",
    "experience_alignment": "Experience",
    "domain_relevance": "Domain",
    "role_shape_match": "Role Shape",
    "location_fit": "Location",
    "company_stage_alignment": "Company Stage",
    "semantic_similarity": "Semantic",
}


def format_posted_freshness(posted_date: Optional[datetime]) -> tuple[str, str]:
    """Return relative label and color token for posted date."""
    cfg = get_frontend_settings()
    if posted_date is None:
        return ("Date unknown", "gray")
    now = datetime.now(timezone.utc)
    if posted_date.tzinfo is None:
        posted_date = posted_date.replace(tzinfo=timezone.utc)
    delta = now - posted_date
    hours = delta.total_seconds() / 3600
    days = delta.days
    if hours < cfg.freshness_hours_green:
        return (f"Posted {max(int(hours), 1)} hours ago", "green")
    if days < cfg.freshness_days_normal:
        return (f"Posted {days} days ago", "normal")
    return (f"Posted {days} days ago", "gray")


def match_pct_color(pct: int) -> str:
    """Return color token for match percentage band."""
    cfg = get_frontend_settings()
    if pct >= cfg.match_pct_green_min:
        return "green"
    if pct >= cfg.match_pct_blue_min:
        return "blue"
    if pct >= cfg.match_pct_yellow_min:
        return "yellow"
    return "gray"


def format_factor_scores(factor_scores: dict[str, Any]) -> list[tuple[str, float]]:
    """Format factor scores with human-readable labels."""
    rows: list[tuple[str, float]] = []
    for key, label in _FACTOR_LABELS.items():
        if key in factor_scores:
            try:
                rows.append((label, float(factor_scores[key])))
            except (TypeError, ValueError):
                continue
    return rows


def parse_explanation(explanation: Any) -> dict[str, Any]:
    """Normalize explanation payload from API."""
    if explanation is None:
        return {}
    if isinstance(explanation, dict):
        return explanation
    if isinstance(explanation, str):
        import json

        try:
            parsed = json.loads(explanation)
            return parsed if isinstance(parsed, dict) else {"summary": explanation}
        except json.JSONDecodeError:
            return {"summary": explanation}
    return {}
