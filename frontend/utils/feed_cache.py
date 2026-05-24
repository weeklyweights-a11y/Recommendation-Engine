"""Clear cached job recommendations when the candidate profile changes."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import streamlit as st


def profile_fingerprint(profile: dict[str, Any] | None) -> str:
    """Stable hash of profile skills + preferences relevant to matching."""
    if not profile:
        return ""
    skills = profile.get("skills") or []
    names: list[str] = []
    for skill in skills:
        if isinstance(skill, dict) and skill.get("name"):
            names.append(str(skill["name"]).strip().lower())
    prefs = profile.get("preferences") or {}
    blob = json.dumps(
        {
            "skills": sorted(names),
            "archetype": profile.get("role_archetype"),
            "years": profile.get("total_years_experience"),
            "prefs": prefs,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def clear_feed_cache() -> None:
    """Drop in-memory feed data and Streamlit's HTTP cache for recommendations."""
    st.session_state.recommendations = []
    st.session_state.rec_meta = {}
    st.session_state.pop("recommendations_profile_fp", None)
    try:
        from frontend.pages.feed import _fetch_recommendations

        _fetch_recommendations.clear()
    except Exception:
        pass


def mark_profile_updated(profile: dict[str, Any] | None) -> None:
    """Invalidate feed cache so the next load re-runs the matching pipeline."""
    st.session_state.profile = profile
    clear_feed_cache()
    st.session_state.force_refresh_feed = True


def profile_changed_since_feed() -> bool:
    """True when session profile differs from the profile used for current recs."""
    current = profile_fingerprint(st.session_state.get("profile"))
    stored = st.session_state.get("recommendations_profile_fp")
    if not current:
        return False
    if not stored:
        return True
    return current != stored
