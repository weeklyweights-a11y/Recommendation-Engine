"""Streamlit application entry point."""

from __future__ import annotations

import streamlit as st

from frontend.config import get_frontend_settings
from frontend.pages import feed, onboarding, profile


def _init_session() -> None:
    """Initialize session state defaults."""
    defaults = {
        "candidate_id": None,
        "profile": None,
        "preferences": None,
        "onboarding_complete": False,
        "onboarding_step": 1,
        "step5_preferences_dirty": False,
        "resume_bytes": None,
        "resume_filename": None,
        "github_username": None,
        "github_preview": None,
        "feedback": {},
        "recommendations": [],
        "rec_filters": {},
        "feed_view_mode": "all",
        "profile_build_started": False,
        "current_page": "Onboarding",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    """Run the Streamlit app."""
    cfg = get_frontend_settings()
    st.set_page_config(
        page_title="PersonalMatch",
        page_icon=cfg.page_icon,
        layout="wide",
    )
    _init_session()

    if not st.session_state.candidate_id:
        st.session_state.current_page = "Onboarding"
    elif st.session_state.get("current_page") == "Onboarding" and st.session_state.onboarding_complete:
        st.session_state.current_page = "Feed"

    pages = ["Onboarding", "Feed", "Profile"]
    if not st.session_state.onboarding_complete:
        default_idx = 0
    else:
        default_idx = pages.index(st.session_state.get("current_page", "Feed"))
    choice = st.sidebar.radio(
        "Navigate",
        pages,
        index=default_idx,
        key="nav_choice",
    )
    st.session_state.current_page = choice

    if choice == "Onboarding":
        onboarding.render()
    elif choice == "Feed":
        if not st.session_state.onboarding_complete:
            st.warning("Complete onboarding first.")
            onboarding.render()
        else:
            feed.render()
    elif choice == "Profile":
        if not st.session_state.candidate_id:
            st.warning("Complete onboarding first.")
            onboarding.render()
        else:
            profile.render()


if __name__ == "__main__":
    main()
