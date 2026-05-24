"""Candidate profile page (expanded in Step 5.5)."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render profile placeholder until Step 5.5."""
    if not st.session_state.get("candidate_id"):
        st.warning("Complete onboarding to view your profile.")
        return
    st.info("Profile page loads in a later step.")
