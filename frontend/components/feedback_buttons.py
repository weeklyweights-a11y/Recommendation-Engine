"""Feedback action buttons for job cards."""

from __future__ import annotations

from typing import Optional

import streamlit as st

from frontend.utils.api_client import ApiError, post_feedback


def hydrate_feedback(candidate_id: str) -> None:
    """Load feedback from API into session state."""
    from frontend.utils.api_client import list_feedback

    try:
        rows = list_feedback(candidate_id)
        mapping = {str(row["job_id"]): row["action"] for row in rows}
        st.session_state.feedback = mapping
    except ApiError:
        pass


def render_feedback_buttons(
    candidate_id: str,
    job_id: str,
    job_source_url: Optional[str],
    *,
    key_suffix: str,
) -> None:
    """Render save, dismiss, and apply buttons for one job card."""
    feedback = st.session_state.setdefault("feedback", {})
    jid = str(job_id)
    current = feedback.get(jid)

    col1, col2, col3 = st.columns(3)
    with col1:
        if current == "saved":
            st.button("Saved", disabled=True, key=f"fb_save_done_{key_suffix}")
        elif st.button("Save", key=f"fb_save_{key_suffix}"):
            try:
                post_feedback(candidate_id, job_id, "saved")
                feedback[jid] = "saved"
                st.toast("Saved!")
                st.rerun()
            except ApiError as exc:
                if exc.status_code == 409:
                    feedback[jid] = "saved"
                    st.rerun()
                else:
                    st.error(exc.message)

    with col2:
        if current == "dismissed":
            st.caption("Dismissed")
        elif st.button("Not for me", key=f"fb_dismiss_{key_suffix}"):
            try:
                post_feedback(candidate_id, job_id, "dismissed")
                feedback[jid] = "dismissed"
                st.toast("Got it — we'll show fewer like this")
                st.rerun()
            except ApiError as exc:
                if exc.status_code == 409:
                    feedback[jid] = "dismissed"
                    st.rerun()
                else:
                    st.error(exc.message)

    with col3:
        if current == "applied":
            st.button("Applied", disabled=True, key=f"fb_apply_done_{key_suffix}")
        else:
            if job_source_url:
                st.markdown(f"[Open posting ↗]({job_source_url})")
            if st.button("Apply", key=f"fb_apply_{key_suffix}"):
                try:
                    post_feedback(candidate_id, job_id, "applied")
                    feedback[jid] = "applied"
                    st.toast("Applied!")
                    st.rerun()
                except ApiError as exc:
                    if exc.status_code == 409:
                        feedback[jid] = "applied"
                        st.rerun()
                    else:
                        st.error(exc.message)
            if not job_source_url:
                st.caption("No direct link — see job details above.")
