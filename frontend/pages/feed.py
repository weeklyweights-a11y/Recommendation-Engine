"""Personalized job feed page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from config.settings import get_settings
from frontend.components.feedback_buttons import hydrate_feedback
from frontend.components.job_card import render_job_card
from frontend.utils.api_client import ApiError, get_recommendations
from frontend.utils.feed_cache import (
    clear_feed_cache,
    profile_changed_since_feed,
    profile_fingerprint,
)
from frontend.utils.preferences import normalize_stage, normalize_work_model


def _profile_skills() -> list[dict[str, Any]]:
    profile = st.session_state.get("profile") or {}
    if isinstance(profile, dict):
        return profile.get("skills") or []
    return []


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_recommendations(
    candidate_id: str,
    page: int,
    per_page: int,
    refresh: bool,
) -> dict[str, Any]:
    return get_recommendations(candidate_id, page=page, per_page=per_page, refresh=refresh)


def _load_all_recommendations(refresh: bool = False) -> list[dict[str, Any]]:
    """Load first page into session; pagination appends."""
    candidate_id = st.session_state.candidate_id
    per_page = get_settings().api.default_per_page
    must_refresh = refresh or profile_changed_since_feed()
    if must_refresh:
        clear_feed_cache()
    if must_refresh or not st.session_state.get("recommendations"):
        data = _fetch_recommendations(str(candidate_id), 1, per_page, must_refresh)
        st.session_state.recommendations = data.get("recommendations") or data.get("items") or []
        st.session_state.rec_meta = {
            "pagination": data.get("pagination") or {},
            "pipeline_stats": data.get("pipeline_stats"),
            "total_pages": data.get("pagination", {}).get("total_pages")
            or data.get("pages", 1),
            "page": 1,
        }
        st.session_state.recommendations_profile_fp = profile_fingerprint(
            st.session_state.get("profile"),
        )
    return st.session_state.recommendations


def _apply_filters(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filters = st.session_state.get("rec_filters") or {}
    work = {normalize_work_model(w) for w in filters.get("work_models", [])}
    stages = {normalize_stage(s) for s in filters.get("company_stages", [])}
    out = []
    for rec in recs:
        job = rec.get("job") or {}
        if work:
            rt = normalize_work_model(str(job.get("remote_type") or ""))
            if rt and rt not in work:
                continue
        if stages:
            cs = normalize_stage(str(job.get("company_stage") or ""))
            if cs and cs not in stages:
                continue
        out.append(rec)
    return out


def _apply_sort(recs: list[dict[str, Any]], sort_key: str) -> list[dict[str, Any]]:
    if sort_key == "Newest First":

        def _date(r: dict[str, Any]) -> str:
            job = r.get("job") or {}
            return str(job.get("posted_date") or "")

        return sorted(recs, key=_date, reverse=True)
    if sort_key == "Highest Salary":

        def _sal(r: dict[str, Any]) -> int:
            job = r.get("job") or {}
            return int(job.get("salary_max") or 0)

        return sorted(recs, key=_sal, reverse=True)
    return sorted(recs, key=lambda r: float(r.get("match_score") or 0), reverse=True)


def _hide_dismissed(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feedback = st.session_state.get("feedback") or {}
    dismissed = {jid for jid, act in feedback.items() if act == "dismissed"}
    mode = st.session_state.get("feed_view_mode", "all")
    if mode == "saved":
        saved = {jid for jid, act in feedback.items() if act == "saved"}
        return [r for r in recs if str(r.get("job_id")) in saved]
    return [r for r in recs if str(r.get("job_id")) not in dismissed]


def _stats_line(visible: int, meta: dict[str, Any]) -> str:
    pagination = meta.get("pagination") or {}
    total = pagination.get("total", visible)
    funnel = (meta.get("pipeline_stats") or {}).get("filter_funnel") if meta.get("pipeline_stats") else None
    if funnel and isinstance(funnel, dict):
        total_in_db = funnel.get("total_jobs", total)
        return f"Showing {visible} matches from {total} jobs, filtered from {total_in_db} total listings"
    return f"Showing {visible} matches from {total} jobs"


def render() -> None:
    """Render the personalized job feed."""
    if not st.session_state.get("onboarding_complete"):
        st.warning("Complete onboarding to see your job feed.")
        return

    name = st.session_state.get("candidate_name") or "Your"
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"## {name}'s job feed")
    with col2:
        if st.button("Edit Preferences", key="feed_edit_prefs"):
            st.session_state.current_page = "Profile"
            st.rerun()
    with col3:
        if st.button("Refresh Feed", key="feed_refresh"):
            with st.spinner("Regenerating recommendations… (may take 1–3 minutes)"):
                try:
                    clear_feed_cache()
                    _load_all_recommendations(refresh=True)
                    st.success("Feed updated from your latest profile.")
                except ApiError as exc:
                    st.error(exc.message)

    hydrate_feedback(str(st.session_state.candidate_id))

    force_refresh = st.session_state.pop("force_refresh_feed", False)
    if profile_changed_since_feed() and not force_refresh:
        st.info(
            "Your profile was updated. Refreshing your job matches… "
            "(this can take 1–3 minutes on first load)."
        )
        force_refresh = True
    try:
        recs = _load_all_recommendations(refresh=force_refresh)
    except ApiError as exc:
        st.error(exc.message)
        return

    if not recs:
        st.warning(
            "We couldn't find any matches with your current preferences. "
            "Try broadening your filters."
        )
        if st.button("Edit preferences", key="empty_edit_prefs"):
            st.session_state.current_page = "Profile"
            st.rerun()
        return

    meta = st.session_state.get("rec_meta") or {}
    filters = st.session_state.setdefault("rec_filters", {})
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        sort_key = st.selectbox(
            "Sort by",
            ["Best Match", "Newest First", "Highest Salary"],
            key="feed_sort",
        )
    with fcol2:
        filters["work_models"] = st.multiselect(
            "Work model",
            ["Remote", "Hybrid", "On-site"],
            default=filters.get("work_models", []),
            key="feed_filter_work",
        )
    with fcol3:
        from frontend.ui_settings import load_frontend_options

        filters["company_stages"] = st.multiselect(
            "Company stage",
            load_frontend_options().get("company_stages", []),
            default=filters.get("company_stages", []),
            key="feed_filter_stage",
        )
    with fcol4:
        if st.button("Clear filters", key="feed_clear_filters"):
            st.session_state.rec_filters = {}
            st.rerun()

    recs = _apply_filters(recs)
    recs = _apply_sort(recs, sort_key)
    recs = _hide_dismissed(recs)

    st.caption(_stats_line(len(recs), meta))

    if not recs:
        st.info("No jobs match your current filters. Try removing some filters.")
        if st.button("Clear filters", key="empty_clear_filters"):
            st.session_state.rec_filters = {}
            st.rerun()
        return

    strong = [r for r in recs if r.get("feed_section") == "strong_match"]
    exploring = [r for r in recs if r.get("feed_section") == "worth_exploring"]
    skills = _profile_skills()
    feedback = st.session_state.get("feedback") or {}

    if strong:
        st.markdown("### Strong Matches")
        st.caption("These roles closely align with your skills, experience, and preferences")
        for rec in strong:
            jid = str(rec.get("job_id"))
            render_job_card(
                rec,
                skills,
                candidate_id=str(st.session_state.candidate_id),
                saved_highlight=feedback.get(jid) == "saved",
            )

    if exploring:
        st.markdown("### Worth Exploring")
        st.caption(
            "These are outside your typical search but your profile suggests they could be interesting"
        )
        for rec in exploring:
            jid = str(rec.get("job_id"))
            render_job_card(
                rec,
                skills,
                candidate_id=str(st.session_state.candidate_id),
                saved_highlight=feedback.get(jid) == "saved",
            )

    meta_page = meta.get("page", 1)
    total_pages = meta.get("total_pages", 1)
    if meta_page < total_pages:
        if st.button("Load more", key="feed_load_more"):
            try:
                next_page = meta_page + 1
                per_page = get_settings().api.default_per_page
                data = get_recommendations(
                    st.session_state.candidate_id,
                    page=next_page,
                    per_page=per_page,
                )
                more = data.get("recommendations") or data.get("items") or []
                st.session_state.recommendations.extend(more)
                st.session_state.rec_meta["page"] = next_page
                st.rerun()
            except ApiError as exc:
                st.error(exc.message)
