"""Candidate profile page."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from frontend.components.feedback_buttons import hydrate_feedback
from frontend.components.preference_form import render_preference_form_api
from frontend.utils.api_client import ApiError, create_candidate, get_candidate, patch_preferences
from frontend.utils.preferences import preferences_to_api_payload


def _load_profile() -> dict[str, Any]:
    """Refresh profile from API."""
    cid = st.session_state.candidate_id
    try:
        data = get_candidate(cid)
        st.session_state.profile = data.get("profile") or {}
        st.session_state.candidate_name = data.get("name")
        st.session_state.utility_weights = data.get("utility_weights")
        return data
    except ApiError as exc:
        st.error(exc.message)
        return {}


def _pref_label(field: dict[str, Any]) -> str:
    source = field.get("source", "default")
    if source == "explicit":
        return "(you set this)"
    if source == "inferred":
        return "(inferred from your career)"
    return ""


def _render_preferences_summary(prefs: dict[str, Any]) -> None:
    """Display merged preferences."""
    if not prefs:
        st.write("No preferences set yet.")
        return
    mapping = [
        ("work_models", "Work model"),
        ("locations", "Locations"),
        ("needs_sponsorship", "Sponsorship"),
        ("salary_min", "Salary min"),
        ("salary_max", "Salary max"),
        ("company_stages", "Company stage"),
        ("target_roles", "Target roles"),
        ("target_industries", "Industries"),
        ("avoid_industries", "Avoiding"),
        ("priorities", "Priorities"),
    ]
    for key, label in mapping:
        field = prefs.get(key)
        if not isinstance(field, dict) or not field.get("value"):
            continue
        value = field["value"]
        tag = _pref_label(field)
        st.write(f"**{label}:** {value} {tag}")


def _render_skills(profile: dict[str, Any]) -> None:
    skills = profile.get("skills") or []
    if not skills:
        return
    st.markdown(f"### Your Skills ({len(skills)})")
    by_category: dict[str, list[dict[str, Any]]] = {}
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        cat = skill.get("category") or "other"
        by_category.setdefault(cat, []).append(skill)
    for category, items in sorted(by_category.items()):
        st.markdown(f"**{category.replace('_', ' ').title()}**")
        for skill in items:
            name = skill.get("name", "")
            depth = float(skill.get("depth_score") or 0)
            sources = skill.get("sources") or []
            border = "gray"
            if "github" in sources:
                border = "green"
            elif "resume" in sources:
                border = "blue"
            esco = skill.get("esco_label")
            tip = f"ESCO: {esco}" if esco else ""
            st.markdown(
                f"<span style='border:2px solid {border}; padding:4px 8px; "
                f"margin:2px; display:inline-block'>{name} ({depth:.0%})</span>",
                unsafe_allow_html=True,
            )
            if tip:
                st.caption(tip)


def _render_experience(profile: dict[str, Any]) -> None:
    experience = profile.get("experience") or []
    if not experience:
        return
    years = profile.get("total_years_experience")
    st.markdown(f"### Your Experience ({years:.1f} years)" if years else "### Your Experience")
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        st.markdown(f"**{exp.get('title', '')}** — {exp.get('company', '')}")
        st.caption(
            f"{exp.get('start_date', '')} – {exp.get('end_date') or 'present'} "
            f"({exp.get('duration_months', '')} mo)"
        )
        tags = [
            exp.get("domain"),
            exp.get("company_stage_estimate"),
            exp.get("role_type"),
        ]
        tags = [t for t in tags if t]
        if tags:
            st.write(" · ".join(tags))
        for ach in (exp.get("key_achievements") or [])[:4]:
            st.markdown(f"- {ach}")


def _render_github(profile: dict[str, Any]) -> None:
    gh = profile.get("github_summary")
    if not gh or not isinstance(gh, dict):
        st.info("GitHub not connected")
        return
    st.markdown("### GitHub Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Repos", gh.get("total_repos", 0))
    c2.metric("Followers", gh.get("followers", 0))
    c3.metric("6mo activity", gh.get("repos_last_6_months", 0))
    langs = gh.get("top_languages") or []
    if langs:
        st.bar_chart({lang: 1 for lang in langs[:8]})
    st.write(f"Activity: **{gh.get('activity_level', 'n/a')}**")
    repos = gh.get("top_repo_names") or []
    for name in repos[:5]:
        st.write(f"- {name}")


def _render_education(profile: dict[str, Any]) -> None:
    education = profile.get("education") or []
    if not education:
        return
    st.markdown("### Education")
    for edu in education:
        if isinstance(edu, dict):
            st.write(
                f"- {edu.get('degree', '')} in {edu.get('field', '')}, "
                f"{edu.get('institution', '')} ({edu.get('graduation_year', '')})"
            )


def render() -> None:
    """Render the profile page."""
    if not st.session_state.get("candidate_id"):
        st.warning("Complete onboarding to view your profile.")
        return

    hydrate_feedback(str(st.session_state.candidate_id))
    data = _load_profile()
    profile = st.session_state.get("profile") or {}
    if isinstance(profile, str):
        profile = json.loads(profile)

    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"# {profile.get('name') or st.session_state.get('candidate_name') or 'Profile'}")
        if profile.get("location"):
            st.write(profile["location"])
        if profile.get("email"):
            st.caption(profile["email"])
        c1, c2 = st.columns(2)
        if profile.get("role_archetype"):
            c1.info(profile["role_archetype"])
        if profile.get("career_trajectory"):
            c2.info(profile["career_trajectory"])
        if profile.get("summary"):
            st.write(profile["summary"])
        _render_skills(profile)
        _render_experience(profile)
        _render_github(profile)
        _render_education(profile)

    with right:
        st.markdown("### Your Preferences")
        prefs = profile.get("preferences") or data.get("preferences") or {}
        _render_preferences_summary(prefs if isinstance(prefs, dict) else {})

        if st.session_state.get("editing_preferences"):
            raw = render_preference_form_api(
                key_prefix="profile_edit",
                defaults=data.get("preferences") if isinstance(data.get("preferences"), dict) else {},
            )
            if st.button("Save preferences", key="profile_save_prefs"):
                try:
                    patch_preferences(st.session_state.candidate_id, preferences_to_api_payload(raw))
                    st.session_state.editing_preferences = False
                    st.success("Preferences saved. Refresh your feed to see new matches.")
                    _load_profile()
                    st.rerun()
                except ApiError as exc:
                    st.error(exc.message)
            if st.button("Cancel", key="profile_cancel_prefs"):
                st.session_state.editing_preferences = False
                st.rerun()
        elif st.button("Edit Preferences", key="profile_edit_prefs_btn"):
            st.session_state.editing_preferences = True
            st.rerun()

        st.markdown("### Your Activity")
        feedback = st.session_state.get("feedback") or {}
        saved = sum(1 for a in feedback.values() if a == "saved")
        dismissed = sum(1 for a in feedback.values() if a == "dismissed")
        applied = sum(1 for a in feedback.values() if a == "applied")
        st.write(f"{saved} jobs saved, {dismissed} dismissed, {applied} applied")

        uw = st.session_state.get("utility_weights") or data.get("utility_weights")
        if isinstance(uw, dict) and uw.get("adjustments_made"):
            st.write("Your feed has been personalized based on your activity:")
            for line in uw["adjustments_made"]:
                st.markdown(f"- {line}")

        if st.button("Refresh Recommendations", key="profile_refresh_recs"):
            st.session_state.current_page = "Feed"
            st.session_state.force_refresh_feed = True
            st.rerun()

        if st.button("View Saved Jobs", key="profile_view_saved"):
            st.session_state.feed_view_mode = "saved"
            st.session_state.current_page = "Feed"
            st.rerun()

        with st.expander("Dismissed jobs"):
            dismissed_ids = [jid for jid, a in feedback.items() if a == "dismissed"]
            if not dismissed_ids:
                st.write("No dismissed jobs yet.")
            else:
                for jid in dismissed_ids:
                    st.write(f"Job {jid[:8]}…")
                    if st.button("Show again", key=f"undismiss_{jid}"):
                        feedback.pop(jid, None)
                        st.session_state.feedback = feedback
                        st.rerun()

        st.markdown("### Re-upload Resume")
        uploaded = st.file_uploader("New resume", type=["pdf", "docx"], key="profile_resume")
        if uploaded and st.button("Update profile", key="profile_reupload"):
            try:
                result = create_candidate(
                    resume_bytes=uploaded.getvalue(),
                    filename=uploaded.name,
                    github_username=st.session_state.get("github_username"),
                    preferences=st.session_state.get("preferences"),
                )
                st.session_state.candidate_id = result["id"]
                st.session_state.profile = result.get("profile")
                st.success("Profile updated. Refresh your feed to see new recommendations.")
                _load_profile()
            except ApiError as exc:
                st.error(exc.message)
