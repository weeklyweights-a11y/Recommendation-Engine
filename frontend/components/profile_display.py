"""Shared profile summary widgets for onboarding and profile pages."""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st


def _depth_color(score: float) -> str:
    if score >= 0.7:
        return "green"
    if score >= 0.4:
        return "orange"
    return "gray"


def render_skills_section(profile: dict[str, Any], *, max_display: Optional[int] = None) -> None:
    """Show all extracted skills grouped by source hints."""
    skills = profile.get("skills") or []
    if not skills:
        st.caption("No skills extracted yet.")
        return
    total = len(skills)
    st.markdown(f"**Skills ({total})**")
    items = skills if max_display is None else skills[:max_display]
    by_source: dict[str, list[dict[str, Any]]] = {
        "From resume": [],
        "From GitHub": [],
        "Both / other": [],
    }
    for skill in items:
        if not isinstance(skill, dict):
            continue
        sources = [str(s).lower() for s in (skill.get("sources") or [])]
        if "github" in sources and "resume" in sources:
            by_source["Both / other"].append(skill)
        elif "github" in sources:
            by_source["From GitHub"].append(skill)
        elif "resume" in sources:
            by_source["From resume"].append(skill)
        else:
            by_source["Both / other"].append(skill)

    for group, group_skills in by_source.items():
        if not group_skills:
            continue
        st.caption(group)
        chips = []
        for skill in group_skills:
            name = skill.get("name", "")
            depth = float(skill.get("depth_score") or 0)
            prof = skill.get("proficiency", "")
            color = _depth_color(depth)
            chips.append(f":{color}[{name} ({prof or 'n/a'}, {depth:.0%})]")
        st.markdown(" ".join(chips))

    gh = profile.get("github_summary")
    if gh and isinstance(gh, dict):
        inferred = gh.get("inferred_skills") or []
        if inferred:
            st.caption("GitHub-inferred (may overlap with skills above)")
            st.write(", ".join(inferred))


def render_experience_section(
    profile: dict[str, Any],
    *,
    limit: Optional[int] = None,
) -> None:
    """Show work history with role, dates, description, and achievements."""
    experience = profile.get("experience") or []
    if not experience:
        st.caption("No experience entries extracted.")
        return
    years = profile.get("total_years_experience")
    heading = "**Experience**"
    if years:
        heading = f"**Experience ({years:.1f} years total)**"
    st.markdown(heading)
    rows = experience if limit is None else experience[:limit]
    for idx, exp in enumerate(rows):
        if not isinstance(exp, dict):
            continue
        title = exp.get("title", "Role")
        company = exp.get("company", "Company")
        start = exp.get("start_date", "")
        end = exp.get("end_date") or "present"
        months = exp.get("duration_months")
        duration = f" · {months} mo" if months else ""
        tags = [
            exp.get("domain"),
            exp.get("company_stage_estimate"),
            exp.get("role_type"),
        ]
        tag_str = " · ".join(t for t in tags if t)
        with st.expander(f"{title} @ {company} ({start} – {end}){duration}", expanded=idx == 0):
            if tag_str:
                st.caption(tag_str)
            desc = (exp.get("description") or "").strip()
            if desc:
                st.markdown("**What they did**")
                st.write(desc)
            achievements = exp.get("key_achievements") or []
            if achievements:
                st.markdown("**Key achievements**")
                for ach in achievements:
                    st.markdown(f"- {ach}")


def render_github_summary(profile: dict[str, Any]) -> None:
    """Show GitHub overview when present."""
    gh = profile.get("github_summary")
    if not gh or not isinstance(gh, dict):
        st.caption("GitHub not connected")
        return
    st.markdown("**GitHub**")
    st.write(
        f"Repos: {gh.get('total_repos', 0)} · "
        f"Followers: {gh.get('followers', 0)} · "
        f"Activity: {gh.get('activity_level', 'n/a')}"
    )
    langs = gh.get("top_languages") or []
    if langs:
        st.write("Languages: " + ", ".join(langs))
    assessment = gh.get("overall_assessment")
    if assessment:
        st.write(assessment)
