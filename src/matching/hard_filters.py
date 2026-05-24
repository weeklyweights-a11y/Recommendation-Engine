"""Non-negotiable SQL filters applied before retrieval and scoring."""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from src.api.schemas.candidate import MergedPreferences
from src.db.models import Job
from src.matching.preference_utils import (
    pref_bool,
    pref_int,
    pref_list,
    should_apply_company_size_filter,
    should_apply_company_stage_filter,
    should_apply_work_model_filter,
)
from src.matching.schemas import FilterFunnel

logger = logging.getLogger(__name__)

_FUNNEL_STAGES: list[tuple[str, str]] = [
    ("location", "after_location"),
    ("work_model", "after_work_model"),
    ("sponsorship", "after_sponsorship"),
    ("salary", "after_salary"),
    ("company_size", "after_company_size"),
    ("company_stage", "after_company_stage"),
    ("industry_exclusion", "after_industry_exclusion"),
]


class HardFilter:
    """Filter jobs by candidate non-negotiable preferences."""

    def __init__(
        self,
        session: Session,
        settings: Optional[Settings] = None,
    ) -> None:
        """Attach database session and settings."""
        self._session = session
        self._settings = settings or get_settings()

    def _embedded_base(self) -> Select[tuple[UUID]]:
        """Base query: only embedded jobs."""
        return select(Job.id).where(Job.is_embedded.is_(True))

    def _apply_location(
        self,
        stmt: Select[tuple[UUID]],
        locations: list[str],
    ) -> Select[tuple[UUID]]:
        if not locations:
            return stmt
        patterns = [loc.lower() for loc in locations]
        location_clauses = [
            func.lower(Job.location).contains(pattern) for pattern in patterns
        ]
        remote_clause = func.lower(Job.remote_type) == "remote"
        return stmt.where(or_(remote_clause, or_(*location_clauses)))

    def _apply_work_model(
        self,
        stmt: Select[tuple[UUID]],
        work_models: list[str],
    ) -> Select[tuple[UUID]]:
        if not should_apply_work_model_filter(work_models):
            return stmt
        normalized = [m.lower() for m in work_models]
        return stmt.where(
            or_(
                func.lower(Job.remote_type).in_(normalized),
                Job.remote_type.is_(None),
            ),
        )

    def _apply_sponsorship(
        self,
        stmt: Select[tuple[UUID]],
        needs_sponsorship: bool,
    ) -> Select[tuple[UUID]]:
        if not needs_sponsorship:
            return stmt
        return stmt.where(
            or_(
                Job.sponsorship_available.is_(True),
                Job.sponsorship_available.is_(None),
            ),
        )

    def _apply_salary(
        self,
        stmt: Select[tuple[UUID]],
        salary_min: int,
    ) -> Select[tuple[UUID]]:
        return stmt.where(
            or_(
                Job.salary_max >= salary_min,
                Job.salary_max.is_(None),
            ),
        )

    def _apply_company_size(
        self,
        stmt: Select[tuple[UUID]],
        sizes: list[str],
    ) -> Select[tuple[UUID]]:
        if not should_apply_company_size_filter(sizes):
            return stmt
        normalized = [s.lower() for s in sizes]
        return stmt.where(
            or_(
                func.lower(Job.company_size).in_(normalized),
                Job.company_size.is_(None),
            ),
        )

    def _apply_company_stage(
        self,
        stmt: Select[tuple[UUID]],
        stages: list[str],
    ) -> Select[tuple[UUID]]:
        if not should_apply_company_stage_filter(stages):
            return stmt
        normalized = [s.lower().replace("_", "-") for s in stages]
        return stmt.where(
            or_(
                func.lower(Job.company_stage).in_(normalized),
                Job.company_stage.is_(None),
            ),
        )

    def _apply_industry_exclusion(
        self,
        stmt: Select[tuple[UUID]],
        avoid: list[str],
    ) -> Select[tuple[UUID]]:
        if not avoid:
            return stmt
        normalized = [a.lower() for a in avoid]
        for industry in normalized:
            stmt = stmt.where(
                or_(
                    Job.industry.is_(None),
                    ~func.lower(Job.industry).contains(industry),
                ),
            )
        return stmt

    def _build_filtered_query(self, preferences: MergedPreferences) -> Select[tuple[UUID]]:
        """Compose filter stages on embedded jobs."""
        stmt = self._embedded_base()
        for _, _, apply_fn in self._stage_appliers(preferences):
            stmt = apply_fn(stmt)
        return stmt

    def _count_query(self, stmt: Select[tuple[UUID]]) -> int:
        count_stmt = select(func.count()).select_from(stmt.subquery())
        return int(self._session.scalar(count_stmt) or 0)

    def filter_jobs(self, preferences: MergedPreferences) -> set[str]:
        """Return job IDs passing all applicable hard constraints."""
        total = self._count_query(self._embedded_base())
        stmt = self._build_filtered_query(preferences)
        ids = self._session.scalars(stmt).all()
        result = {str(job_id) for job_id in ids}
        final = len(result)

        logger.info(
            "Hard filter funnel: total_embedded=%s final=%s",
            total,
            final,
        )
        warn_at = self._settings.hard_filter.min_results_warn
        if final < warn_at:
            logger.warning(
                "Hard filter returned %s jobs (below %s); preferences may be too restrictive",
                final,
                warn_at,
            )
        return result

    def _stage_appliers(
        self,
        preferences: MergedPreferences,
    ) -> list[tuple[str, str, object]]:
        """Return ordered (stage_name, funnel_attr, apply_fn) for active filters."""
        appliers: list[tuple[str, str, object]] = []
        locations = pref_list(preferences.locations)
        if locations:
            appliers.append(
                ("location", "after_location", lambda s: self._apply_location(s, locations)),
            )
        work_models = pref_list(preferences.work_models)
        if should_apply_work_model_filter(work_models):
            appliers.append(
                (
                    "work_model",
                    "after_work_model",
                    lambda s: self._apply_work_model(s, work_models),
                ),
            )
        needs = pref_bool(preferences.needs_sponsorship)
        if needs is True:
            appliers.append(
                (
                    "sponsorship",
                    "after_sponsorship",
                    lambda s: self._apply_sponsorship(s, True),
                ),
            )
        salary_min = pref_int(preferences.salary_min)
        if salary_min is not None:
            appliers.append(
                ("salary", "after_salary", lambda s: self._apply_salary(s, salary_min)),
            )
        sizes = pref_list(preferences.company_sizes)
        if should_apply_company_size_filter(sizes):
            appliers.append(
                (
                    "company_size",
                    "after_company_size",
                    lambda s: self._apply_company_size(s, sizes),
                ),
            )
        stages_list = pref_list(preferences.company_stages)
        if should_apply_company_stage_filter(stages_list):
            appliers.append(
                (
                    "company_stage",
                    "after_company_stage",
                    lambda s: self._apply_company_stage(s, stages_list),
                ),
            )
        avoid = pref_list(preferences.avoid_industries)
        if avoid:
            appliers.append(
                (
                    "industry_exclusion",
                    "after_industry_exclusion",
                    lambda s: self._apply_industry_exclusion(s, avoid),
                ),
            )
        return appliers

    def get_filter_funnel(self, preferences: MergedPreferences) -> FilterFunnel:
        """Run filters incrementally for analytics (slower than filter_jobs)."""
        total = self._count_query(self._embedded_base())
        funnel = FilterFunnel(
            total_jobs=total,
            after_location=total,
            after_work_model=total,
            after_sponsorship=total,
            after_salary=total,
            after_company_size=total,
            after_company_stage=total,
            after_industry_exclusion=total,
        )
        stmt = self._embedded_base()
        prev = total
        drops: dict[str, int] = {}
        appliers = self._stage_appliers(preferences)

        for stage_name, attr, apply_fn in appliers:
            stmt = apply_fn(stmt)
            current = self._count_query(stmt)
            setattr(funnel, attr, current)
            drops[stage_name] = prev - current
            prev = current

        # Propagate last count through inactive stage slots
        last_count = total
        for stage_name, attr in _FUNNEL_STAGES:
            val = getattr(funnel, attr)
            if stage_name in drops:
                last_count = val
            else:
                setattr(funnel, attr, last_count)

        funnel.final_count = self._count_query(self._build_filtered_query(preferences))
        funnel.most_restrictive_filter = (
            max(drops, key=drops.get) if drops else "none"  # type: ignore[arg-type]
        )
        logger.info(
            "Filter funnel: total=%s final=%s most_restrictive=%s",
            funnel.total_jobs,
            funnel.final_count,
            funnel.most_restrictive_filter,
        )
        return funnel
