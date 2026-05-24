"""Tests for HardFilter."""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import JSON, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.api.schemas.candidate import MergedPreferences, PreferenceField
from src.db.models import Base, Job

# SQLite tests: use JSON instead of PostgreSQL JSONB on job columns.
for _col in Job.__table__.columns:
    if "JSON" in type(_col.type).__name__:
        _col.type = JSON()
from src.matching.hard_filters import HardFilter


def _prefs(**kwargs: object) -> MergedPreferences:
    """Build MergedPreferences with explicit source."""
    data: dict[str, PreferenceField] = {}
    for key, value in kwargs.items():
        data[key] = PreferenceField(value=value, source="explicit")
    return MergedPreferences(**data)


@pytest.fixture
def filter_session() -> Session:
    """In-memory DB with seeded jobs."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[Job.__table__])
    factory = sessionmaker(bind=engine)
    session = factory()
    now = datetime.now(timezone.utc)
    jobs = [
        Job(
            id=uuid.uuid4(),
            title="NYC Onsite",
            company="A",
            description="d",
            location="New York, NY",
            remote_type="onsite",
            salary_max=180000,
            sponsorship_available=True,
            company_size="11-50",
            company_stage="seed",
            industry="fintech",
            is_embedded=True,
            created_at=now,
            updated_at=now,
        ),
        Job(
            id=uuid.uuid4(),
            title="Remote US",
            company="B",
            description="d",
            location="USA",
            remote_type="remote",
            salary_max=200000,
            sponsorship_available=None,
            company_size="1-10",
            company_stage="series-a",
            industry="saas",
            is_embedded=True,
            created_at=now,
            updated_at=now,
        ),
        Job(
            id=uuid.uuid4(),
            title="Low pay",
            company="C",
            description="d",
            location="Austin, TX",
            remote_type="hybrid",
            salary_max=90000,
            sponsorship_available=False,
            company_size="201-1000",
            company_stage="enterprise",
            industry="defense",
            is_embedded=True,
            created_at=now,
            updated_at=now,
        ),
        Job(
            id=uuid.uuid4(),
            title="Not embedded",
            company="D",
            description="d",
            location="New York, NY",
            remote_type="remote",
            is_embedded=False,
            created_at=now,
            updated_at=now,
        ),
        Job(
            id=uuid.uuid4(),
            title="Unknown fields",
            company="E",
            description="d",
            location=None,
            remote_type=None,
            salary_max=None,
            sponsorship_available=None,
            company_size=None,
            company_stage=None,
            industry=None,
            is_embedded=True,
            created_at=now,
            updated_at=now,
        ),
    ]
    session.add_all(jobs)
    session.commit()
    session._seed_ids = {str(j.id): j for j in jobs}  # type: ignore[attr-defined]
    yield session
    session.close()


def test_no_preferences_returns_all_embedded(filter_session: Session) -> None:
    hf = HardFilter(filter_session)
    result = hf.filter_jobs(MergedPreferences())
    assert len(result) == 4
    not_embedded = [str(j.id) for j in filter_session._seed_ids.values() if not j.is_embedded]  # type: ignore[attr-defined]
    assert not_embedded[0] not in result


def test_location_filter_nyc_and_remote(filter_session: Session) -> None:
    hf = HardFilter(filter_session)
    prefs = _prefs(locations=["New York"])
    result = hf.filter_jobs(prefs)
    titles = {filter_session._seed_ids[jid].title for jid in result}  # type: ignore[attr-defined]
    assert "NYC Onsite" in titles
    assert "Remote US" in titles
    assert "Low pay" not in titles


def test_sponsorship_filter(filter_session: Session) -> None:
    hf = HardFilter(filter_session)
    prefs = _prefs(needs_sponsorship=True)
    result = hf.filter_jobs(prefs)
    for jid in result:
        job = filter_session._seed_ids[jid]  # type: ignore[attr-defined]
        assert job.sponsorship_available is not False


def test_salary_filter(filter_session: Session) -> None:
    hf = HardFilter(filter_session)
    prefs = _prefs(salary_min=150000)
    result = hf.filter_jobs(prefs)
    for jid in result:
        job = filter_session._seed_ids[jid]  # type: ignore[attr-defined]
        assert job.salary_max is None or job.salary_max >= 150000


def test_industry_exclusion(filter_session: Session) -> None:
    hf = HardFilter(filter_session)
    prefs = _prefs(avoid_industries=["defense"])
    result = hf.filter_jobs(prefs)
    for jid in result:
        job = filter_session._seed_ids[jid]  # type: ignore[attr-defined]
        assert job.industry is None or "defense" not in (job.industry or "").lower()


def test_combined_filters(filter_session: Session) -> None:
    hf = HardFilter(filter_session)
    prefs = _prefs(
        locations=["New York"],
        salary_min=100000,
        avoid_industries=["defense"],
    )
    result = hf.filter_jobs(prefs)
    remote_id = next(
        str(j.id)
        for j in filter_session._seed_ids.values()  # type: ignore[attr-defined]
        if j.title == "Remote US"
    )
    assert remote_id in result


def test_overly_restrictive_empty(filter_session: Session) -> None:
    hf = HardFilter(filter_session)
    prefs = _prefs(
        locations=["Antarctica"],
        salary_min=500000,
    )
    result = hf.filter_jobs(prefs)
    assert result == set()


def test_filter_funnel_most_restrictive(filter_session: Session) -> None:
    hf = HardFilter(filter_session)
    prefs = _prefs(avoid_industries=["defense"], salary_min=150000)
    funnel = hf.get_filter_funnel(prefs)
    assert funnel.total_jobs == 4
    assert funnel.final_count <= funnel.total_jobs
    assert funnel.most_restrictive_filter in {
        "none",
        "location",
        "work_model",
        "sponsorship",
        "salary",
        "company_size",
        "company_stage",
        "industry_exclusion",
    }
