"""Tests for GitHub profile fetcher."""

import re
from datetime import datetime, timedelta, timezone

import pytest
import respx

from config.settings import get_settings
from src.ingestion.exceptions import GitHubUserNotFoundError
from src.ingestion.github_fetcher import (
    _aggregate_languages,
    _complexity_score,
    _keep_repo,
    fetch_github_profile,
    sanitize_github_username,
)


@pytest.fixture
def github_base() -> str:
    return get_settings().ingestion.github_api_base_url.rstrip("/")


@pytest.mark.asyncio
async def test_fetch_github_profile_success(github_base: str) -> None:
    """Happy path returns profile with languages and repos."""
    username = "octocat"
    recent = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with respx.mock(using="httpx") as mock:
        mock.get(f"{github_base}/users/{username}").respond(
            200,
            json={
                "login": username,
                "name": "Octo Cat",
                "bio": "Dev",
                "public_repos": 2,
                "followers": 10,
                "following": 2,
                "created_at": "2015-01-01T00:00:00Z",
            },
        )
        mock.get(url=re.compile(rf"{re.escape(github_base)}/users/{username}/repos.*")).respond(
            200,
            json=[
                {
                    "name": "hello",
                    "description": "Demo",
                    "language": "Python",
                    "stargazers_count": 6,
                    "fork": False,
                    "pushed_at": recent,
                    "size": 2000,
                    "topics": ["fastapi"],
                    "has_wiki": False,
                    "default_branch": "main",
                },
            ],
        )
        mock.get(f"{github_base}/repos/{username}/hello/languages").respond(
            200,
            json={"Python": 900, "Shell": 100},
        )
        mock.get(f"{github_base}/repos/{username}/hello/readme").respond(
            200,
            json={"content": "IyBIZW1v", "encoding": "base64"},
        )
        mock.get(f"{github_base}/repos/{username}/hello/contents/").respond(
            200,
            json=[
                {"name": "Dockerfile"},
                {"name": "tests"},
                {"name": "requirements.txt"},
            ],
        )

        profile = await fetch_github_profile(username)
        assert profile.username == username
        assert profile.followers == 10
        assert profile.following == 2
        assert profile.languages_distribution["Python"] == pytest.approx(0.9, rel=0.01)
        assert profile.top_repos[0].production_signals


@pytest.mark.asyncio
async def test_rate_limit_uses_partial_repos(github_base: str) -> None:
    """When rate limited mid-fetch, return profile using repos already fetched."""
    username = "partialuser"
    recent = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    repos = [
        {
            "name": f"repo-{i}",
            "description": f"Repo {i}",
            "language": "Python",
            "stargazers_count": i,
            "fork": False,
            "pushed_at": recent,
            "size": 100,
            "topics": [],
            "has_wiki": False,
            "default_branch": "main",
        }
        for i in range(5)
    ]
    with respx.mock(using="httpx") as mock:
        mock.get(f"{github_base}/users/{username}").respond(
            200,
            json={
                "login": username,
                "public_repos": 5,
                "followers": 1,
                "following": 0,
                "created_at": "2020-01-01T00:00:00Z",
            },
        )
        mock.get(url=re.compile(rf"{re.escape(github_base)}/users/{username}/repos.*")).respond(
            200,
            json=repos,
        )
        mock.get(url=re.compile(rf"{re.escape(github_base)}/repos/{username}/repo-0/languages")).respond(
            200,
            json={"Python": 100},
        )
        mock.get(url=re.compile(rf"{re.escape(github_base)}/repos/{username}/repo-0/readme")).respond(404)
        mock.get(url=re.compile(rf"{re.escape(github_base)}/repos/{username}/repo-0/contents/")).respond(
            200,
            json=[],
        )
        mock.get(url=re.compile(rf"{re.escape(github_base)}/repos/{username}/repo-1/languages")).respond(
            403,
        )

        profile = await fetch_github_profile(username)
        assert profile.username == username
        assert len(profile.top_repos) >= 3


@pytest.mark.asyncio
async def test_user_not_found(github_base: str) -> None:
    """404 on user raises GitHubUserNotFoundError."""
    with respx.mock(using="httpx") as mock:
        mock.get(f"{github_base}/users/missing").respond(404)
        with pytest.raises(GitHubUserNotFoundError):
            await fetch_github_profile("missing")


@pytest.mark.asyncio
async def test_zero_repos_returns_empty_profile(github_base: str) -> None:
    """User with no repos still returns a valid profile."""
    username = "emptyuser"
    with respx.mock(using="httpx") as mock:
        mock.get(f"{github_base}/users/{username}").respond(
            200,
            json={
                "login": username,
                "public_repos": 0,
                "followers": 0,
                "following": 0,
                "created_at": "2020-01-01T00:00:00Z",
            },
        )
        mock.get(url=re.compile(rf"{re.escape(github_base)}/users/{username}/repos.*")).respond(
            200,
            json=[],
        )
        profile = await fetch_github_profile(username)
        assert profile.activity_metrics.total_repos == 0
        assert profile.overall_assessment == "inactive"


def test_fork_filter_keeps_starred_fork() -> None:
    """Forks with stars are retained."""
    settings = get_settings()
    assert _keep_repo({"fork": True, "stargazers_count": 3, "pushed_at": "2020-01-01T00:00:00Z"}, settings)


def test_fork_filter_drops_stale_fork() -> None:
    """Stale forks without stars are dropped."""
    settings = get_settings()
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat().replace("+00:00", "Z")
    assert not _keep_repo({"fork": True, "stargazers_count": 0, "pushed_at": old}, settings)


def test_language_distribution() -> None:
    """Language bytes aggregate to percentages."""
    dist = _aggregate_languages([{"Python": 75, "Go": 25}])
    assert dist["Python"] == 0.75
    assert dist["Go"] == 0.25


def test_complexity_high() -> None:
    """High complexity when many signals present."""
    assert _complexity_score(True, True, True, True, 5000, 10) == "high"


def test_invalid_username() -> None:
    """Invalid usernames are rejected."""
    with pytest.raises(ValueError):
        sanitize_github_username("bad username!")
