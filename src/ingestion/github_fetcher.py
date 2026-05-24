"""Fetch and analyze public GitHub profiles."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional
import httpx

from config.settings import Settings, get_settings
from src.ingestion.exceptions import GitHubRateLimitedError, GitHubUserNotFoundError
from src.ingestion.schemas import ActivityMetrics, GitHubProfile, RepoAnalysis

logger = logging.getLogger(__name__)

_USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,37}[a-zA-Z0-9])?$")


def sanitize_github_username(username: str, settings: Settings | None = None) -> str:
    """Validate and normalize a GitHub username."""
    cleaned = username.strip().lstrip("@")
    max_len = (settings or get_settings()).ingestion.github_username_max_length
    if not cleaned or len(cleaned) > max_len or not _USERNAME_PATTERN.match(cleaned):
        raise ValueError(f"Invalid GitHub username: {username}")
    return cleaned


def _headers(settings: Settings) -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = (settings.github.github_token or "").strip()
    placeholders = {"", "your_github_token_here", "ghp_placeholder", "none", "null"}
    if token.lower() not in placeholders and not token.startswith("your_"):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _parse_github_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            return parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None


def _relative_time(dt: datetime | None) -> str:
    if dt is None:
        return "unknown"
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    days = delta.days
    if days < 14:
        return f"{max(days, 1)} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    if days < 365:
        return f"{days // 30} months ago"
    return f"{days // 365} years ago"


def _parse_link_header(link: str | None) -> dict[str, str]:
    if not link:
        return {}
    links: dict[str, str] = {}
    for part in link.split(","):
        section = part.strip().split(";")
        if len(section) < 2:
            continue
        url = section[0].strip("<>")
        rel = section[1].strip().replace('rel="', "").replace('"', "")
        links[rel] = url
    return links


async def _request_json(
    client: httpx.AsyncClient,
    url: str,
    settings: Settings,
) -> Any:
    """GET JSON with rate-limit handling and exponential backoff."""
    max_retries = settings.ingestion.github_rate_limit_max_retries
    warn_threshold = settings.ingestion.github_rate_limit_warn_threshold

    for attempt in range(max_retries + 1):
        response = await client.get(url)
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining is not None:
            try:
                if int(remaining) < warn_threshold:
                    logger.warning("GitHub rate limit low: %s remaining", remaining)
            except ValueError:
                pass

        if response.status_code == 404:
            return None
        if response.status_code in (403, 429):
            max_wait = settings.ingestion.github_rate_limit_max_wait_seconds
            if attempt < max_retries:
                reset = response.headers.get("X-RateLimit-Reset")
                wait_seconds = min(2**attempt, max_wait)
                if reset:
                    try:
                        reset_dt = datetime.fromtimestamp(int(reset), tz=timezone.utc)
                        wait_seconds = min(
                            max_wait,
                            max(1, int((reset_dt - datetime.now(timezone.utc)).total_seconds())),
                        )
                    except ValueError:
                        pass
                logger.warning("GitHub rate limited; sleeping %ss", wait_seconds)
                await asyncio.sleep(wait_seconds)
                continue
            raise GitHubRateLimitedError(f"GitHub rate limited: {url}")

        response.raise_for_status()
        return response.json()

    raise GitHubRateLimitedError(f"GitHub rate limited after retries: {url}")


async def _fetch_all_repos(
    client: httpx.AsyncClient,
    username: str,
    settings: Settings,
) -> list[dict[str, Any]]:
    base = settings.ingestion.github_api_base_url.rstrip("/")
    url: str | None = f"{base}/users/{username}/repos?sort=updated&per_page=100&type=owner"
    repos: list[dict[str, Any]] = []
    max_repos = settings.ingestion.github_max_repos

    while url and len(repos) < max_repos:
        response = await client.get(url)
        if response.status_code == 404:
            return repos
        if response.status_code in (403, 429):
            if repos:
                logger.warning(
                    "GitHub rate limited listing repos; using %s fetched so far",
                    len(repos),
                )
                return repos[:max_repos]
            raise GitHubRateLimitedError("GitHub rate limited fetching repo list")
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            break
        repos.extend(batch)
        links = _parse_link_header(response.headers.get("Link"))
        url = links.get("next")
        if len(repos) >= max_repos:
            repos = repos[:max_repos]
            break
    return repos


def _keep_repo(repo: dict[str, Any], settings: Settings) -> bool:
    if not repo.get("fork"):
        return True
    if int(repo.get("stargazers_count") or 0) > 0:
        return True
    pushed = _parse_github_datetime(repo.get("pushed_at"))
    if pushed is None:
        return False
    cutoff_days = settings.ingestion.github_fork_recency_days
    age_days = (datetime.now(timezone.utc) - pushed).days
    return age_days <= cutoff_days


async def _fetch_languages(
    client: httpx.AsyncClient,
    username: str,
    repo_name: str,
    settings: Settings,
) -> dict[str, int]:
    base = settings.ingestion.github_api_base_url.rstrip("/")
    url = f"{base}/repos/{username}/{repo_name}/languages"
    data = await _request_json(client, url, settings)
    return data if isinstance(data, dict) else {}


async def _fetch_readme(
    client: httpx.AsyncClient,
    username: str,
    repo_name: str,
    settings: Settings,
) -> Optional[str]:
    base = settings.ingestion.github_api_base_url.rstrip("/")
    url = f"{base}/repos/{username}/{repo_name}/readme"
    data = await _request_json(client, url, settings)
    if not data or not isinstance(data, dict):
        return None
    content = data.get("content")
    if not content:
        return None
    try:
        decoded = base64.b64decode(content).decode("utf-8", errors="replace")
        return decoded[:2000]
    except (ValueError, TypeError):
        return None


async def _detect_production_signals(
    client: httpx.AsyncClient,
    username: str,
    repo_name: str,
    settings: Settings,
) -> tuple[bool, bool, bool, bool]:
    base = settings.ingestion.github_api_base_url.rstrip("/")
    url = f"{base}/repos/{username}/{repo_name}/contents/"
    data = await _request_json(client, url, settings)
    names: set[str] = set()
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("name"):
                names.add(str(item["name"]).lower())

    has_docker = any(
        name in names or name.startswith("dockerfile") for name in names
    ) or "docker-compose.yml" in names
    has_ci = (
        ".github" in names
        or ".circleci" in names
        or "jenkinsfile" in names
    )
    has_tests = "tests" in names or any(
        name.startswith("test_") or name.endswith("_test.py") for name in names
    )
    has_deps = any(
        name in names
        for name in ("requirements.txt", "pyproject.toml", "package.json", "go.mod")
    )
    return has_docker, has_ci, has_tests, has_deps


def _complexity_score(
    has_tests: bool,
    has_docker: bool,
    has_ci: bool,
    multi_language: bool,
    size: int,
    stars: int,
) -> str:
    score = sum(
        [
            has_tests,
            has_docker,
            has_ci,
            multi_language,
            size > 1000,
            stars > 5,
        ],
    )
    if score <= 1:
        return "low"
    if score <= 3:
        return "medium"
    return "high"


def _aggregate_languages(repo_langs: list[dict[str, int]]) -> dict[str, float]:
    totals: dict[str, int] = {}
    for lang_map in repo_langs:
        for lang, count in lang_map.items():
            totals[lang] = totals.get(lang, 0) + int(count)
    grand = sum(totals.values()) or 1
    return {lang: round(count / grand, 4) for lang, count in totals.items()}


def _infer_skills_from_github(
    languages: dict[str, float],
    repos: list[dict[str, Any]],
    max_skills: int = 0,
) -> list[str]:
    """Collect languages, topics, and repo languages from GitHub metadata."""
    seen: set[str] = set()
    skills: list[str] = []

    def add(raw: str) -> None:
        cleaned = str(raw).strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        skills.append(cleaned)

    for lang in languages:
        add(lang)
    for repo in repos:
        lang = repo.get("language")
        if lang:
            add(str(lang))
        for topic in repo.get("topics") or []:
            if topic:
                add(str(topic).replace("-", " "))
    if max_skills > 0:
        return skills[:max_skills]
    return skills


def _analysis_from_listing(repo: dict[str, Any]) -> RepoAnalysis:
    """Build a lightweight repo analysis from the /repos list payload (no extra API calls)."""
    pushed = _parse_github_datetime(repo.get("pushed_at"))
    language = repo.get("language")
    languages = [str(language)] if language else []
    return RepoAnalysis(
        name=str(repo.get("name", "")),
        complexity="low",
        languages=languages,
        description=str(repo.get("description") or ""),
        stars=int(repo.get("stargazers_count") or 0),
        last_active=_relative_time(pushed),
        readme_summary=None,
        production_signals=[],
        topics=[str(t) for t in repo.get("topics") or []],
        has_wiki=bool(repo.get("has_wiki")),
        default_branch=str(repo.get("default_branch") or "main"),
    )


def _ensure_min_repo_analyses(
    analyses: list[RepoAnalysis],
    filtered: list[dict[str, Any]],
    min_count: int,
) -> list[RepoAnalysis]:
    """Backfill repo analyses from listing metadata when detail fetches were rate limited."""
    if len(analyses) >= min_count:
        return analyses
    seen = {analysis.name for analysis in analyses}
    for repo in filtered:
        if len(analyses) >= min_count:
            break
        name = str(repo.get("name", ""))
        if name and name not in seen:
            analyses.append(_analysis_from_listing(repo))
            seen.add(name)
    return analyses


def _overall_assessment(repos: list[dict[str, Any]], repos_6mo: int) -> str:
    if not repos:
        return "inactive"
    if len(repos) < 5:
        return "beginner"
    if repos_6mo == 0:
        return "inactive"
    fork_ratio = sum(1 for repo in repos if repo.get("fork")) / max(len(repos), 1)
    if fork_ratio > 0.5:
        return "contributor"
    if len(repos) <= 8 and sum(int(repo.get("stargazers_count") or 0) for repo in repos) > 20:
        return "portfolio_focused"
    return "active_builder"


def format_github_for_llm(profile: GitHubProfile, settings: Settings | None = None) -> str:
    """Build a GitHub summary for the LLM prompt (all languages and inferred skills)."""
    cfg = (settings or get_settings()).ingestion
    skill_list = ", ".join(profile.inferred_skills) if profile.inferred_skills else "none"
    lines = [
        f"User: {profile.username}",
        f"Languages (distribution): {profile.languages_distribution}",
        f"Assessment: {profile.overall_assessment}",
        f"All inferred skills from GitHub ({len(profile.inferred_skills)}): {skill_list}",
        "Include every language and topic above in the skills array if not already on the resume.",
    ]
    for repo in profile.top_repos[:15]:
        langs = ", ".join(repo.languages) if repo.languages else "n/a"
        topics = ", ".join(repo.topics[:12]) if repo.topics else "n/a"
        lines.append(
            f"Repo {repo.name}: langs=[{langs}] topics=[{topics}] "
            f"desc={(repo.description[:100] if repo.description else '')} "
            f"signals={repo.production_signals}",
        )
    text = "\n".join(lines)
    return text[: cfg.github_llm_summary_max_chars]


async def fetch_github_profile(
    username: str,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
) -> GitHubProfile:
    """Fetch and analyze a GitHub user's public profile."""
    cfg = settings or get_settings()
    user = sanitize_github_username(username, settings=cfg)
    base = cfg.ingestion.github_api_base_url.rstrip("/")

    owns_client = client is None
    http_client = client or httpx.AsyncClient(
        headers=_headers(cfg),
        timeout=30.0,
    )

    try:
        user_data = await _request_json(http_client, f"{base}/users/{user}", cfg)
        if user_data is None:
            raise GitHubUserNotFoundError(f"GitHub user not found: {user}")

        raw_repos = await _fetch_all_repos(http_client, user, cfg)
        filtered = [repo for repo in raw_repos if _keep_repo(repo, cfg)]
        filtered.sort(
            key=lambda r: _parse_github_datetime(r.get("pushed_at")) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        top_lang_n = cfg.ingestion.github_top_repos_languages
        top_readme_n = cfg.ingestion.github_top_repos_readme
        top_signals_n = cfg.ingestion.github_top_repos_signals

        repo_lang_maps: list[dict[str, int]] = []
        analyses: list[RepoAnalysis] = []
        min_partial = cfg.ingestion.github_min_repos_partial

        try:
            for idx, repo in enumerate(filtered[:top_signals_n]):
                name = str(repo.get("name", ""))
                try:
                    langs: dict[str, int] = {}
                    if idx < top_lang_n:
                        langs = await _fetch_languages(http_client, user, name, cfg)
                        if langs:
                            repo_lang_maps.append(langs)

                    readme_full: Optional[str] = None
                    if idx < top_readme_n:
                        readme_full = await _fetch_readme(http_client, user, name, cfg)

                    has_docker = has_ci = has_tests = has_deps = False
                    if idx < top_signals_n:
                        try:
                            has_docker, has_ci, has_tests, has_deps = await _detect_production_signals(
                                http_client,
                                user,
                                name,
                                cfg,
                            )
                        except httpx.HTTPError as exc:
                            logger.warning(
                                "Production signal check failed for %s/%s: %s",
                                user,
                                name,
                                exc,
                            )

                    signals: list[str] = []
                    if has_docker:
                        signals.append("docker")
                    if has_ci:
                        signals.append("ci")
                    if has_tests:
                        signals.append("tests")
                    if has_deps:
                        signals.append("deps")

                    lang_keys = list(langs.keys()) if langs else (
                        [repo.get("language")] if repo.get("language") else []
                    )
                    multi_lang = len(lang_keys) > 1
                    pushed = _parse_github_datetime(repo.get("pushed_at"))
                    analyses.append(
                        RepoAnalysis(
                            name=name,
                            complexity=_complexity_score(
                                has_tests,
                                has_docker,
                                has_ci,
                                multi_lang,
                                int(repo.get("size") or 0),
                                int(repo.get("stargazers_count") or 0),
                            ),
                            languages=[str(lang) for lang in lang_keys if lang],
                            description=str(repo.get("description") or ""),
                            stars=int(repo.get("stargazers_count") or 0),
                            last_active=_relative_time(pushed),
                            readme_summary=(readme_full[:500] if readme_full else None),
                            production_signals=signals,
                            topics=[str(t) for t in repo.get("topics") or []],
                            has_wiki=bool(repo.get("has_wiki")),
                            default_branch=str(repo.get("default_branch") or "main"),
                        ),
                    )
                except GitHubRateLimitedError:
                    if len(analyses) >= min_partial:
                        logger.warning(
                            "GitHub rate limited after %s detailed repos; using partial profile",
                            len(analyses),
                        )
                        break
                    raise
        except GitHubRateLimitedError:
            if not filtered:
                raise
            logger.warning("GitHub rate limited during repo analysis; using listing metadata")

        analyses = _ensure_min_repo_analyses(analyses, filtered, min_partial)

        now = datetime.now(timezone.utc)
        repos_6mo = sum(
            1
            for repo in filtered
            if (pushed := _parse_github_datetime(repo.get("pushed_at"))) and (now - pushed).days <= 183
        )
        repos_year = sum(
            1
            for repo in filtered
            if (pushed := _parse_github_datetime(repo.get("pushed_at"))) and (now - pushed).days <= 365
        )
        lang_dist = _aggregate_languages(repo_lang_maps)
        most_active = max(lang_dist, key=lang_dist.get) if lang_dist else ""
        total_stars = sum(int(repo.get("stargazers_count") or 0) for repo in filtered)

        created = _parse_github_datetime(user_data.get("created_at"))
        account_age = 0.0
        if created:
            account_age = round((now - created).days / 365.25, 2)

        return GitHubProfile(
            username=user,
            name=user_data.get("name"),
            bio=user_data.get("bio"),
            public_repos=int(user_data.get("public_repos") or 0),
            followers=int(user_data.get("followers") or 0),
            following=int(user_data.get("following") or 0),
            account_age_years=account_age,
            languages_distribution=lang_dist,
            activity_metrics=ActivityMetrics(
                total_repos=len(filtered),
                repos_last_6_months=repos_6mo,
                repos_last_year=repos_year,
                most_active_language=most_active,
                avg_stars=round(total_stars / max(len(filtered), 1), 2),
                total_stars=total_stars,
            ),
            top_repos=analyses,
            inferred_skills=_infer_skills_from_github(
                lang_dist,
                filtered,
                max_skills=cfg.ingestion.github_max_inferred_skills,
            ),
            overall_assessment=_overall_assessment(filtered, repos_6mo),
        )
    finally:
        if owns_client:
            await http_client.aclose()


async def fetch_github_preview(
    username: str,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch a lightweight GitHub preview for onboarding UI."""
    cfg = settings or get_settings()
    user = sanitize_github_username(username, settings=cfg)
    base = cfg.ingestion.github_api_base_url.rstrip("/")
    owns_client = client is None
    http_client = client or httpx.AsyncClient(
        headers=_headers(cfg),
        timeout=15.0,
    )
    try:
        user_data = await _request_json(http_client, f"{base}/users/{user}", cfg)
        if user_data is None:
            raise GitHubUserNotFoundError(f"GitHub user not found: {user}")
        repos_raw = await _request_json(
            http_client,
            f"{base}/users/{user}/repos?sort=updated&per_page=30&type=owner",
            cfg,
        )
        repos = repos_raw if isinstance(repos_raw, list) else []
        lang_counts: dict[str, int] = {}
        for repo in repos:
            if repo.get("fork"):
                continue
            language = repo.get("language")
            if language:
                lang_counts[str(language)] = lang_counts.get(str(language), 0) + 1
        top_languages = sorted(lang_counts, key=lambda k: lang_counts[k], reverse=True)[:3]
        return {
            "username": user,
            "name": user_data.get("name") or user,
            "avatar_url": user_data.get("avatar_url"),
            "public_repos": int(user_data.get("public_repos") or 0),
            "top_languages": top_languages,
        }
    finally:
        if owns_client:
            await http_client.aclose()
