"""HTTP client wrappers for the FastAPI backend."""

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import UUID

import httpx

from frontend.ui_settings import get_frontend_settings


class ApiError(Exception):
    """API request failed with a user-facing message."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _base_url() -> str:
    return get_frontend_settings().api_base_url.rstrip("/")


def _friendly_error(status_code: int, detail: str) -> str:
    if status_code in (502, 503, 504) or status_code is None:
        return "Service is starting up, please try again in a moment."
    if status_code == 413:
        return "File is too large. Try a smaller PDF or DOCX under the size limit."
    if status_code == 422:
        return detail or "Invalid input. Check your file and preferences."
    if status_code == 404:
        return detail or "Not found."
    if status_code == 409:
        return detail or "Already recorded."
    return detail or "Something went wrong. Please try again."


def _raise_for_response(response: httpx.Response) -> None:
    if response.is_success:
        return
    detail = "Request failed"
    try:
        body = response.json()
        if isinstance(body, dict) and "detail" in body:
            raw = body["detail"]
            detail = raw if isinstance(raw, str) else str(raw)
    except json.JSONDecodeError:
        detail = response.text[:200] or detail
    raise ApiError(_friendly_error(response.status_code, detail), response.status_code)


def create_candidate(
    resume_bytes: bytes,
    filename: str,
    github_username: Optional[str],
    preferences: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """POST /api/v1/candidates with resume upload."""
    cfg = get_frontend_settings()
    files = {"resume": (filename, resume_bytes)}
    data: dict[str, str] = {}
    if github_username:
        data["github_username"] = github_username.strip().lstrip("@")
    if preferences is not None:
        data["preferences"] = json.dumps(preferences)
    with httpx.Client(timeout=cfg.candidate_create_timeout_seconds) as client:
        response = client.post(f"{_base_url()}/api/v1/candidates", files=files, data=data)
    _raise_for_response(response)
    return response.json()


def get_candidate(candidate_id: UUID | str) -> dict[str, Any]:
    """GET /api/v1/candidates/{id}."""
    cfg = get_frontend_settings()
    with httpx.Client(timeout=cfg.api_timeout_seconds) as client:
        response = client.get(f"{_base_url()}/api/v1/candidates/{candidate_id}")
    _raise_for_response(response)
    return response.json()


def patch_preferences(candidate_id: UUID | str, preferences: dict[str, Any]) -> dict[str, Any]:
    """PATCH /api/v1/candidates/{id}/preferences."""
    cfg = get_frontend_settings()
    with httpx.Client(timeout=cfg.api_timeout_seconds) as client:
        response = client.patch(
            f"{_base_url()}/api/v1/candidates/{candidate_id}/preferences",
            json=preferences,
        )
    _raise_for_response(response)
    return response.json()


def get_recommendations(
    candidate_id: UUID | str,
    *,
    page: int = 1,
    per_page: int = 20,
    refresh: bool = False,
) -> dict[str, Any]:
    """GET /api/v1/recommendations/{id}."""
    cfg = get_frontend_settings()
    params: dict[str, Any] = {"page": page, "per_page": per_page}
    if refresh:
        params["refresh"] = "true"
    timeout = cfg.candidate_create_timeout_seconds if refresh else cfg.api_timeout_seconds
    with httpx.Client(timeout=timeout) as client:
        response = client.get(
            f"{_base_url()}/api/v1/recommendations/{candidate_id}",
            params=params,
        )
    _raise_for_response(response)
    return response.json()


def post_feedback(candidate_id: UUID | str, job_id: UUID | str, action: str) -> dict[str, Any]:
    """POST /api/v1/feedback."""
    cfg = get_frontend_settings()
    payload = {
        "candidate_id": str(candidate_id),
        "job_id": str(job_id),
        "action": action,
    }
    with httpx.Client(timeout=cfg.api_timeout_seconds) as client:
        response = client.post(f"{_base_url()}/api/v1/feedback", json=payload)
    if response.status_code == 409:
        raise ApiError("Already recorded.", 409)
    _raise_for_response(response)
    return response.json()


def list_feedback(candidate_id: UUID | str) -> list[dict[str, Any]]:
    """GET /api/v1/feedback/{candidate_id}."""
    cfg = get_frontend_settings()
    with httpx.Client(timeout=cfg.api_timeout_seconds) as client:
        response = client.get(f"{_base_url()}/api/v1/feedback/{candidate_id}")
    _raise_for_response(response)
    return response.json()


def github_preview(username: str) -> dict[str, Any]:
    """GET /api/v1/github/preview."""
    cfg = get_frontend_settings()
    with httpx.Client(timeout=cfg.api_timeout_seconds) as client:
        response = client.get(
            f"{_base_url()}/api/v1/github/preview",
            params={"username": username.strip().lstrip("@")},
        )
    _raise_for_response(response)
    return response.json()
