"""API request/response schemas."""

from src.api.schemas.candidate import (
    CandidateCreate,
    CandidatePreferences,
    CandidateProfile,
    CandidateResponse,
)
from src.api.schemas.feedback import FeedbackCreate, FeedbackResponse
from src.api.schemas.job import JobBase, JobCreate, JobListResponse, JobResponse
from src.api.schemas.recommendation import (
    RecommendationListResponse,
    RecommendationResponse,
    ScoredJob,
)

__all__ = [
    "CandidateCreate",
    "CandidatePreferences",
    "CandidateProfile",
    "CandidateResponse",
    "FeedbackCreate",
    "FeedbackResponse",
    "JobBase",
    "JobCreate",
    "JobListResponse",
    "JobResponse",
    "RecommendationListResponse",
    "RecommendationResponse",
    "ScoredJob",
]
