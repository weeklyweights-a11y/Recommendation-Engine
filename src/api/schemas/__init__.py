"""API request/response schemas."""

from src.api.schemas.candidate import (
    CandidateCreate,
    CandidatePreferences,
    CandidateProfile,
    CandidateResponse,
    ESCOLinkedSkill,
    GitHubSummary,
    MergedPreferences,
    PreferenceField,
    ProfileEducation,
    ProfileExperience,
    ProfileSkill,
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
    "ESCOLinkedSkill",
    "GitHubSummary",
    "MergedPreferences",
    "PreferenceField",
    "ProfileEducation",
    "ProfileExperience",
    "ProfileSkill",
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
