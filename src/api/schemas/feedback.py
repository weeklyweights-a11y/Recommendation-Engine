"""Pydantic schemas for user feedback."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

FeedbackAction = Literal["saved", "dismissed", "applied"]


class FeedbackCreate(BaseModel):
    """Create feedback for a job."""

    candidate_id: UUID
    job_id: UUID
    action: FeedbackAction


class FeedbackResponse(BaseModel):
    """Feedback response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    candidate_id: UUID
    job_id: UUID
    action: str
    created_at: datetime
