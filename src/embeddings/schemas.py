"""Embedding result models."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class CandidateEmbeddings(BaseModel):
    """Four semantic vectors describing a candidate profile."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skill: np.ndarray = Field(...)
    domain: np.ndarray = Field(...)
    role: np.ndarray = Field(...)
    environment: np.ndarray = Field(...)
