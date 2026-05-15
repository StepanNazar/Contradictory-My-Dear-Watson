from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    premise: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    language: str | None = None


class PredictionResponse(BaseModel):
    label: int
    label_name: str
    probabilities: dict[str, float] | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: str | None = None
