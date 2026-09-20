"""Request and response contracts for the credit-scoring HTTP API."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CreditApplication(BaseModel):
    """Raw Home Credit application fields, before deterministic feature engineering."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(
        ..., description="Columns of one Home Credit application, excluding TARGET and SK_ID_CURR."
    )

    @field_validator("features")
    @classmethod
    def features_must_not_be_empty(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("features must not be empty")
        return value


class PredictionResponse(BaseModel):
    prediction: str
    default_probability: float = Field(..., ge=0, le=1)
    model_version: str
    request_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    database_logging: bool
