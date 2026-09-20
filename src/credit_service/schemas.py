"""Request and response contracts for the credit-scoring HTTP API."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, create_model

ARTIFACT_METADATA_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "credit_scoring_metadata.json"
)


def _field_definition(name: str, numeric_columns: set[str]) -> tuple[object, Field]:
    """Describe one raw application field for OpenAPI and request validation."""
    if name not in numeric_columns:
        return str | None, Field(default=None, description="Categorical application field.")
    if name.startswith("EXT_SOURCE_"):
        return float | None, Field(default=None, ge=0, le=1, description="External score.")
    if name.startswith("AMT_") or name.startswith("CNT_"):
        return float | None, Field(default=None, ge=0, description="Non-negative amount or count.")
    if name in {"REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"}:
        return float | None, Field(default=None, ge=1, le=3, description="Regional rating.")
    if name.startswith("FLAG_"):
        return float | None, Field(default=None, ge=0, le=1, description="Binary application flag.")
    return float | None, Field(default=None, description="Numeric application field.")


def _credit_application_schema() -> type[BaseModel]:
    """Build an explicit schema from the same artifact passport used by the service."""
    metadata = json.loads(ARTIFACT_METADATA_PATH.read_text(encoding="utf-8"))
    numeric_columns = set(metadata["numeric_raw_feature_names"])
    fields = {
        name: _field_definition(name, numeric_columns) for name in metadata["raw_feature_names"]
    }
    return create_model("CreditApplication", __config__=ConfigDict(extra="forbid"), **fields)


CreditApplication = _credit_application_schema()


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
