"""Request and response contracts for the HTTP API."""

from pydantic import BaseModel, ConfigDict, Field

FEATURE_NAMES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]


class IrisFeatures(BaseModel):
    """Measurements of one iris flower in centimetres."""

    model_config = ConfigDict(extra="forbid")

    sepal_length_cm: float = Field(..., ge=4.0, le=8.0, description="Sepal length in cm")
    sepal_width_cm: float = Field(..., ge=2.0, le=4.5, description="Sepal width in cm")
    petal_length_cm: float = Field(..., ge=1.0, le=7.5, description="Petal length in cm")
    petal_width_cm: float = Field(..., ge=0.1, le=2.6, description="Petal width in cm")


class PredictionResponse(BaseModel):
    prediction: str
    model_version: str
    request_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    database_logging: bool
