"""FastAPI application for the Iris classifier."""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from iris_service.database import PredictionLogStore
from iris_service.model import ModelBundle, load_model_bundle
from iris_service.schemas import HealthResponse, IrisFeatures, PredictionResponse
from iris_service.settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once and initialize PostgreSQL logging when configured."""
    app.state.model = load_model_bundle()
    settings = get_settings()
    app.state.log_store = None

    if settings.database_url:
        log_store = PredictionLogStore(settings.database_url)
        app.state.log_store = log_store
        try:
            log_store.initialize()
        except SQLAlchemyError:
            # The API may still serve a prediction; /ready will remain unavailable.
            logger.exception("PostgreSQL is unavailable during service startup")

    yield

    if app.state.log_store is not None:
        app.state.log_store.dispose()


app = FastAPI(
    title="Iris classification service",
    version="1.0.0",
    description="Predicts an Iris species from four flower measurements.",
    lifespan=lifespan,
)


def get_model() -> ModelBundle:
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model = get_model()
    return HealthResponse(
        status="ok",
        model_version=model.version,
        database_logging=app.state.log_store is not None,
    )


@app.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    log_store: PredictionLogStore | None = app.state.log_store
    if log_store is not None:
        try:
            log_store.ping()
        except SQLAlchemyError as error:
            logger.warning("Readiness check failed because PostgreSQL is unavailable")
            raise HTTPException(status_code=503, detail="Database is unavailable") from error
    return health()


def log_prediction_safely(log_store: PredictionLogStore, **values: object) -> None:
    """Persist a successful prediction without delaying or breaking the API response."""
    try:
        log_store.log_prediction(**values)  # type: ignore[arg-type]
    except SQLAlchemyError:
        logger.exception("Could not write prediction log to PostgreSQL")


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(
    features: IrisFeatures,
    background_tasks: BackgroundTasks,
    x_request_id: str | None = Header(default=None),
) -> PredictionResponse:
    model = get_model()
    request_id = x_request_id or str(uuid.uuid4())
    values = features.model_dump()

    started_at = time.perf_counter()
    prediction = model.predict(values)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 3)

    log_store: PredictionLogStore | None = app.state.log_store
    if log_store is not None:
        background_tasks.add_task(
            log_prediction_safely,
            log_store,
            request_id=request_id,
            model_version=model.version,
            features=values,
            prediction=prediction,
            latency_ms=latency_ms,
            status_code=200,
        )

    return PredictionResponse(
        prediction=prediction,
        model_version=model.version,
        request_id=request_id,
        latency_ms=latency_ms,
    )
