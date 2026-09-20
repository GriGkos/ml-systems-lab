"""PostgreSQL persistence for successful prediction requests."""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    insert,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

metadata = MetaData()

prediction_logs = Table(
    "prediction_logs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("request_id", String(64), nullable=False, index=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("model_version", String(32), nullable=False),
    Column("features", JSONB, nullable=False),
    Column("prediction", String(64), nullable=False),
    Column("latency_ms", Float, nullable=False),
    Column("status_code", Integer, nullable=False),
)

Index(
    "ix_prediction_logs_created_at_model_version",
    prediction_logs.c.created_at,
    prediction_logs.c.model_version,
)


class PredictionLogStore:
    """A tiny repository that is created only when DATABASE_URL is configured."""

    def __init__(self, database_url: str) -> None:
        self.engine: Engine = create_engine(database_url, pool_pre_ping=True)

    def initialize(self) -> None:
        metadata.create_all(self.engine)

    def log_prediction(
        self,
        *,
        request_id: str,
        model_version: str,
        features: dict[str, float],
        prediction: str,
        latency_ms: float,
        status_code: int,
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                insert(prediction_logs).values(
                    request_id=request_id,
                    model_version=model_version,
                    features=features,
                    prediction=prediction,
                    latency_ms=latency_ms,
                    status_code=status_code,
                    created_at=datetime.now().astimezone(),
                )
            )

    def ping(self) -> None:
        """Raise an SQLAlchemy error when PostgreSQL cannot answer a simple query."""
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def dispose(self) -> None:
        self.engine.dispose()
