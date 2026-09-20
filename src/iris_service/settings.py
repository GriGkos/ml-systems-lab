"""Configuration read from environment variables (and an optional local .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings; environment variables take precedence over defaults."""

    model_path: Path = Path("artifacts/iris_pipeline.joblib")
    database_url: str | None = None
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return one immutable configuration object per process."""
    return Settings()
