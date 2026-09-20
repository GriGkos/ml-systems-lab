"""Loading and using the serialized sklearn model bundle."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from iris_service.settings import get_settings


def default_artifact_path() -> Path:
    """Resolve the artifact path from the application's runtime settings."""
    return get_settings().model_path


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Any
    metadata: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.metadata["version"])

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata["feature_names"])

    def predict(self, values: dict[str, float]) -> str:
        frame = pd.DataFrame(
            [[values[name] for name in self.feature_names]], columns=self.feature_names
        )
        return str(self.pipeline.predict(frame)[0])


def load_model_bundle(path: Path | None = None) -> ModelBundle:
    """Load a Pipeline and its model passport from the joblib artifact."""
    path = path or default_artifact_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run `uv run python scripts/train_model.py`."
        )

    artifact = joblib.load(path)
    return ModelBundle(pipeline=artifact["pipeline"], metadata=artifact["metadata"])
