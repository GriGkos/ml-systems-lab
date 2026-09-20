"""Loading and using the serialized sklearn model bundle."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from credit_service.credit_features import enrich_application
from credit_service.settings import get_settings


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
    def raw_feature_names(self) -> list[str]:
        return list(self.metadata["raw_feature_names"])

    def predict(self, values: dict[str, Any]) -> tuple[str, float]:
        """Return the default decision and its probability for one raw application."""
        unknown_columns = set(values) - set(self.raw_feature_names)
        if unknown_columns:
            raise ValueError(f"Unknown application fields: {', '.join(sorted(unknown_columns))}")

        frame = pd.DataFrame([values]).reindex(columns=self.raw_feature_names)
        for column in self.metadata["numeric_raw_feature_names"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        model_frame = enrich_application(frame).reindex(
            columns=self.metadata["model_feature_names"]
        )
        probability = float(self.pipeline.predict_proba(model_frame)[0, 1])
        threshold = float(self.metadata["decision_threshold"])
        return ("default" if probability >= threshold else "no_default", probability)


def load_model_bundle(path: Path | None = None) -> ModelBundle:
    """Load a Pipeline and its model passport from the joblib artifact."""
    path = path or default_artifact_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run `uv run python scripts/train_credit_model.py`."
        )

    artifact = joblib.load(path)
    return ModelBundle(pipeline=artifact["pipeline"], metadata=artifact["metadata"])
