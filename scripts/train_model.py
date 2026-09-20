"""Train and serialize the model artifact used by the API service."""

import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from iris_service.schemas import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACT_PATH = ARTIFACTS_DIR / "iris_pipeline.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


def main() -> None:
    dataset = load_iris(as_frame=True)
    frame = dataset.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length_cm",
            "sepal width (cm)": "sepal_width_cm",
            "petal length (cm)": "petal_length_cm",
            "petal width (cm)": "petal_width_cm",
        }
    )
    target = dataset.target.map(dict(enumerate(dataset.target_names)))

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    pipeline.fit(frame[FEATURE_NAMES], target)

    metadata = {
        "version": "1.0.0",
        "model_type": "LogisticRegression",
        "feature_names": FEATURE_NAMES,
        "target_names": list(dataset.target_names),
        "training_rows": int(len(frame)),
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, ARTIFACT_PATH)
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {ARTIFACT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
