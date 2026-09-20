"""Train the fixed application-only credit-scoring pipeline from the source project."""

import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_service.credit_features import enrich_application

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = Path(
    r"C:\Users\kosti\OneDrive\Документы\projects\projects_ml\projects\04_home_credit_scoring\data\raw\application_train.csv"
)
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACT_PATH = ARTIFACTS_DIR / "credit_scoring_pipeline.joblib"
METADATA_PATH = ARTIFACTS_DIR / "credit_scoring_metadata.json"


def source_path() -> Path:
    """Return the explicit training CSV path or the known Home Credit source file."""
    return Path(os.getenv("CREDIT_TRAIN_DATA_PATH", DEFAULT_DATA_PATH))


def build_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    """Use the exact fixed Logistic Regression configuration from the source project."""
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    (
                                        "imputer",
                                        SimpleImputer(strategy="median", add_indicator=True),
                                    ),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            numeric_columns,
                        ),
                        (
                            "categorical",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            categorical_columns,
                        ),
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    max_iter=300,
                    tol=1e-3,
                    class_weight="balanced",
                    solver="saga",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    path = source_path()
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    raw = pd.read_csv(path)
    target = raw.pop("TARGET").astype("int8")
    raw_features = raw.drop(columns="SK_ID_CURR")
    model_frame = enrich_application(raw_features)
    numeric_columns = model_frame.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in model_frame if column not in numeric_columns]
    pipeline = build_pipeline(numeric_columns, categorical_columns)
    pipeline.fit(model_frame, target)

    data_hash = sha256(
        pd.util.hash_pandas_object(raw_features, index=True).values.tobytes()
    ).hexdigest()
    example_input = json.loads(raw_features.head(1).to_json(orient="records"))[0]
    metadata = {
        "version": "2.0.0",
        "model_type": "LogisticRegression application-only",
        "source_project": "04_home_credit_scoring",
        "source_dataset": "Home Credit Default Risk / application_train.csv",
        "raw_feature_names": raw_features.columns.tolist(),
        "numeric_raw_feature_names": raw_features.select_dtypes(include="number").columns.tolist(),
        "model_feature_names": model_frame.columns.tolist(),
        "training_rows": int(len(raw_features)),
        "positive_class_rate": round(float(target.mean()), 6),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_data_sha256": data_hash,
        "reference_metrics": {"oof_roc_auc": 0.75141, "oof_pr_auc": 0.23143},
        "decision_threshold": 0.5,
        "dependencies": {
            "joblib": joblib.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "example_input": example_input,
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, ARTIFACT_PATH)
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {ARTIFACT_PATH.relative_to(ROOT)} from {len(raw_features):,} applications")


if __name__ == "__main__":
    main()
