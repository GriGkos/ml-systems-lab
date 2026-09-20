from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from iris_service.model import load_model_bundle


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "1.0.0",
        "database_logging": False,
    }


def test_ready_returns_ok(client: TestClient) -> None:
    assert client.get("/ready").status_code == 200


def test_ready_returns_503_when_configured_database_is_unavailable(client: TestClient) -> None:
    class UnavailableLogStore:
        def ping(self) -> None:
            raise SQLAlchemyError("database is unavailable")

    previous_store = client.app.state.log_store
    client.app.state.log_store = UnavailableLogStore()
    try:
        assert client.get("/ready").status_code == 503
    finally:
        client.app.state.log_store = previous_store


def test_docs_are_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200


def test_predict_returns_expected_contract(
    client: TestClient, valid_payload: dict[str, float]
) -> None:
    response = client.post("/v1/predict", json=valid_payload, headers={"X-Request-ID": "smoke-123"})

    assert response.status_code == 200
    assert response.json()["prediction"] == "setosa"
    assert response.json()["model_version"] == "1.0.0"
    assert response.json()["request_id"] == "smoke-123"
    assert response.json()["latency_ms"] >= 0


def test_rejects_unexpected_feature(client: TestClient, valid_payload: dict[str, float]) -> None:
    response = client.post("/v1/predict", json={**valid_payload, "colour": "red"})

    assert response.status_code == 422


def test_rejects_out_of_range_value(client: TestClient, valid_payload: dict[str, float]) -> None:
    response = client.post("/v1/predict", json={**valid_payload, "petal_width_cm": 20})

    assert response.status_code == 422


def test_same_request_is_deterministic(client: TestClient, valid_payload: dict[str, float]) -> None:
    first = client.post("/v1/predict", json=valid_payload).json()
    second = client.post("/v1/predict", json=valid_payload).json()

    assert first["prediction"] == second["prediction"]
    assert first["model_version"] == second["model_version"]


def test_model_passport_contains_reproducibility_data() -> None:
    metadata = load_model_bundle().metadata

    assert metadata["training_rows"] == 150
    assert len(metadata["training_data_sha256"]) == 64
    assert metadata["metrics"]["train_accuracy"] >= 0.9
    assert set(metadata["example_input"]) == set(metadata["feature_names"])
