from fastapi.testclient import TestClient


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
