import os

import pytest
from fastapi.testclient import TestClient

os.environ.pop("DATABASE_URL", None)

from iris_service.main import app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def valid_payload() -> dict[str, float]:
    return {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
    }
