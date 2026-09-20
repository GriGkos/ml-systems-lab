import os

import pytest
from fastapi.testclient import TestClient

os.environ.pop("DATABASE_URL", None)

from credit_service.main import app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def valid_payload() -> dict[str, object]:
    from credit_service.model import load_model_bundle

    return load_model_bundle().metadata["example_input"]
