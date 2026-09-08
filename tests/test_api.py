import pytest
from fastapi.testclient import TestClient

from app.llm_client import LLMClient, LLMError, MockLLMClient
from app.main import app, get_llm_client


@pytest.fixture()
def client() -> TestClient:
    app.dependency_overrides[get_llm_client] = lambda: MockLLMClient()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_metrics_exposes_prometheus_format(client: TestClient) -> None:
    client.get("/healthz")  # generate at least one request metric
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_chat_returns_reply(client: TestClient) -> None:
    response = client.post("/api/chat", json={"prompt": "Hello"})
    assert response.status_code == 200
    body = response.json()
    assert "Hello" in body["reply"]
    assert body["model"] == "mock-model"
    assert body["latency_ms"] >= 0


def test_chat_rejects_empty_prompt(client: TestClient) -> None:
    response = client.post("/api/chat", json={"prompt": ""})
    assert response.status_code == 422


def test_chat_maps_llm_error_to_502(client: TestClient) -> None:
    class FailingClient(LLMClient):
        async def chat(self, prompt: str, model: str | None = None) -> dict:
            raise LLMError("backend down")

    app.dependency_overrides[get_llm_client] = lambda: FailingClient()
    response = client.post("/api/chat", json={"prompt": "Hello"})
    assert response.status_code == 502
    assert "backend down" in response.json()["detail"]
