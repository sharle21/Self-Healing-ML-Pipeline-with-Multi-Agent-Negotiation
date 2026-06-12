from fastapi.testclient import TestClient

from self_healing_pipeline.api import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "self-healing-pipeline"
    assert isinstance(body["version"], str)
    assert len(body["version"]) > 0


def test_health_response_shape() -> None:
    resp = client.get("/health")
    assert set(resp.json().keys()) == {"status", "service", "version"}
