import pytest
from fastapi.testclient import TestClient

from self_healing_pipeline.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_metrics_endpoint_returns_200(client):
    """Test metrics endpoint returns 200."""
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_endpoint_returns_prometheus_content_type(client):
    """Test metrics endpoint returns Prometheus text format."""
    response = client.get("/metrics")
    assert response.headers["content-type"] == "text/plain; charset=utf-8"


def test_metrics_endpoint_contains_prometheus_format(client):
    """Test metrics endpoint returns valid Prometheus format."""
    response = client.get("/metrics")
    content = response.text
    # Check for Prometheus format markers
    assert "# HELP" in content
    assert "# TYPE" in content
    # Should have some metrics (at least Python/process metrics)
    assert len(content) > 0
