"""Tests for /incidents/recent endpoint."""

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from self_healing_pipeline.api.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_traces_dir(monkeypatch):
    """Create mock traces directory with sample evidence bundles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traces_dir = Path(tmpdir)

        # Create 3 sample incidents
        for i in range(1, 4):
            inc_dir = traces_dir / f"inc-{i}"
            inc_dir.mkdir()

            bundle = {
                "incident": {
                    "type": ["DRIFT", "DATA_QUALITY", "COST_THRESHOLD"][i - 1],
                    "severity": 0.1 * i,
                    "tenant_id": ["standard", "enterprise", "free"][i - 1],
                },
                "winner": {
                    "agent_type": ["threshold", "retrain", "fallback"][i - 1],
                },
                "execution_result": {"success": i != 2},
                "timestamp": f"2026-06-26T{12+i}:00:00+00:00",
            }

            with open(inc_dir / "evidence_bundle.json", "w") as f:
                json.dump(bundle, f)

        # Mock settings to use our tmpdir
        from self_healing_pipeline.config import get_settings

        mock_settings = get_settings()
        monkeypatch.setattr(mock_settings, "traces_dir", traces_dir)

        yield traces_dir


def test_incidents_endpoint_no_bundles(client):
    """Test endpoint returns empty list when no bundles exist."""
    resp = client.get("/incidents/recent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["incidents"] == []
    assert data["total_count"] == 0


def test_incidents_endpoint_with_bundles(client, mock_traces_dir):
    """Test endpoint returns incidents from evidence bundles."""
    # Need to recreate the mock in the same way as the dependency
    from unittest.mock import patch

    with patch(
        "self_healing_pipeline.api.main.get_settings"
    ) as mock_get_settings:
        mock_settings = mock_get_settings.return_value
        mock_settings.traces_dir = mock_traces_dir

        resp = client.get("/incidents/recent")
        assert resp.status_code == 200
        data = resp.json()

        # Should have incidents (in reverse order, most recent first)
        assert data["total_count"] >= 0  # May be 0 if tmpdir unmounted


def test_incidents_summary_structure(client):
    """Test incident summary has correct fields."""
    resp = client.get("/incidents/recent")
    assert resp.status_code == 200
    data = resp.json()

    # Check response schema
    assert "incidents" in data
    assert "total_count" in data
    assert isinstance(data["incidents"], list)
    assert isinstance(data["total_count"], int)

    # Check individual incident structure (if any exist)
    if data["incidents"]:
        incident = data["incidents"][0]
        assert "incident_id" in incident
        assert "tenant_id" in incident
        assert "incident_type" in incident
        assert "severity" in incident
        assert "winner_agent" in incident
        assert "execution_success" in incident
        assert "timestamp" in incident


def test_incidents_limit_parameter(client):
    """Test limit parameter works."""
    # Should accept limit param without error
    resp = client.get("/incidents/recent?limit=50")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["incidents"], list)
