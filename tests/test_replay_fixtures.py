"""Tests for replay fixture harness (deterministic LLM responses)."""

import json
import tempfile
from pathlib import Path

import pytest

from self_healing_pipeline.agents.llm import LLMClient


class TestLLMClientFixtures:
    """LLM client with replay fixtures tests."""

    def test_init_creates_fixtures_dir(self):
        """Test that init creates fixtures directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_dir = Path(tmpdir) / "fixtures"
            client = LLMClient(fixtures_dir=fixtures_dir)
            assert fixtures_dir.exists()

    def test_replay_mode_loads_fixture(self):
        """Test loading response from fixture in replay mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_dir = Path(tmpdir)
            # Create a fixture file
            fixture_path = fixtures_dir / "threshold_DRIFT.json"
            fixture_data = {
                "model": "claude-sonnet-4-6",
                "agent_type": "threshold",
                "incident_type": "DRIFT",
                "response": '{"confidence": 0.85, "estimated_business_savings": 1500.0}',
            }
            with open(fixture_path, "w") as f:
                json.dump(fixture_data, f)

            client = LLMClient(fixtures_dir=fixtures_dir, replay_mode=True)
            response = client.call("dummy prompt", "threshold", "DRIFT")
            assert "0.85" in response
            assert "1500" in response

    def test_replay_mode_fallback_when_fixture_missing(self):
        """Test fallback response when fixture not found in replay mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_dir = Path(tmpdir)
            client = LLMClient(fixtures_dir=fixtures_dir, replay_mode=True)
            response = client.call("dummy prompt", "retrain", "DATA_QUALITY")
            # Should get fallback response
            data = json.loads(response)
            assert "confidence" in data
            assert "estimated_business_savings" in data

    def test_record_mode_saves_fixture(self):
        """Test saving response to fixture in record mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_dir = Path(tmpdir)
            client = LLMClient(fixtures_dir=fixtures_dir, replay_mode=False)
            response = client.call("test prompt", "threshold", "DRIFT")
            # Should save fixture
            fixture_path = fixtures_dir / "threshold_DRIFT.json"
            assert fixture_path.exists()
            with open(fixture_path) as f:
                data = json.load(f)
            assert data["agent_type"] == "threshold"
            assert data["incident_type"] == "DRIFT"

    def test_fixture_key_override(self):
        """Test using custom fixture key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_dir = Path(tmpdir)
            # Create custom fixture
            custom_path = fixtures_dir / "custom_key.json"
            with open(custom_path, "w") as f:
                json.dump({"response": '{"custom": true}'}, f)

            client = LLMClient(fixtures_dir=fixtures_dir, replay_mode=True)
            response = client.call(
                "prompt",
                "agent_a",
                "incident_type",
                fixture_key="custom_key",
            )
            assert "custom" in response

    def test_fallback_response_structure(self):
        """Test fallback response has required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = LLMClient(fixtures_dir=Path(tmpdir), replay_mode=True)
            response = client.call("prompt", "any_agent", "any_type")
            data = json.loads(response)
            assert "confidence" in data
            assert "estimated_business_savings" in data
            assert "estimated_risk" in data
            assert "estimated_compute_cost" in data
            assert "estimated_time" in data
            assert "rationale" in data
            # Validate value ranges
            assert 0 <= data["confidence"] <= 1
            assert 0 <= data["estimated_risk"] <= 1
            assert data["estimated_business_savings"] >= 0
            assert data["estimated_compute_cost"] >= 0
            assert data["estimated_time"] >= 0

    def test_fixture_data_preserved_on_load(self):
        """Test that all fixture data is preserved on load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_dir = Path(tmpdir)
            fixture_data = {
                "model": "claude-sonnet-4-6",
                "agent_type": "rollback",
                "incident_type": "COST_THRESHOLD",
                "prompt_hash": 12345,
                "response": '{"confidence": 0.9, "savings": 3000}',
            }
            fixture_path = fixtures_dir / "rollback_COST_THRESHOLD.json"
            with open(fixture_path, "w") as f:
                json.dump(fixture_data, f)

            client = LLMClient(fixtures_dir=fixtures_dir, replay_mode=True)
            response = client.call("prompt", "rollback", "COST_THRESHOLD")
            # Should get the exact response we stored
            assert response == '{"confidence": 0.9, "savings": 3000}'

    def test_model_name_stored_in_fixture(self):
        """Test that model name is recorded in fixture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_dir = Path(tmpdir)
            client = LLMClient(
                model_name="claude-opus-4-8",
                fixtures_dir=fixtures_dir,
                replay_mode=False,
            )
            client.call("prompt", "fallback", "DATA_QUALITY")
            fixture_path = fixtures_dir / "fallback_DATA_QUALITY.json"
            with open(fixture_path) as f:
                data = json.load(f)
            assert data["model"] == "claude-opus-4-8"
