"""Tests for weight version control."""

import json
import tempfile
from pathlib import Path

import pytest

from self_healing_pipeline.meta_harness.version_control import (
    WeightVersion,
    WeightVersionControl,
)


@pytest.fixture
def versions_dir():
    """Temporary versions directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestWeightVersion:
    """WeightVersion tests."""

    def test_to_dict(self):
        """Test conversion to dict."""
        version = WeightVersion(
            version=1,
            timestamp="2026-06-26T14:00:00+00:00",
            weights={"confidence": 0.25},
            reason="Initial weights",
        )
        d = version.to_dict()
        assert d["version"] == 1
        assert d["weights"]["confidence"] == 0.25
        assert "timestamp" in d


class TestWeightVersionControl:
    """WeightVersionControl tests."""

    def test_init_creates_dir(self, versions_dir):
        """Test init creates versions directory."""
        vc = WeightVersionControl(versions_dir)
        assert versions_dir.exists()

    def test_save_version(self, versions_dir):
        """Test saving a version."""
        vc = WeightVersionControl(versions_dir)
        weights = {"confidence": 0.25, "business_value": 0.30}
        version = vc.save_version(weights, "High performers boost confidence")

        assert version.version == 1
        assert version.weights == weights
        assert version.reason == "High performers boost confidence"

        # Check file exists
        version_file = versions_dir / "weights_v1.json"
        assert version_file.exists()

    def test_save_multiple_versions(self, versions_dir):
        """Test saving multiple versions increments correctly."""
        vc = WeightVersionControl(versions_dir)

        v1 = vc.save_version(
            {"confidence": 0.20}, "Version 1"
        )
        v2 = vc.save_version(
            {"confidence": 0.25}, "Version 2"
        )
        v3 = vc.save_version(
            {"confidence": 0.30}, "Version 3"
        )

        assert v1.version == 1
        assert v2.version == 2
        assert v3.version == 3

    def test_load_latest(self, versions_dir):
        """Test loading latest version."""
        vc = WeightVersionControl(versions_dir)
        vc.save_version({"confidence": 0.20}, "V1")
        vc.save_version({"confidence": 0.25}, "V2")

        latest = vc.load_latest()
        assert latest is not None
        assert latest.version == 2
        assert latest.weights["confidence"] == 0.25

    def test_load_latest_empty(self, versions_dir):
        """Test load_latest with no versions."""
        vc = WeightVersionControl(versions_dir)
        latest = vc.load_latest()
        assert latest is None

    def test_load_specific_version(self, versions_dir):
        """Test loading specific version."""
        vc = WeightVersionControl(versions_dir)
        vc.save_version({"confidence": 0.20}, "V1")
        vc.save_version({"confidence": 0.25}, "V2")

        v1 = vc.load_version(1)
        assert v1 is not None
        assert v1.version == 1
        assert v1.weights["confidence"] == 0.20

    def test_load_nonexistent_version(self, versions_dir):
        """Test loading nonexistent version returns None."""
        vc = WeightVersionControl(versions_dir)
        v = vc.load_version(999)
        assert v is None

    def test_save_with_validation_stats(self, versions_dir):
        """Test saving version with validation stats."""
        vc = WeightVersionControl(versions_dir)
        stats = {"high_performers": 2, "low_performers": 1}
        version = vc.save_version(
            {"confidence": 0.25},
            "Tuned weights",
            validation_stats=stats,
        )

        assert version.validation_stats == stats

        # Reload and verify
        loaded = vc.load_version(version.version)
        assert loaded.validation_stats == stats

    def test_list_versions(self, versions_dir):
        """Test listing all versions in order."""
        vc = WeightVersionControl(versions_dir)
        vc.save_version({"confidence": 0.20}, "V1")
        vc.save_version({"confidence": 0.22}, "V2")
        vc.save_version({"confidence": 0.25}, "V3")

        versions = vc.list_versions()
        assert len(versions) == 3
        assert versions[0].version == 1
        assert versions[1].version == 2
        assert versions[2].version == 3
        # Verify order is oldest first
        assert versions[0].weights["confidence"] == 0.20
        assert versions[2].weights["confidence"] == 0.25

    def test_version_persistence(self, versions_dir):
        """Test that versions persist across instances."""
        vc1 = WeightVersionControl(versions_dir)
        vc1.save_version({"confidence": 0.25}, "V1")

        # New instance should see the same version
        vc2 = WeightVersionControl(versions_dir)
        latest = vc2.load_latest()
        assert latest is not None
        assert latest.version == 1
