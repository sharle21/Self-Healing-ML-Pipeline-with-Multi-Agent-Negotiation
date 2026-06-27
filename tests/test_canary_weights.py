"""Tests for canary weight version management."""

import tempfile
from pathlib import Path

import pytest

from self_healing_pipeline.meta_harness.canary import CanaryWeightManager
from self_healing_pipeline.meta_harness.version_control import WeightVersionControl


@pytest.fixture
def version_control():
    """Version control with test weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vc = WeightVersionControl(Path(tmpdir))
        # Create v1 (stable)
        vc.save_version(
            {"confidence": 0.20, "business_value": 0.30, "risk_inverse": 0.20, "cost_efficiency": 0.10, "time_inverse": 0.05, "historical_success": 0.15},
            "Stable v1"
        )
        # Create v2 (canary)
        vc.save_version(
            {"confidence": 0.25, "business_value": 0.28, "risk_inverse": 0.20, "cost_efficiency": 0.10, "time_inverse": 0.05, "historical_success": 0.12},
            "Canary v2"
        )
        yield vc


class TestCanaryWeightManager:
    """Canary weight management tests."""

    def test_select_weight_no_canary(self, version_control):
        """Test weight selection with no active canary."""
        manager = CanaryWeightManager(version_control)
        weights = manager.select_weight_version("inc-123")
        assert weights.confidence == 0.25  # Latest (v2)

    def test_select_weight_with_canary(self, version_control):
        """Test deterministic routing to canary/stable."""
        from self_healing_pipeline.meta_harness.canary import CanaryConfig

        manager = CanaryWeightManager(version_control)
        # Set v1 as stable, v2 as canary (manually to avoid auto-detection)
        manager.canary_config = CanaryConfig(
            stable_version=1, canary_version=2, canary_percentage=50.0
        )

        # Some incidents should go to canary (v2), some to stable (v1)
        canary_hits = 0
        stable_hits = 0

        for i in range(100):
            weights = manager.select_weight_version(f"inc-{i}")
            if weights.confidence == 0.25:  # v2
                canary_hits += 1
            elif weights.confidence == 0.20:  # v1
                stable_hits += 1

        # At 50%, expect roughly half of each (with some variance)
        assert 30 < canary_hits < 70
        assert 30 < stable_hits < 70

    def test_deterministic_routing(self, version_control):
        """Test that same incident always routes same way."""
        manager = CanaryWeightManager(version_control)
        manager.start_canary(canary_version=2, percentage=50.0)

        # Same incident ID should always route the same way
        w1 = manager.select_weight_version("inc-123")
        w2 = manager.select_weight_version("inc-123")
        w3 = manager.select_weight_version("inc-123")

        assert w1.confidence == w2.confidence == w3.confidence

    def test_record_outcome(self, version_control):
        """Test outcome recording for canary metrics."""
        manager = CanaryWeightManager(version_control)
        manager.start_canary(canary_version=2, percentage=50.0)

        # Record some outcomes
        manager.record_outcome("inc-1", success=True)
        manager.record_outcome("inc-2", success=False)
        manager.record_outcome("inc-3", success=True)

        # Metrics should be updated
        total_attempts = (
            manager.canary_metrics["canary_attempts"]
            + manager.canary_metrics["stable_attempts"]
        )
        assert total_attempts == 3

    def test_rollback_detection(self, version_control):
        """Test rollback recommendation when canary performs poorly."""
        manager = CanaryWeightManager(version_control)
        manager.start_canary(canary_version=2, percentage=50.0, min_incidents=5)

        # Record outcomes: canary fails, stable succeeds
        for i in range(10):
            manager.record_outcome(f"inc-{i}-stable", success=True)
            manager.record_outcome(f"inc-{i}-canary", success=False)

        # Check rollback
        should_rollback = manager.check_rollback()
        assert should_rollback

    def test_no_rollback_when_canary_good(self, version_control):
        """Test no rollback when canary performs well."""
        manager = CanaryWeightManager(version_control)
        manager.start_canary(canary_version=2, percentage=50.0, min_incidents=5)

        # Record good outcomes for both
        for i in range(10):
            manager.record_outcome(f"inc-{i}-a", success=True)
            manager.record_outcome(f"inc-{i}-b", success=True)

        should_rollback = manager.check_rollback()
        assert not should_rollback

    def test_promote_canary(self, version_control):
        """Test promoting canary to stable."""
        manager = CanaryWeightManager(version_control)
        manager.start_canary(canary_version=2, percentage=50.0)

        manager.promote_canary()

        # After promotion, all traffic should go to canary (now stable)
        weights = manager.select_weight_version("inc-any")
        assert weights.confidence == 0.25

    def test_min_incidents_check(self, version_control):
        """Test rollback only triggers with minimum incidents."""
        manager = CanaryWeightManager(version_control)
        manager.start_canary(
            canary_version=2, percentage=50.0, min_incidents=100
        )

        # Record only 5 failed canary incidents
        for i in range(5):
            manager.record_outcome(f"inc-{i}-canary", success=False)

        # Not enough data; should not recommend rollback
        should_rollback = manager.check_rollback()
        assert not should_rollback
