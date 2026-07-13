"""Tests for verification layer (reward calculation)."""

import pytest

from self_healing_pipeline.observability import TelemetryCollector
from self_healing_pipeline.verification import RewardCalculator


class TestRewardCalculator:
    """Test reward calculation."""

    def test_drift_reward_improvement(self):
        """Test drift incident reward when improvement detected."""
        collector = TelemetryCollector(use_mock=True)
        state_before = collector.collect()

        # Mock state after: AUC improved
        state_after = collector.collect()
        state_after.model.auc = 0.82  # Improved from 0.75

        reward, breakdown = RewardCalculator.calculate_drift_reward(
            state_before, state_after, "retrain", incident_resolved=True
        )

        assert -1 <= reward <= 1
        assert breakdown.metric_improvement > 0
        assert breakdown.incident_resolution == 1.0

    def test_drift_reward_no_improvement(self):
        """Test drift reward when no improvement."""
        collector = TelemetryCollector(use_mock=True)
        state_before = collector.collect()
        state_after = collector.collect()

        reward, breakdown = RewardCalculator.calculate_drift_reward(
            state_before, state_after, "threshold", incident_resolved=False
        )

        assert reward < 0.5  # Should be low without improvement

    def test_data_quality_reward(self):
        """Test data quality incident reward."""
        collector = TelemetryCollector(use_mock=True)
        state_before = collector.collect()

        state_after = collector.collect()
        state_after.data.missing_rate = 0.02  # Improved from 0.08

        reward, breakdown = RewardCalculator.calculate_data_quality_reward(
            state_before, state_after, "data_repair", incident_resolved=True
        )

        assert 0 <= reward <= 1
        assert breakdown.metric_improvement > 0

    def test_latency_reward(self):
        """Test latency incident reward."""
        collector = TelemetryCollector(use_mock=True)
        state_before = collector.collect()

        state_after = collector.collect()
        state_after.system.latency_p95 = 70  # Improved from 85

        reward, breakdown = RewardCalculator.calculate_latency_reward(
            state_before, state_after, "rollback", incident_resolved=True
        )

        assert 0 <= reward <= 1

    def test_cost_reward(self):
        """Test cost threshold incident reward."""
        collector = TelemetryCollector(use_mock=True)
        state_before = collector.collect()

        state_after = collector.collect()
        state_after.system.cost_per_prediction = 0.0015  # Reduced from 0.002

        reward, breakdown = RewardCalculator.calculate_cost_reward(
            state_before, state_after, "threshold", incident_resolved=True
        )

        assert 0 <= reward <= 1

    def test_reward_risk_penalty(self):
        """Test that risk penalty applies correctly."""
        collector = TelemetryCollector(use_mock=True)
        state_before = collector.collect()
        state_after = collector.collect()

        # Data repair has higher risk penalty
        reward_repair, _ = RewardCalculator.calculate_drift_reward(
            state_before, state_after, "data_repair"
        )

        # Threshold has lower risk penalty
        reward_threshold, _ = RewardCalculator.calculate_drift_reward(
            state_before, state_after, "threshold"
        )

        # Same conditions, but data_repair should have lower reward due to risk
        assert reward_repair < reward_threshold
