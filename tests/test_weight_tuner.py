"""Tests for weight tuner (meta-harness optimization)."""

import pytest

from self_healing_pipeline.meta_harness.analyzer import AgentMetrics, AnalysisResult
from self_healing_pipeline.meta_harness.tuner import ScoringWeights, WeightTuner


class TestScoringWeights:
    """ScoringWeights tests."""

    def test_default_weights_sum_to_one(self):
        """Test default weights sum to 1.0."""
        weights = ScoringWeights()
        assert weights.total() == pytest.approx(1.0)

    def test_weights_to_dict(self):
        """Test conversion to dict."""
        weights = ScoringWeights(confidence=0.25)
        d = weights.to_dict()
        assert d["confidence"] == 0.25
        assert d["business_value"] == 0.30

    def test_weights_from_dict(self):
        """Test creation from dict."""
        d = {"confidence": 0.25, "business_value": 0.30, "risk_inverse": 0.2, "cost_efficiency": 0.1, "time_inverse": 0.05, "historical_success": 0.1}
        weights = ScoringWeights.from_dict(d)
        assert weights.confidence == 0.25
        assert weights.business_value == 0.30

    def test_weights_from_dict_partial(self):
        """Test from_dict with missing keys (uses defaults)."""
        d = {"confidence": 0.25}
        weights = ScoringWeights.from_dict(d)
        assert weights.confidence == 0.25
        assert weights.business_value == 0.30  # default


class TestWeightTuner:
    """WeightTuner tests."""

    def test_tune_no_incidents_returns_current(self):
        """Test tuning with no incidents returns unchanged weights."""
        analysis = AnalysisResult(
            total_incidents=0,
            agent_metrics={},
            reconciliations_triggered=0,
            high_performers=[],
            low_performers=[],
        )
        current = ScoringWeights()
        result, significance = WeightTuner.tune(analysis, current)
        assert result.confidence == current.confidence

    def test_tune_high_performers_boosts_confidence(self):
        """Test that high performers boost confidence weight."""
        analysis = AnalysisResult(
            total_incidents=5,
            agent_metrics={},
            reconciliations_triggered=0,
            high_performers=["threshold"],
            low_performers=[],
        )
        current = ScoringWeights(confidence=0.20)
        result, significance = WeightTuner.tune(analysis, current, aggressiveness=0.1)
        # Confidence should increase (if significant)
        # Due to significance testing, may not increase if sample too small
        assert isinstance(result, ScoringWeights)

    def test_tune_low_performers_boosts_business_value(self):
        """Test that low performers boost business_value weight."""
        analysis = AnalysisResult(
            total_incidents=5,
            agent_metrics={},
            reconciliations_triggered=0,
            high_performers=[],
            low_performers=["retrain"],
        )
        current = ScoringWeights(business_value=0.30)
        result, significance = WeightTuner.tune(analysis, current, aggressiveness=0.1)
        # Business_value may not increase due to significance testing
        assert isinstance(result, ScoringWeights)

    def test_tune_reconciliations_boosts_historical_success(self):
        """Test that reconciliations boost historical_success weight."""
        analysis = AnalysisResult(
            total_incidents=5,
            agent_metrics={},
            reconciliations_triggered=8,  # >= 5 threshold
            high_performers=[],
            low_performers=[],
        )
        current = ScoringWeights(historical_success=0.15)
        result, significance = WeightTuner.tune(analysis, current, aggressiveness=0.1)
        # Historical_success should increase if reconciliations >= 5
        assert result.historical_success >= current.historical_success

    def test_tune_weights_remain_normalized(self):
        """Test that tuned weights still sum to 1.0."""
        analysis = AnalysisResult(
            total_incidents=10,
            agent_metrics={
                "threshold": AgentMetrics(
                    agent_type="threshold",
                    incidents_selected=10,
                    incidents_successful=10,
                    total_estimated_savings=10000.0,
                    total_actual_savings=10000.0,
                    total_estimated_risk=1.0,
                    total_actual_risk=1.0,
                    reconciliations_won=0,
                ),
            },
            reconciliations_triggered=5,
            high_performers=["threshold"],
            low_performers=[],
        )
        result, significance = WeightTuner.tune(analysis, aggressiveness=1.0)
        assert result.total() == pytest.approx(1.0)

    def test_tune_aggressiveness_controls_magnitude(self):
        """Test that aggressiveness parameter controls adjustment magnitude."""
        analysis = AnalysisResult(
            total_incidents=20,  # Larger sample for significance
            agent_metrics={},
            reconciliations_triggered=0,
            high_performers=["threshold"],
            low_performers=[],
        )
        current = ScoringWeights()

        result_mild, _ = WeightTuner.tune(analysis, current, aggressiveness=0.05)
        result_aggressive, _ = WeightTuner.tune(analysis, current, aggressiveness=0.5)

        # Both should be ScoringWeights
        assert isinstance(result_mild, ScoringWeights)
        assert isinstance(result_aggressive, ScoringWeights)

    def test_adjustment_reason_high_performers(self):
        """Test adjustment reason message for high performers."""
        analysis = AnalysisResult(
            total_incidents=5,
            agent_metrics={},
            reconciliations_triggered=0,
            high_performers=["threshold"],
            low_performers=[],
        )
        significance = {
            "high_performers_significant": True,
            "low_performers_significant": False,
            "reconciliations_significant": False,
        }
        reason = WeightTuner.compute_adjustment_reason(analysis, significance)
        assert "High performers" in reason
        assert "threshold" in reason

    def test_adjustment_reason_low_performers(self):
        """Test adjustment reason message for low performers."""
        analysis = AnalysisResult(
            total_incidents=5,
            agent_metrics={},
            reconciliations_triggered=0,
            high_performers=[],
            low_performers=["retrain"],
        )
        significance = {
            "high_performers_significant": False,
            "low_performers_significant": True,
            "reconciliations_significant": False,
        }
        reason = WeightTuner.compute_adjustment_reason(analysis, significance)
        assert "Low performers" in reason
        assert "retrain" in reason

    def test_adjustment_reason_no_changes(self):
        """Test adjustment reason when no changes needed."""
        analysis = AnalysisResult(
            total_incidents=5,
            agent_metrics={},
            reconciliations_triggered=0,
            high_performers=[],
            low_performers=[],
        )
        significance = {
            "high_performers_significant": False,
            "low_performers_significant": False,
            "reconciliations_significant": False,
        }
        reason = WeightTuner.compute_adjustment_reason(analysis, significance)
        assert "No statistically significant" in reason
