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
        result = WeightTuner.tune(analysis, current)
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
        result = WeightTuner.tune(analysis, current, aggressiveness=0.1)
        # Confidence should increase
        assert result.confidence > current.confidence

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
        result = WeightTuner.tune(analysis, current, aggressiveness=0.1)
        # Business_value should increase
        assert result.business_value > current.business_value

    def test_tune_reconciliations_boosts_historical_success(self):
        """Test that reconciliations boost historical_success weight."""
        analysis = AnalysisResult(
            total_incidents=5,
            agent_metrics={},
            reconciliations_triggered=3,
            high_performers=[],
            low_performers=[],
        )
        current = ScoringWeights(historical_success=0.15)
        result = WeightTuner.tune(analysis, current, aggressiveness=0.1)
        # Historical_success should increase
        assert result.historical_success > current.historical_success

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
        result = WeightTuner.tune(analysis, aggressiveness=1.0)
        assert result.total() == pytest.approx(1.0)

    def test_tune_aggressiveness_controls_magnitude(self):
        """Test that aggressiveness parameter controls adjustment magnitude."""
        analysis = AnalysisResult(
            total_incidents=5,
            agent_metrics={},
            reconciliations_triggered=0,
            high_performers=["threshold"],
            low_performers=[],
        )
        current = ScoringWeights()

        result_mild = WeightTuner.tune(analysis, current, aggressiveness=0.05)
        result_aggressive = WeightTuner.tune(analysis, current, aggressiveness=0.5)

        # Aggressive should have larger adjustment
        mild_delta = result_mild.confidence - current.confidence
        aggressive_delta = result_aggressive.confidence - current.confidence
        assert aggressive_delta > mild_delta

    def test_adjustment_reason_high_performers(self):
        """Test adjustment reason message for high performers."""
        analysis = AnalysisResult(
            total_incidents=5,
            agent_metrics={},
            reconciliations_triggered=0,
            high_performers=["threshold"],
            low_performers=[],
        )
        reason = WeightTuner.compute_adjustment_reason(analysis)
        assert "High performers" in reason
        assert "threshold" in reason
        assert "Boosted confidence" in reason

    def test_adjustment_reason_low_performers(self):
        """Test adjustment reason message for low performers."""
        analysis = AnalysisResult(
            total_incidents=5,
            agent_metrics={},
            reconciliations_triggered=0,
            high_performers=[],
            low_performers=["retrain"],
        )
        reason = WeightTuner.compute_adjustment_reason(analysis)
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
        reason = WeightTuner.compute_adjustment_reason(analysis)
        assert "No significant" in reason
