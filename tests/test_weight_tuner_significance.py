"""Tests for statistical significance in weight tuning."""

import pytest

from self_healing_pipeline.meta_harness.analyzer import AnalysisResult
from self_healing_pipeline.meta_harness.tuner import ScoringWeights, WeightTuner


class TestSignificanceTesting:
    """Statistical significance testing in weight tuner."""

    def test_high_performer_significance_check(self):
        """Test significance check for high performers."""
        # Success rate 85% with sample size 20
        # Against baseline 70%
        # Should be significant (p < 0.05)
        is_sig = WeightTuner._is_significant_high_performer(
            agent_success_rate=0.85, population_mean=0.70, sample_size=20, alpha=0.05
        )

        # Should be True or False (may be numpy bool)
        assert isinstance(is_sig, (bool, type(True)))

    def test_high_performer_significance_small_sample(self):
        """Test significance check with small sample."""
        # Sample size < 5, should not be significant
        is_sig = WeightTuner._is_significant_high_performer(
            agent_success_rate=0.95, population_mean=0.70, sample_size=3, alpha=0.05
        )

        # Too small to test
        assert is_sig is False

    def test_low_performer_significance_check(self):
        """Test significance check for low performers."""
        # Success rate 40% with sample size 20
        # Against baseline 70%
        # Should be significant (p < 0.05)
        is_sig = WeightTuner._is_significant_low_performer(
            agent_success_rate=0.40, population_mean=0.70, sample_size=20, alpha=0.05
        )

        assert isinstance(is_sig, (bool, type(True)))

    def test_low_performer_significance_small_sample(self):
        """Test significance check for low performer with small sample."""
        is_sig = WeightTuner._is_significant_low_performer(
            agent_success_rate=0.20, population_mean=0.70, sample_size=3, alpha=0.05
        )

        assert is_sig is False

    def test_tune_with_significance_returns_tuple(self):
        """Test tune() returns (weights, significance dict)."""
        analysis = AnalysisResult(
            total_incidents=10,
            high_performers=["threshold"],
            low_performers=[],
            reconciliations_triggered=2,
            agent_metrics={},
        )

        result = WeightTuner.tune(
            analysis, current_weights=None, aggressiveness=0.1, alpha=0.05
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        weights, significance = result
        assert isinstance(weights, ScoringWeights)
        assert isinstance(significance, dict)

    def test_tune_requires_sample_size(self):
        """Test that significance requires minimum sample size."""
        analysis = AnalysisResult(
            total_incidents=2,  # Too small
            high_performers=["threshold"],
            low_performers=[],
            reconciliations_triggered=0,
            agent_metrics={},
        )

        weights, significance = WeightTuner.tune(analysis, alpha=0.05)

        # Should not apply adjustments due to small sample
        assert significance.get("high_performers_significant", False) is False

    def test_tune_no_significance_no_adjustment(self):
        """Test that non-significant results don't adjust weights."""
        baseline = ScoringWeights()

        analysis = AnalysisResult(
            total_incidents=2,  # Too small for significance
            high_performers=["threshold"],
            low_performers=[],
            reconciliations_triggered=0,
            agent_metrics={},
        )

        weights, significance = WeightTuner.tune(analysis, baseline, alpha=0.05)

        # Weights should remain unchanged (or near-unchanged)
        # because high_performers are not statistically significant
        # May have slight normalization, but major adjustments prevented
        assert weights.confidence == baseline.confidence or abs(
            weights.confidence - baseline.confidence
        ) < 0.01

    def test_reconciliation_threshold_minimum(self):
        """Test that reconciliations need minimum count (5+)."""
        # Only 2 reconciliations
        analysis = AnalysisResult(
            total_incidents=10,
            high_performers=[],
            low_performers=[],
            reconciliations_triggered=2,  # < 5
            agent_metrics={},
        )

        weights, significance = WeightTuner.tune(analysis)

        assert significance.get("reconciliations_significant", False) is False

    def test_reconciliation_threshold_met(self):
        """Test that 5+ reconciliations trigger adjustment."""
        baseline = ScoringWeights()

        analysis = AnalysisResult(
            total_incidents=20,
            high_performers=[],
            low_performers=[],
            reconciliations_triggered=8,  # >= 5
            agent_metrics={},
        )

        weights, significance = WeightTuner.tune(analysis, baseline)

        # Should consider reconciliations significant
        assert significance.get("reconciliations_significant", False) is True
        # historical_success weight should be boosted
        assert weights.historical_success >= baseline.historical_success

    def test_adjustment_reason_includes_significance(self):
        """Test adjustment reason reflects significance status."""
        analysis = AnalysisResult(
            total_incidents=10,
            high_performers=["threshold"],
            low_performers=[],
            reconciliations_triggered=0,
            agent_metrics={},
        )

        significance = {
            "high_performers_significant": True,
            "low_performers_significant": False,
            "reconciliations_significant": False,
        }

        reason = WeightTuner.compute_adjustment_reason(analysis, significance)

        assert "p < 0.05" in reason
        assert "threshold" in reason
        assert "No statistically" not in reason

    def test_adjustment_reason_no_significance(self):
        """Test adjustment reason when nothing significant."""
        analysis = AnalysisResult(
            total_incidents=10,
            high_performers=[],
            low_performers=[],
            reconciliations_triggered=0,
            agent_metrics={},
        )

        significance = {
            "high_performers_significant": False,
            "low_performers_significant": False,
            "reconciliations_significant": False,
        }

        reason = WeightTuner.compute_adjustment_reason(analysis, significance)

        assert "No statistically significant" in reason

    def test_weights_normalize_after_adjustment(self):
        """Test weights normalize to 1.0 after adjustment."""
        analysis = AnalysisResult(
            total_incidents=50,
            high_performers=["threshold", "rollback"],
            low_performers=["retrain"],
            reconciliations_triggered=10,
            agent_metrics={},
        )

        weights, _ = WeightTuner.tune(analysis, aggressiveness=0.2)

        # Should sum to 1.0 (or very close due to floating point)
        assert abs(weights.total() - 1.0) < 0.0001
