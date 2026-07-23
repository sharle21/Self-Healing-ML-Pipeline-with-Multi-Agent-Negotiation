"""Tests for Phase 14: outcome-based reward from real IncidentState before/after."""

from __future__ import annotations

import pytest

from self_healing_pipeline.observability import IncidentState, IncidentStateBuilder, TelemetryCollector
from self_healing_pipeline.verification.reward import RewardCalculator, _clip


# ---------------------------------------------------------------------------
# Helpers: build IncidentState with controlled fields
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> IncidentState:
    """Build a minimal IncidentState with sensible defaults + caller overrides."""
    defaults = dict(
        tenant_id="standard",
        incident_type="drift",
        current_auc=0.75,
        baseline_auc=0.80,
        current_precision=0.82,
        baseline_precision=0.82,
        current_recall=0.71,
        baseline_recall=0.71,
        auc_drop=0.05,
        false_positive_rate=0.05,
        false_negative_rate=0.10,
        max_feature_drift=1.8,
        drifted_features=["LIMIT_BAL"],
        missing_rate=0.05,
        duplicate_rate=0.01,
        schema_violation_rate=0.0,
        latency_p95_ms=90.0,
        latency_p99_ms=140.0,
        latency_sla_ms=100.0,
        cost_per_1000_predictions=5.0,
        cost_budget_per_1000=2.0,
        current_threshold=0.50,
        current_model_version="v1",
        previous_model_version=None,
        last_training_age_days=30.0,
        min_auc=0.75,
        max_latency_ms=120.0,
        max_missing_rate=0.05,
        historical_agent_success={},
        severity=0.5,
        severity_components={"impact": 0.5, "deviation": 0.5, "urgency": 0.5, "business_risk": 0.5},
    )
    defaults.update(overrides)
    return IncidentState(**defaults)


class _FakeExecResult:
    def __init__(self, success=True, duration=1.0):
        self.success = success
        self.duration = duration


# ---------------------------------------------------------------------------
# check_resolved
# ---------------------------------------------------------------------------

class TestCheckResolved:

    def test_drift_resolved_auc_ok_and_drift_low(self):
        state = _make_state(
            incident_type="drift",
            current_auc=0.78,
            min_auc=0.75,
            max_feature_drift=1.2,  # < 1.5
        )
        assert RewardCalculator.check_resolved(state) is True

    def test_drift_not_resolved_auc_below_min(self):
        state = _make_state(
            incident_type="drift",
            current_auc=0.73,
            min_auc=0.75,
            max_feature_drift=0.5,
        )
        assert RewardCalculator.check_resolved(state) is False

    def test_drift_not_resolved_drift_still_high(self):
        state = _make_state(
            incident_type="drift",
            current_auc=0.80,
            min_auc=0.75,
            max_feature_drift=2.0,  # > 1.5
        )
        assert RewardCalculator.check_resolved(state) is False

    def test_data_quality_resolved(self):
        state = _make_state(
            incident_type="data_quality",
            missing_rate=0.03,
            max_missing_rate=0.05,
            schema_violation_rate=0.001,
        )
        assert RewardCalculator.check_resolved(state) is True

    def test_data_quality_not_resolved_high_missing(self):
        state = _make_state(
            incident_type="data_quality",
            missing_rate=0.10,
            max_missing_rate=0.05,
            schema_violation_rate=0.0,
        )
        assert RewardCalculator.check_resolved(state) is False

    def test_latency_resolved(self):
        state = _make_state(
            incident_type="latency_breach",
            latency_p95_ms=95.0,
            latency_sla_ms=100.0,
        )
        assert RewardCalculator.check_resolved(state) is True

    def test_latency_not_resolved(self):
        state = _make_state(
            incident_type="latency_breach",
            latency_p95_ms=115.0,
            latency_sla_ms=100.0,
        )
        assert RewardCalculator.check_resolved(state) is False

    def test_cost_resolved(self):
        state = _make_state(
            incident_type="cost_threshold",
            cost_per_1000_predictions=1.8,
            cost_budget_per_1000=2.0,
        )
        assert RewardCalculator.check_resolved(state) is True

    def test_cost_not_resolved(self):
        state = _make_state(
            incident_type="cost_threshold",
            cost_per_1000_predictions=3.0,
            cost_budget_per_1000=2.0,
        )
        assert RewardCalculator.check_resolved(state) is False

    def test_unknown_type_not_resolved(self):
        state = _make_state(incident_type="unknown")
        assert RewardCalculator.check_resolved(state) is False


# ---------------------------------------------------------------------------
# calculate_from_incident_states
# ---------------------------------------------------------------------------

class TestCalculateFromIncidentStates:

    def test_reward_clipped_to_minus_one_plus_one(self):
        before = _make_state()
        after = _make_state()
        reward, _ = RewardCalculator.calculate_from_incident_states(
            before, after, "threshold", _FakeExecResult()
        )
        assert -1.0 <= reward <= 1.0

    def test_perfect_auc_recovery_boosts_reward(self):
        before = _make_state(incident_type="drift", current_auc=0.70)
        after = _make_state(incident_type="drift", current_auc=0.80)  # +10% AUC
        reward_good, breakdown = RewardCalculator.calculate_from_incident_states(
            before, after, "retrain", _FakeExecResult()
        )
        # With 10% AUC recovery, quality_gain should be ~1.0
        assert breakdown.quality_gain == pytest.approx(1.0)
        assert reward_good > 0.0

    def test_no_change_modest_reward(self):
        """When before == after metrics, reward comes only from resolution_score."""
        state = _make_state(incident_type="drift", current_auc=0.75, min_auc=0.75)
        reward, breakdown = RewardCalculator.calculate_from_incident_states(
            state, state, "threshold", _FakeExecResult()
        )
        assert breakdown.quality_gain == pytest.approx(0.0)
        # resolved: auc=0.75 >= min_auc=0.75 and drift=1.8 — not resolved (drift high)
        assert breakdown.resolution_score == pytest.approx(0.0)

    def test_resolved_incident_increases_reward(self):
        before = _make_state(incident_type="drift", current_auc=0.72, max_feature_drift=2.0)
        after = _make_state(incident_type="drift", current_auc=0.78, max_feature_drift=1.2)
        reward, breakdown = RewardCalculator.calculate_from_incident_states(
            before, after, "retrain", _FakeExecResult()
        )
        assert breakdown.resolution_score == pytest.approx(1.0)
        assert reward > 0.0

    def test_latency_regression_penalizes(self):
        before = _make_state(incident_type="drift", latency_p95_ms=90.0)
        after = _make_state(incident_type="drift", latency_p95_ms=200.0)  # 110ms worse
        _, breakdown_bad = RewardCalculator.calculate_from_incident_states(
            before, after, "retrain", _FakeExecResult()
        )
        # No change baseline
        _, breakdown_flat = RewardCalculator.calculate_from_incident_states(
            before, before, "retrain", _FakeExecResult()
        )
        # Latency regression should penalize vs. flat
        assert breakdown_bad.regression_penalty > breakdown_flat.regression_penalty
        assert breakdown_bad.latency_gain < 0

    def test_cost_reduction_boosts_reward(self):
        before = _make_state(incident_type="cost_threshold", cost_per_1000_predictions=5.0)
        after = _make_state(incident_type="cost_threshold", cost_per_1000_predictions=3.0)
        reward, breakdown = RewardCalculator.calculate_from_incident_states(
            before, after, "threshold", _FakeExecResult()
        )
        assert breakdown.cost_gain > 0

    def test_slow_execution_time_penalty(self):
        state = _make_state()
        _, fast = RewardCalculator.calculate_from_incident_states(
            state, state, "retrain", _FakeExecResult(duration=1.0)
        )
        _, slow = RewardCalculator.calculate_from_incident_states(
            state, state, "retrain", _FakeExecResult(duration=300.0)
        )
        assert slow.time_penalty > fast.time_penalty
        assert slow.time_penalty == pytest.approx(0.10)  # max time penalty

    def test_threshold_agent_zero_exec_cost(self):
        state = _make_state()
        _, breakdown = RewardCalculator.calculate_from_incident_states(
            state, state, "threshold", _FakeExecResult()
        )
        assert breakdown.exec_cost_penalty == pytest.approx(0.0)

    def test_retrain_agent_has_exec_cost(self):
        state = _make_state()
        _, breakdown = RewardCalculator.calculate_from_incident_states(
            state, state, "retrain", _FakeExecResult()
        )
        assert breakdown.exec_cost_penalty == pytest.approx(0.10)

    def test_data_quality_weights_favor_reliability(self):
        """For data_quality incident, reliability_gain has highest weight."""
        before = _make_state(incident_type="data_quality", missing_rate=0.20, max_missing_rate=0.05)
        after_fixed = _make_state(incident_type="data_quality", missing_rate=0.02, max_missing_rate=0.05)
        reward_fixed, breakdown = RewardCalculator.calculate_from_incident_states(
            before, after_fixed, "data_repair", _FakeExecResult()
        )
        assert breakdown.reliability_gain > 0
        assert reward_fixed > 0

    def test_breakdown_contains_all_fields(self):
        state = _make_state()
        _, breakdown = RewardCalculator.calculate_from_incident_states(
            state, state, "threshold", _FakeExecResult()
        )
        assert hasattr(breakdown, "quality_gain")
        assert hasattr(breakdown, "cost_gain")
        assert hasattr(breakdown, "reliability_gain")
        assert hasattr(breakdown, "latency_gain")
        assert hasattr(breakdown, "resolution_score")
        assert hasattr(breakdown, "exec_cost_penalty")
        assert hasattr(breakdown, "time_penalty")
        assert hasattr(breakdown, "regression_penalty")
        assert hasattr(breakdown, "reward")
        assert hasattr(breakdown, "incident_type")
        assert hasattr(breakdown, "agent_type")

    def test_latency_incident_weights_favor_latency(self):
        """For latency_breach, latency_gain has highest weight (0.45)."""
        before = _make_state(incident_type="latency_breach", latency_p95_ms=180.0, latency_sla_ms=100.0)
        after = _make_state(incident_type="latency_breach", latency_p95_ms=85.0, latency_sla_ms=100.0)
        reward, breakdown = RewardCalculator.calculate_from_incident_states(
            before, after, "threshold", _FakeExecResult()
        )
        assert breakdown.latency_gain > 0.5
        assert reward > 0

    def test_auc_before_after_recorded(self):
        before = _make_state(current_auc=0.72)
        after = _make_state(current_auc=0.79)
        _, breakdown = RewardCalculator.calculate_from_incident_states(
            before, after, "retrain", _FakeExecResult()
        )
        assert breakdown.auc_before == pytest.approx(0.72)
        assert breakdown.auc_after == pytest.approx(0.79)

    def test_none_auc_falls_back_to_baseline(self):
        """current_auc=None → use baseline_auc for delta computation."""
        before = _make_state(current_auc=None, baseline_auc=0.80)
        after = _make_state(current_auc=None, baseline_auc=0.80)
        reward, breakdown = RewardCalculator.calculate_from_incident_states(
            before, after, "threshold", _FakeExecResult()
        )
        assert breakdown.quality_gain == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _clip helper
# ---------------------------------------------------------------------------

class TestClip:
    def test_clip_within(self): assert _clip(0.5, 0.0, 1.0) == pytest.approx(0.5)
    def test_clip_above(self):  assert _clip(2.0, 0.0, 1.0) == pytest.approx(1.0)
    def test_clip_below(self):  assert _clip(-2.0, -1.0, 1.0) == pytest.approx(-1.0)
