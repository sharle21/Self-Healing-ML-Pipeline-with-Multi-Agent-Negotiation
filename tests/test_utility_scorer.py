"""Tests for Phase 11: UtilityScorer and commander utility-based ranking."""

from __future__ import annotations

import asyncio

import pytest

from self_healing_pipeline.agents.fallback_v2 import FallbackAgent
from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
from self_healing_pipeline.agents.rollback_v2 import RollbackAgent
from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
from self_healing_pipeline.commander.utility import (
    UtilityScorer,
    UtilityWeights,
    _DEFAULT_UTILITY_WEIGHTS,
    _FALLBACK_WEIGHTS,
)
from self_healing_pipeline.gateway.events import Incident, IncidentType
from self_healing_pipeline.observability import IncidentState


def _make_state(**overrides) -> IncidentState:
    defaults = dict(
        tenant_id="standard",
        incident_type="drift",
        current_auc=0.72,
        baseline_auc=0.80,
        current_precision=0.80,
        baseline_precision=0.85,
        current_recall=0.70,
        baseline_recall=0.83,
        auc_drop=0.08,
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
        severity_components={"auc_drop": 0.5, "drift": 0.5, "affected_volume": 0.5},
    )
    defaults.update(overrides)
    return IncidentState(**defaults)


class _MockPlan:
    def __init__(
        self,
        agent_type="threshold",
        confidence=0.75,
        risk=0.05,
        auc_delta=0.0,
        fnr_delta=0.0,
        latency_delta=0.0,
        cost_delta=0.0,
        availability_delta=0.0,
    ):
        self.agent_type = agent_type
        self.confidence = confidence
        self.risk = risk
        self.expected_effect = {
            "auc_delta": auc_delta,
            "false_negative_rate_delta": fnr_delta,
            "latency_p95_delta_ms": latency_delta,
            "cost_delta_usd": cost_delta,
            "availability_delta": availability_delta,
        }


class _MockAgent:
    def __init__(self, agent_type):
        self.agent_type = agent_type


# ---------------------------------------------------------------------------
# UtilityWeights default tables
# ---------------------------------------------------------------------------

class TestDefaultWeights:
    def test_drift_has_highest_quality_weight(self):
        w = _DEFAULT_UTILITY_WEIGHTS["drift"]
        assert w.quality >= w.cost
        assert w.quality >= w.speed

    def test_data_quality_has_highest_reliability_weight(self):
        w = _DEFAULT_UTILITY_WEIGHTS["data_quality"]
        assert w.reliability >= w.quality
        assert w.reliability >= w.cost

    def test_latency_breach_has_highest_speed_weight(self):
        w = _DEFAULT_UTILITY_WEIGHTS["latency_breach"]
        assert w.speed >= w.quality
        assert w.speed >= w.reliability

    def test_cost_threshold_has_highest_cost_weight(self):
        w = _DEFAULT_UTILITY_WEIGHTS["cost_threshold"]
        assert w.cost >= w.quality
        assert w.cost >= w.speed

    def test_fallback_weights_used_for_unknown_type(self):
        state = _make_state(incident_type="unknown_future_type")
        plan = _MockPlan()
        score = UtilityScorer.score(plan, state)
        assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# UtilityScorer.score normalization
# ---------------------------------------------------------------------------

class TestUtilityScorerNormalization:

    def test_score_always_in_valid_range(self):
        # Extreme positive plan — raw sum bounded by weights (< 1.0), not clipped here
        state = _make_state()
        plan = _MockPlan(auc_delta=1.0, fnr_delta=-1.0, latency_delta=-500.0, confidence=1.0, risk=0.0)
        score = UtilityScorer.score(plan, state)
        assert -1.0 <= score <= 1.0
        assert score > 0.5  # strongly positive with all-good signals

    def test_score_negative_for_bad_plan(self):
        state = _make_state()
        plan = _MockPlan(auc_delta=-1.0, latency_delta=500.0, confidence=0.0, risk=1.0)
        score = UtilityScorer.score(plan, state)
        assert -1.0 <= score <= 1.0
        assert score < 0.0  # negative with all-bad signals

    def test_zero_effect_score_driven_by_confidence_minus_risk(self):
        state = _make_state()
        plan = _MockPlan(confidence=0.80, risk=0.05)
        w = _DEFAULT_UTILITY_WEIGHTS["drift"]
        expected_approx = w.confidence * 0.80 - w.risk * 0.05
        score = UtilityScorer.score(plan, state)
        # All effect dimensions = 0; score = confidence weight * 0.80 - risk weight * 0.05
        assert score == pytest.approx(expected_approx, abs=1e-9)

    def test_auc_improvement_raises_score(self):
        state = _make_state(incident_type="drift")
        plan_flat = _MockPlan(auc_delta=0.0)
        plan_good = _MockPlan(auc_delta=0.10)  # 10% gain = full quality signal
        assert UtilityScorer.score(plan_good, state) > UtilityScorer.score(plan_flat, state)

    def test_fnr_improvement_raises_reliability(self):
        state = _make_state(incident_type="data_quality", false_negative_rate=0.10)
        plan_flat = _MockPlan(fnr_delta=0.0)
        plan_good = _MockPlan(fnr_delta=-0.05)  # FNR down by 50%
        assert UtilityScorer.score(plan_good, state) > UtilityScorer.score(plan_flat, state)

    def test_latency_improvement_raises_speed(self):
        state = _make_state(incident_type="latency_breach", latency_sla_ms=100.0)
        plan_flat = _MockPlan(latency_delta=0.0)
        plan_fast = _MockPlan(latency_delta=-50.0)  # 50ms faster
        assert UtilityScorer.score(plan_fast, state) > UtilityScorer.score(plan_flat, state)

    def test_high_risk_lowers_score(self):
        state = _make_state()
        plan_safe = _MockPlan(risk=0.05)
        plan_risky = _MockPlan(risk=0.80)
        assert UtilityScorer.score(plan_safe, state) > UtilityScorer.score(plan_risky, state)

    def test_availability_delta_boosts_reliability(self):
        state = _make_state()
        plan_no_avail = _MockPlan(availability_delta=0.0)
        plan_avail = _MockPlan(availability_delta=0.15)
        assert UtilityScorer.score(plan_avail, state) > UtilityScorer.score(plan_no_avail, state)


# ---------------------------------------------------------------------------
# UtilityScorer.rank
# ---------------------------------------------------------------------------

class TestUtilityScorerRank:

    def test_rank_returns_sorted_descending(self):
        state = _make_state(incident_type="drift")
        a1 = _MockAgent("retrain")
        p1 = _MockPlan(auc_delta=0.08, confidence=0.75, risk=0.15)
        a2 = _MockAgent("threshold")
        p2 = _MockPlan(auc_delta=0.00, confidence=0.65, risk=0.05)
        a3 = _MockAgent("fallback")
        p3 = _MockPlan(auc_delta=-0.05, latency_delta=-30, confidence=0.50, risk=0.15)

        ranked = UtilityScorer.rank([(a1, p1), (a2, p2), (a3, p3)], state)
        utilities = [u for _, _, u in ranked]
        assert utilities == sorted(utilities, reverse=True)

    def test_rank_includes_utility_score(self):
        state = _make_state()
        plan = _MockPlan()
        ranked = UtilityScorer.rank([(_MockAgent("threshold"), plan)], state)
        assert len(ranked) == 1
        agent, plan_out, util = ranked[0]
        assert -1.0 <= util <= 1.0

    def test_drift_prefers_retrain_over_fallback(self):
        """For drift, quality weight is high → retrain (AUC recovery) beats fallback."""
        state = _make_state(incident_type="drift")
        retrain_plan = _MockPlan("retrain", auc_delta=0.07, confidence=0.72, risk=0.15)
        fallback_plan = _MockPlan("fallback", latency_delta=-30.0, availability_delta=0.15, confidence=0.60, risk=0.15)

        ranked = UtilityScorer.rank(
            [(_MockAgent("retrain"), retrain_plan), (_MockAgent("fallback"), fallback_plan)],
            state,
        )
        assert ranked[0][0].agent_type == "retrain"

    def test_latency_breach_prefers_fast_agent(self):
        """For latency_breach, speed weight is high → fast agent wins."""
        state = _make_state(incident_type="latency_breach")
        threshold_plan = _MockPlan("threshold", auc_delta=0.01, confidence=0.80, risk=0.05)
        fallback_plan = _MockPlan("fallback", latency_delta=-60.0, confidence=0.60, risk=0.15)

        ranked = UtilityScorer.rank(
            [(_MockAgent("threshold"), threshold_plan), (_MockAgent("fallback"), fallback_plan)],
            state,
        )
        assert ranked[0][0].agent_type == "fallback"


# ---------------------------------------------------------------------------
# weights_from_tier_config
# ---------------------------------------------------------------------------

class TestWeightsFromTierConfig:
    def test_none_config_returns_per_type_default(self):
        w = UtilityScorer.weights_from_tier_config(None, "drift")
        assert w.quality == _DEFAULT_UTILITY_WEIGHTS["drift"].quality

    def test_tier_config_overrides_quality_weight(self):
        class FakeTierConfig:
            business_value_weight = 0.50
            cost_efficiency_weight = 0.05
            risk_inverse_weight = 0.10
            time_inverse_weight = 0.05
            confidence_weight = 0.15

        w = UtilityScorer.weights_from_tier_config(FakeTierConfig(), "drift")
        assert w.quality == pytest.approx(0.50)

    def test_tier_config_reliability_from_default(self):
        class FakeTierConfig:
            business_value_weight = 0.30
            cost_efficiency_weight = 0.10
            risk_inverse_weight = 0.15
            time_inverse_weight = 0.05
            confidence_weight = 0.20

        default_reliability = _DEFAULT_UTILITY_WEIGHTS["data_quality"].reliability
        w = UtilityScorer.weights_from_tier_config(FakeTierConfig(), "data_quality")
        assert w.reliability == pytest.approx(default_reliability)


# ---------------------------------------------------------------------------
# Commander integration: utility_score in result
# ---------------------------------------------------------------------------

class TestCommanderUtilityIntegration:

    @pytest.mark.asyncio
    async def test_result_has_utility_score(self):
        from self_healing_pipeline.commander.commander_v3 import CommanderV3

        agents = [RetrainAgent("retrain-1"), ThresholdAdjustmentAgent("threshold-1")]
        commander = CommanderV3(agents)
        incident = Incident(
            tenant_id="standard",
            type=IncidentType.DRIFT,
            payload={"auc": 0.72},
            severity=0.6,
        )
        result = await commander.handle_incident(incident)
        assert hasattr(result, "utility_score")
        assert -1.0 <= result.utility_score <= 1.0

    @pytest.mark.asyncio
    async def test_winner_selected_by_utility_not_just_confidence(self):
        """Retrain should win for drift incident (high AUC delta), not fallback (no AUC delta)."""
        from self_healing_pipeline.commander.commander_v3 import CommanderV3

        agents = [RetrainAgent("retrain-1"), FallbackAgent("fallback-1")]
        commander = CommanderV3(agents)
        incident = Incident(
            tenant_id="standard",
            type=IncidentType.DRIFT,
            payload={"drift_score": 2.0, "auc_drop": 0.08},
            severity=0.7,
        )
        result = await commander.handle_incident(incident)
        # For drift, quality weight is 0.40; retrain promises auc_delta > 0; fallback does not
        assert result.winning_agent_type == "retrain"
