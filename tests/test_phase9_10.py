"""Tests for Phase 9 (per-type severity) + Phase 10 (smarter agent proposals)."""

from __future__ import annotations

import asyncio

import pytest

from self_healing_pipeline.agents.datarepair_v2 import DataRepairAgent
from self_healing_pipeline.agents.fallback_v2 import FallbackAgent
from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
from self_healing_pipeline.agents.rollback_v2 import RollbackAgent
from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
from self_healing_pipeline.gateway.events import IncidentType
from self_healing_pipeline.observability.severity import SeverityBreakdown, SeverityCalculator
from self_healing_pipeline.observability.telemetry import TelemetryCollector


def _telemetry():
    return TelemetryCollector(use_mock=True).collect()


# ---------------------------------------------------------------------------
# Phase 9: per-type severity formulas
# ---------------------------------------------------------------------------

class TestDriftSeverity:
    def test_components_named_correctly(self):
        calc = SeverityCalculator()
        _, bd = calc.calculate(IncidentType.DRIFT, _telemetry())
        assert "auc_drop" in bd.components
        assert "drift" in bd.components
        assert "affected_volume" in bd.components

    def test_three_components_sum_to_one_weight(self):
        # Weights: 0.45 + 0.35 + 0.20 = 1.0
        calc = SeverityCalculator()
        s, bd = calc.calculate(IncidentType.DRIFT, _telemetry())
        reconstructed = 0.45 * bd.components["auc_drop"] + 0.35 * bd.components["drift"] + 0.20 * bd.components["affected_volume"]
        assert s == pytest.approx(reconstructed, abs=1e-9)

    def test_all_components_in_0_1(self):
        _, bd = SeverityCalculator().calculate(IncidentType.DRIFT, _telemetry())
        for k, v in bd.components.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_backward_compat_impact_property(self):
        _, bd = SeverityCalculator().calculate(IncidentType.DRIFT, _telemetry())
        assert bd.impact == bd.components["auc_drop"]

    def test_backward_compat_deviation_property(self):
        _, bd = SeverityCalculator().calculate(IncidentType.DRIFT, _telemetry())
        assert bd.deviation == bd.components["drift"]


class TestDataQualitySeverity:
    def test_components_named_correctly(self):
        _, bd = SeverityCalculator().calculate(IncidentType.DATA_QUALITY, _telemetry())
        assert "missing_rate" in bd.components
        assert "schema_rate" in bd.components
        assert "duplicate_rate" in bd.components
        assert "affected_volume" in bd.components

    def test_four_components_sum_to_one_weight(self):
        _, bd = SeverityCalculator().calculate(IncidentType.DATA_QUALITY, _telemetry())
        c = bd.components
        reconstructed = 0.40 * c["missing_rate"] + 0.25 * c["schema_rate"] + 0.15 * c["duplicate_rate"] + 0.20 * c["affected_volume"]
        assert bd.severity == pytest.approx(reconstructed, abs=1e-9)

    def test_high_missing_rate_raises_severity(self):
        t_normal = _telemetry()
        t_bad = _telemetry()
        t_bad.data.missing_rate = 0.50
        s_normal, _ = SeverityCalculator().calculate(IncidentType.DATA_QUALITY, t_normal)
        s_bad, _ = SeverityCalculator().calculate(IncidentType.DATA_QUALITY, t_bad)
        assert s_bad > s_normal


class TestLatencySeverity:
    def test_components_named_correctly(self):
        _, bd = SeverityCalculator().calculate(IncidentType.LATENCY_BREACH, _telemetry())
        assert "latency_ratio" in bd.components
        assert "error_rate" in bd.components
        assert "traffic_volume" in bd.components

    def test_latency_ratio_formula(self):
        t = _telemetry()
        cfg = {"latency_sla_ms": 100.0}
        _, bd = SeverityCalculator().calculate(IncidentType.LATENCY_BREACH, t, tenant_config=cfg)
        reconstructed = 0.60 * bd.components["latency_ratio"] + 0.25 * bd.components["error_rate"] + 0.15 * bd.components["traffic_volume"]
        assert bd.severity == pytest.approx(reconstructed, abs=1e-9)


class TestCostSeverity:
    def test_components_named_correctly(self):
        _, bd = SeverityCalculator().calculate(IncidentType.COST_THRESHOLD, _telemetry())
        assert "budget_overrun" in bd.components
        assert "cost_growth" in bd.components

    def test_cost_formula(self):
        _, bd = SeverityCalculator().calculate(IncidentType.COST_THRESHOLD, _telemetry())
        reconstructed = 0.70 * bd.components["budget_overrun"] + 0.30 * bd.components["cost_growth"]
        assert bd.severity == pytest.approx(reconstructed, abs=1e-9)


# ---------------------------------------------------------------------------
# Phase 10: ThresholdAgent — threshold search
# ---------------------------------------------------------------------------

class TestThresholdSearch:
    """Phase 10: 61-candidate threshold search minimizes FP/FN business cost."""

    def _plan(self, **state_overrides):
        state = {
            "current_threshold": 0.50,
            "false_positive_rate": 0.05,
            "false_negative_rate": 0.10,
            "cost_false_positive": 20.0,
            "cost_false_negative": 500.0,
            "recall_drop": 0.10,
            "historical_threshold_success": 0.75,
            "tenant_id": "standard",
        }
        state.update(state_overrides)
        return asyncio.run(ThresholdAdjustmentAgent("t-1").analyze(state))

    def test_high_fn_cost_lowers_threshold(self):
        plan = self._plan(cost_false_negative=500.0, cost_false_positive=20.0)
        assert plan.expected_effect["new_threshold"] < 0.50

    def test_high_fp_cost_raises_threshold(self):
        plan = self._plan(cost_false_positive=500.0, cost_false_negative=5.0)
        assert plan.expected_effect["new_threshold"] > 0.50

    def test_balanced_cost_stays_near_current(self):
        plan = self._plan(
            cost_false_positive=50.0,
            cost_false_negative=50.0,
            false_positive_rate=0.05,
            false_negative_rate=0.05,
        )
        # With equal costs + equal rates, optimal ≈ current
        assert 0.40 <= plan.expected_effect["new_threshold"] <= 0.60

    def test_threshold_stays_in_valid_range(self):
        for cost_fn in [10, 100, 500, 2000]:
            plan = self._plan(cost_false_negative=cost_fn)
            t = plan.expected_effect["new_threshold"]
            assert 0.10 <= t <= 0.90, f"threshold {t} out of [0.10, 0.90] at cost_fn={cost_fn}"

    def test_expected_effect_has_fpr_fnr_delta(self):
        plan = self._plan()
        assert "false_positive_rate_delta" in plan.expected_effect
        assert "false_negative_rate_delta" in plan.expected_effect

    def test_estimated_cost_in_expected_effect(self):
        plan = self._plan()
        assert "estimated_cost_per_prediction" in plan.expected_effect
        assert plan.expected_effect["estimated_cost_per_prediction"] >= 0

    def test_reasoning_mentions_search(self):
        plan = self._plan()
        assert "61 candidates" in plan.reasoning

    def test_execute_uses_new_threshold_from_search(self):
        plan = self._plan(cost_false_negative=500.0)
        assert plan.expected_effect["new_threshold"] != pytest.approx(0.50)
        result = asyncio.run(ThresholdAdjustmentAgent("t-1").execute(plan))
        assert result.success is True


# ---------------------------------------------------------------------------
# Phase 10: RetrainAgent — training cost in expected_effect
# ---------------------------------------------------------------------------

class TestRetrainAgentPhase10:
    def test_expected_effect_has_auc_delta(self):
        state = {"drift_score": 1.8, "auc_drop": 0.08, "data_quality_score": 0.92, "model_age_days": 30}
        plan = asyncio.run(RetrainAgent("r-1").analyze(state))
        assert "auc_delta" in plan.expected_effect

    def test_expected_effect_has_training_cost(self):
        state = {"drift_score": 1.8, "auc_drop": 0.08, "data_quality_score": 0.92, "model_age_days": 30}
        plan = asyncio.run(RetrainAgent("r-1").analyze(state))
        assert "estimated_training_cost_usd" in plan.expected_effect
        assert plan.expected_effect["estimated_training_cost_usd"] > 0

    def test_old_model_has_higher_training_cost(self):
        state_fresh = {"drift_score": 1.5, "auc_drop": 0.06, "data_quality_score": 0.92, "model_age_days": 5}
        state_stale = {"drift_score": 1.5, "auc_drop": 0.06, "data_quality_score": 0.92, "model_age_days": 90}
        plan_fresh = asyncio.run(RetrainAgent("r-1").analyze(state_fresh))
        plan_stale = asyncio.run(RetrainAgent("r-1").analyze(state_stale))
        assert plan_stale.expected_effect["estimated_training_cost_usd"] > plan_fresh.expected_effect["estimated_training_cost_usd"]

    def test_poor_data_quality_reduces_confidence(self):
        state_good = {"drift_score": 1.5, "auc_drop": 0.08, "data_quality_score": 0.95, "model_age_days": 30}
        state_bad = {"drift_score": 1.5, "auc_drop": 0.08, "data_quality_score": 0.52, "model_age_days": 30}
        plan_good = asyncio.run(RetrainAgent("r-1").analyze(state_good))
        plan_bad = asyncio.run(RetrainAgent("r-1").analyze(state_bad))
        assert plan_good.confidence > plan_bad.confidence


# ---------------------------------------------------------------------------
# Phase 10: RollbackAgent — deployment probability gates confidence
# ---------------------------------------------------------------------------

class TestRollbackAgentPhase10:
    def _plan(self, deployment_hours=4.0, incident_prob=0.8):
        state = {
            "current_auc": 0.70,
            "previous_auc": 0.80,
            "deployment_age_hours": deployment_hours,
            "current_error_rate": 0.15,
            "previous_error_rate": 0.09,
            "deployment_related_incident_probability": incident_prob,
            "historical_rollback_success": 0.91,
            "current_model": "v2",
            "previous_model": "v1",
        }
        return asyncio.run(RollbackAgent("rb-1").analyze(state))

    def test_high_deployment_prob_high_confidence(self):
        plan = self._plan(deployment_hours=2, incident_prob=0.90)
        assert plan.confidence > 0.6

    def test_low_deployment_prob_low_confidence(self):
        plan_high = self._plan(deployment_hours=2, incident_prob=0.90)
        plan_low = self._plan(deployment_hours=200, incident_prob=0.10)
        assert plan_high.confidence > plan_low.confidence

    def test_reasoning_mentions_deployment_prob(self):
        plan = self._plan()
        assert "deployment_prob" in plan.reasoning

    def test_expected_effect_has_unified_keys(self):
        plan = self._plan()
        assert "auc_delta" in plan.expected_effect
        assert "false_negative_rate_delta" in plan.expected_effect


# ---------------------------------------------------------------------------
# Phase 10: FallbackAgent — latency ratio confidence
# ---------------------------------------------------------------------------

class TestFallbackAgentPhase10:
    def _plan(self, latency_p95=85.0, error_rate=0.20):
        state = {
            "error_rate": error_rate,
            "latency_p95": latency_p95,
            "prediction_failure_rate": 0.10,
            "confidence_distribution_mean": 0.55,
            "missing_rate": 0.08,
            "acceptable_accuracy_loss": 0.05,
            "fallback_quality": 0.70,
            "historical_fallback_success": 0.85,
        }
        return asyncio.run(FallbackAgent("fb-1").analyze(state))

    def test_latency_above_sla_raises_confidence(self):
        plan_ok = self._plan(latency_p95=80.0)
        plan_breach = self._plan(latency_p95=300.0)
        assert plan_breach.confidence > plan_ok.confidence

    def test_expected_effect_has_latency_delta(self):
        plan = self._plan(latency_p95=200.0)
        assert "latency_p95_delta_ms" in plan.expected_effect
        assert plan.expected_effect["latency_p95_delta_ms"] < 0

    def test_expected_effect_has_auc_delta(self):
        plan = self._plan()
        assert "auc_delta" in plan.expected_effect


# ---------------------------------------------------------------------------
# Phase 10: DataRepairAgent — all 3 quality dimensions
# ---------------------------------------------------------------------------

class TestDataRepairAgentPhase10:
    def _plan(self, missing=0.20, dup=0.05, schema=30):
        state = {
            "missing_rate": missing,
            "duplicate_rate": dup,
            "schema_error_count": schema,
            "affected_features": ["LIMIT_BAL"],
            "available_backup_data": True,
            "data_pipeline_health": 0.70,
            "historical_repair_success": 0.75,
        }
        return asyncio.run(DataRepairAgent("dr-1").analyze(state))

    def test_expected_effect_has_all_deltas(self):
        plan = self._plan()
        assert "missing_rate_delta" in plan.expected_effect
        assert "duplicate_rate_delta" in plan.expected_effect
        assert "schema_error_delta" in plan.expected_effect
        assert "false_negative_rate_delta" in plan.expected_effect

    def test_high_schema_errors_increase_confidence(self):
        plan_low = self._plan(schema=5)
        plan_high = self._plan(schema=90)
        assert plan_high.confidence >= plan_low.confidence

    def test_no_backup_reduces_confidence(self):
        state_with = {
            "missing_rate": 0.20, "duplicate_rate": 0.05, "schema_error_count": 30,
            "affected_features": [], "available_backup_data": True,
            "data_pipeline_health": 0.70, "historical_repair_success": 0.75,
        }
        state_without = dict(state_with)
        state_without["available_backup_data"] = False
        plan_with = asyncio.run(DataRepairAgent("dr-1").analyze(state_with))
        plan_without = asyncio.run(DataRepairAgent("dr-1").analyze(state_without))
        assert plan_with.confidence > plan_without.confidence
