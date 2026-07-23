"""Tests for Phase 13: GuardrailChecker — multi-dimensional post-action guardrails."""

from __future__ import annotations

import pytest

from self_healing_pipeline.observability import IncidentState
from self_healing_pipeline.verification.guardrails import GuardrailChecker, GuardrailResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**overrides) -> IncidentState:
    defaults = dict(
        tenant_id="standard",
        incident_type="drift",
        current_auc=0.82,
        baseline_auc=0.85,
        current_precision=0.80,
        baseline_precision=0.85,
        current_recall=0.78,
        baseline_recall=0.83,
        auc_drop=0.03,
        false_positive_rate=0.04,
        false_negative_rate=0.08,
        max_feature_drift=1.2,
        drifted_features=[],
        missing_rate=0.02,
        duplicate_rate=0.01,
        schema_violation_rate=0.0,
        latency_p95_ms=85.0,
        latency_p99_ms=120.0,
        latency_sla_ms=100.0,
        cost_per_1000_predictions=4.0,
        cost_budget_per_1000=5.0,
        current_threshold=0.50,
        current_model_version="v2",
        previous_model_version="v1",
        last_training_age_days=5.0,
        min_auc=0.75,
        max_latency_ms=150.0,
        max_missing_rate=0.05,
        historical_agent_success={},
        severity=0.3,
        severity_components={"auc_drop": 0.3, "drift": 0.2, "affected_volume": 0.1},
    )
    defaults.update(overrides)
    return IncidentState(**defaults)


# ---------------------------------------------------------------------------
# Happy path: all guardrails pass
# ---------------------------------------------------------------------------

class TestCleanPass:
    def test_resolved_true_when_all_limits_met(self):
        before = _state(false_negative_rate=0.08, cost_per_1000_predictions=4.0)
        after = _state(current_auc=0.82, latency_p95_ms=85.0, missing_rate=0.02,
                       false_negative_rate=0.08, cost_per_1000_predictions=4.0)
        result = GuardrailChecker.check(before, after)
        assert result.resolved is True

    def test_no_regression_true_when_cost_and_fnr_stable(self):
        before = _state(false_negative_rate=0.08, cost_per_1000_predictions=4.0)
        after = _state(false_negative_rate=0.08, cost_per_1000_predictions=4.0)
        result = GuardrailChecker.check(before, after)
        assert result.no_regression is True

    def test_should_rollback_false_when_resolved(self):
        before = _state()
        after = _state()
        result = GuardrailChecker.check(before, after)
        assert result.should_rollback is False

    def test_no_violations_on_clean_pass(self):
        before = _state()
        after = _state()
        result = GuardrailChecker.check(before, after)
        assert result.violations == []

    def test_all_dimension_flags_true_on_clean_pass(self):
        before = _state()
        after = _state()
        result = GuardrailChecker.check(before, after)
        assert result.auc_ok
        assert result.latency_ok
        assert result.missing_ok
        assert result.cost_ok
        assert result.fnr_ok


# ---------------------------------------------------------------------------
# Resolution guardrail failures
# ---------------------------------------------------------------------------

class TestResolutionGuardrails:
    def test_auc_below_min_fails_resolved(self):
        before = _state()
        after = _state(current_auc=0.70, min_auc=0.75)  # 0.70 < 0.75
        result = GuardrailChecker.check(before, after)
        assert result.resolved is False
        assert result.auc_ok is False

    def test_auc_exactly_at_min_is_ok(self):
        before = _state()
        after = _state(current_auc=0.75, min_auc=0.75)
        result = GuardrailChecker.check(before, after)
        assert result.auc_ok is True

    def test_auc_none_treated_as_ok(self):
        before = _state()
        after = _state(current_auc=None)
        result = GuardrailChecker.check(before, after)
        assert result.auc_ok is True

    def test_latency_above_sla_fails_resolved(self):
        before = _state()
        after = _state(latency_p95_ms=120.0, latency_sla_ms=100.0)
        result = GuardrailChecker.check(before, after)
        assert result.resolved is False
        assert result.latency_ok is False

    def test_latency_exactly_at_sla_is_ok(self):
        before = _state()
        after = _state(latency_p95_ms=100.0, latency_sla_ms=100.0)
        result = GuardrailChecker.check(before, after)
        assert result.latency_ok is True

    def test_missing_rate_above_max_fails_resolved(self):
        before = _state()
        after = _state(missing_rate=0.08, max_missing_rate=0.05)
        result = GuardrailChecker.check(before, after)
        assert result.resolved is False
        assert result.missing_ok is False

    def test_missing_rate_exactly_at_max_is_ok(self):
        before = _state()
        after = _state(missing_rate=0.05, max_missing_rate=0.05)
        result = GuardrailChecker.check(before, after)
        assert result.missing_ok is True

    def test_all_three_resolution_failures_accumulated(self):
        before = _state()
        after = _state(current_auc=0.60, latency_p95_ms=200.0, missing_rate=0.20)
        result = GuardrailChecker.check(before, after)
        assert result.resolved is False
        assert len(result.violations) >= 3

    def test_violation_message_contains_metric_name(self):
        before = _state()
        after = _state(current_auc=0.70, min_auc=0.75)
        result = GuardrailChecker.check(before, after)
        assert any("auc" in v for v in result.violations)

    def test_latency_violation_message_contains_value(self):
        before = _state()
        after = _state(latency_p95_ms=130.0, latency_sla_ms=100.0)
        result = GuardrailChecker.check(before, after)
        violation = next(v for v in result.violations if "latency" in v)
        assert "130.0" in violation


# ---------------------------------------------------------------------------
# Regression guardrail failures
# ---------------------------------------------------------------------------

class TestRegressionGuardrails:
    def test_cost_spike_beyond_110pct_fails_no_regression(self):
        before = _state(cost_per_1000_predictions=4.0)
        after = _state(cost_per_1000_predictions=4.5)  # 4.5/4.0 = 1.125 > 1.10
        result = GuardrailChecker.check(before, after)
        assert result.no_regression is False
        assert result.cost_ok is False

    def test_cost_exactly_at_110pct_is_ok(self):
        before = _state(cost_per_1000_predictions=4.0)
        after = _state(cost_per_1000_predictions=4.4)  # exactly 1.10x
        result = GuardrailChecker.check(before, after)
        assert result.cost_ok is True

    def test_fnr_increase_beyond_005_fails_no_regression(self):
        before = _state(false_negative_rate=0.10)
        after = _state(false_negative_rate=0.16)  # +0.06 > 0.05 allowed
        result = GuardrailChecker.check(before, after)
        assert result.no_regression is False
        assert result.fnr_ok is False

    def test_fnr_increase_exactly_at_005_is_ok(self):
        before = _state(false_negative_rate=0.10)
        after = _state(false_negative_rate=0.15)  # exactly +0.05
        result = GuardrailChecker.check(before, after)
        assert result.fnr_ok is True

    def test_cost_violation_message_shows_multiplier(self):
        before = _state(cost_per_1000_predictions=4.0)
        after = _state(cost_per_1000_predictions=5.0)
        result = GuardrailChecker.check(before, after)
        violation = next(v for v in result.violations if "cost" in v)
        assert "1.1" in violation

    def test_fnr_violation_message_shows_ceiling(self):
        before = _state(false_negative_rate=0.10)
        after = _state(false_negative_rate=0.20)
        result = GuardrailChecker.check(before, after)
        violation = next(v for v in result.violations if "fnr" in v)
        assert "0.15" in violation  # before + 0.05


# ---------------------------------------------------------------------------
# should_rollback logic
# ---------------------------------------------------------------------------

class TestShouldRollback:
    def test_rollback_when_not_resolved_and_regression(self):
        before = _state(false_negative_rate=0.08, cost_per_1000_predictions=4.0)
        after = _state(
            current_auc=0.60,          # AUC below min → not resolved
            false_negative_rate=0.20,  # FNR spiked → regression
        )
        result = GuardrailChecker.check(before, after)
        assert result.should_rollback is True

    def test_no_rollback_when_resolved(self):
        before = _state()
        after = _state()  # all guardrails pass
        result = GuardrailChecker.check(before, after)
        assert result.should_rollback is False

    def test_no_rollback_when_not_resolved_but_no_regression(self):
        # AUC below min BUT no cost/FNR regression → don't rollback
        before = _state(false_negative_rate=0.08, cost_per_1000_predictions=4.0)
        after = _state(
            current_auc=0.70,          # AUC below min → not resolved
            false_negative_rate=0.08,  # FNR stable
            cost_per_1000_predictions=4.0,  # cost stable
        )
        result = GuardrailChecker.check(before, after)
        assert result.resolved is False
        assert result.no_regression is True
        assert result.should_rollback is False

    def test_no_rollback_when_resolved_even_if_minor_cost_increase(self):
        # Resolved + cost spiked: no rollback (incident resolved takes precedence)
        before = _state(cost_per_1000_predictions=4.0)
        after = _state(
            current_auc=0.82,
            latency_p95_ms=85.0,
            missing_rate=0.02,
            cost_per_1000_predictions=5.0,  # cost spike → no_regression=False
        )
        result = GuardrailChecker.check(before, after)
        assert result.resolved is True
        assert result.should_rollback is False


# ---------------------------------------------------------------------------
# Commander integration: Phase 13 fields in result
# ---------------------------------------------------------------------------

class TestCommanderGuardrailIntegration:

    @pytest.mark.asyncio
    async def test_result_has_guardrail_fields(self):
        from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
        from self_healing_pipeline.commander.commander_v3 import CommanderV3
        from self_healing_pipeline.gateway.events import Incident, IncidentType

        agents = [RetrainAgent("retrain-1")]
        commander = CommanderV3(agents)
        incident = Incident(
            tenant_id="standard",
            type=IncidentType.DRIFT,
            payload={"auc": 0.72},
            severity=0.6,
        )
        result = await commander.handle_incident(incident)
        assert hasattr(result, "guardrail_violations")
        assert hasattr(result, "auto_rollback_triggered")
        assert isinstance(result.guardrail_violations, list)
        assert isinstance(result.auto_rollback_triggered, bool)

    @pytest.mark.asyncio
    async def test_verification_breakdown_has_guardrail_keys(self):
        from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
        from self_healing_pipeline.commander.commander_v3 import CommanderV3
        from self_healing_pipeline.gateway.events import Incident, IncidentType

        agents = [ThresholdAdjustmentAgent("threshold-1")]
        commander = CommanderV3(agents)
        incident = Incident(
            tenant_id="standard",
            type=IncidentType.DRIFT,
            payload={},
            severity=0.5,
        )
        result = await commander.handle_incident(incident)
        bd = result.verification_breakdown
        assert "guardrail_resolved" in bd
        assert "guardrail_no_regression" in bd
        assert "guardrail_violations" in bd
        assert "auto_rollback_triggered" in bd

    @pytest.mark.asyncio
    async def test_incident_resolved_from_guardrail_not_score(self):
        """incident_resolved must reflect GuardrailChecker verdict, not reward threshold."""
        from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
        from self_healing_pipeline.commander.commander_v3 import CommanderV3
        from self_healing_pipeline.gateway.events import Incident, IncidentType

        agents = [RetrainAgent("retrain-1")]
        commander = CommanderV3(agents)
        incident = Incident(
            tenant_id="standard",
            type=IncidentType.DRIFT,
            payload={},
            severity=0.5,
        )
        result = await commander.handle_incident(incident)
        # guardrail_resolved and incident_resolved must agree
        assert result.incident_resolved == result.verification_breakdown["guardrail_resolved"]

    @pytest.mark.asyncio
    async def test_auto_rollback_not_triggered_without_rollback_agent(self):
        """No rollback agents in pool → auto_rollback_triggered stays False."""
        from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
        from self_healing_pipeline.commander.commander_v3 import CommanderV3
        from self_healing_pipeline.gateway.events import Incident, IncidentType

        agents = [RetrainAgent("retrain-1")]  # no rollback agent
        commander = CommanderV3(agents)
        incident = Incident(
            tenant_id="standard",
            type=IncidentType.DRIFT,
            payload={},
            severity=0.5,
        )
        result = await commander.handle_incident(incident)
        # Even if guardrail fails, no rollback agent → stays False
        assert result.auto_rollback_triggered is False
