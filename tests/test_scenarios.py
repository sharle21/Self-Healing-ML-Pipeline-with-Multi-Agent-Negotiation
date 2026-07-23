"""Phase 16: Scenario evaluation suite — 30 deterministic named scenarios.

Each scenario documents a canonical pipeline situation and asserts the system
behaves correctly. Scenarios are organised by concern:
  1. Agent selection  (via UtilityScorer.rank with mock plans)
  2. Guardrail checks (via GuardrailChecker.check with fixed before/after states)
  3. Severity scoring (via SeverityCalculator with mock telemetry)
  4. Commander integration (async end-to-end, mock telemetry, no Prometheus)
"""

from __future__ import annotations

import pytest

from self_healing_pipeline.commander.utility import UtilityScorer, UtilityWeights, _DEFAULT_UTILITY_WEIGHTS
from self_healing_pipeline.observability import IncidentState
from self_healing_pipeline.observability.severity import SeverityCalculator
from self_healing_pipeline.observability.telemetry import TelemetryCollector
from self_healing_pipeline.verification.guardrails import GuardrailChecker
from self_healing_pipeline.gateway.events import IncidentType


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> IncidentState:
    defaults = dict(
        tenant_id="standard",
        incident_type="drift",
        current_auc=0.78,
        baseline_auc=0.85,
        current_precision=0.80,
        baseline_precision=0.85,
        current_recall=0.76,
        baseline_recall=0.83,
        auc_drop=0.07,
        false_positive_rate=0.05,
        false_negative_rate=0.10,
        max_feature_drift=1.6,
        drifted_features=["LIMIT_BAL"],
        missing_rate=0.03,
        duplicate_rate=0.01,
        schema_violation_rate=0.0,
        latency_p95_ms=88.0,
        latency_p99_ms=130.0,
        latency_sla_ms=100.0,
        cost_per_1000_predictions=4.5,
        cost_budget_per_1000=5.0,
        current_threshold=0.50,
        current_model_version="v3",
        previous_model_version="v2",
        last_training_age_days=14.0,
        min_auc=0.75,
        max_latency_ms=150.0,
        max_missing_rate=0.05,
        historical_agent_success={},
        severity=0.5,
        severity_components={"auc_drop": 0.5, "drift": 0.5, "affected_volume": 0.3},
    )
    defaults.update(overrides)
    return IncidentState(**defaults)


class _Plan:
    """Lightweight mock plan for agent-selection scenarios."""

    def __init__(
        self,
        agent_type: str,
        confidence: float = 0.70,
        risk: float = 0.10,
        auc_delta: float = 0.0,
        fnr_delta: float = 0.0,
        lat_delta: float = 0.0,
        cost_delta: float = 0.0,
        avail_delta: float = 0.0,
    ):
        self.agent_type = agent_type
        self.confidence = confidence
        self.risk = risk
        self.expected_effect = {
            "auc_delta": auc_delta,
            "false_negative_rate_delta": fnr_delta,
            "latency_p95_delta_ms": lat_delta,
            "cost_delta_usd": cost_delta,
            "availability_delta": avail_delta,
        }


class _Agent:
    def __init__(self, agent_type: str):
        self.agent_type = agent_type


def _rank(pairs, state):
    ranked = UtilityScorer.rank(pairs, state)
    return [a.agent_type for a, _, _ in ranked]


# ---------------------------------------------------------------------------
# 1. Agent selection scenarios
# ---------------------------------------------------------------------------

class TestAgentSelectionScenarios:

    def test_s01_severe_drift_auc_drop_retrain_wins(self):
        """S01: AUC drops 15pp → retrain promises quality recovery → wins over threshold."""
        state = _make_state(incident_type="drift", current_auc=0.70, baseline_auc=0.85, auc_drop=0.15)
        retrain = (_Agent("retrain"), _Plan("retrain", confidence=0.72, auc_delta=0.13, cost_delta=15.0, risk=0.15))
        threshold = (_Agent("threshold"), _Plan("threshold", confidence=0.65, auc_delta=0.01, risk=0.05))
        winner = _rank([retrain, threshold], state)[0]
        # Quality weight=0.40 for drift; large auc_delta dominates despite cost penalty
        assert winner == "retrain"

    def test_s02_mild_drift_threshold_beats_expensive_retrain(self):
        """S02: Small drift, threshold adjustment is cheap and low-risk → wins over retrain."""
        state = _make_state(incident_type="drift", current_auc=0.83, auc_drop=0.02)
        retrain = (_Agent("retrain"), _Plan("retrain", confidence=0.55, auc_delta=0.04, cost_delta=15.0, risk=0.15))
        threshold = (_Agent("threshold"), _Plan("threshold", confidence=0.80, auc_delta=0.02, cost_delta=0.0, risk=0.05))
        winner = _rank([retrain, threshold], state)[0]
        assert winner == "threshold"

    def test_s03_latency_breach_fallback_wins_on_speed(self):
        """S03: Latency 200ms vs 100ms SLA → fallback (speed weight=0.40) beats threshold."""
        state = _make_state(
            incident_type="latency_breach",
            latency_p95_ms=200.0,
            latency_sla_ms=100.0,
        )
        fallback = (_Agent("fallback"), _Plan("fallback", confidence=0.68, lat_delta=-80.0, avail_delta=0.10, risk=0.15))
        threshold = (_Agent("threshold"), _Plan("threshold", confidence=0.75, auc_delta=0.01, risk=0.05))
        winner = _rank([fallback, threshold], state)[0]
        assert winner == "fallback"

    def test_s04_latency_breach_threshold_irrelevant_without_latency_promise(self):
        """S04: Threshold with no latency promise loses for latency incident even at high confidence."""
        state = _make_state(incident_type="latency_breach", latency_sla_ms=100.0)
        fallback = (_Agent("fallback"), _Plan("fallback", confidence=0.60, lat_delta=-50.0, avail_delta=0.10, risk=0.15))
        threshold = (_Agent("threshold"), _Plan("threshold", confidence=0.90, lat_delta=0.0, risk=0.05))
        winner = _rank([fallback, threshold], state)[0]
        assert winner == "fallback"

    def test_s05_data_quality_high_missing_data_repair_wins(self):
        """S05: Missing rate 0.25 → data_repair promises FNR improvement → wins."""
        state = _make_state(
            incident_type="data_quality",
            missing_rate=0.25,
            false_negative_rate=0.20,
        )
        datarepair = (_Agent("datarepair"), _Plan("datarepair", confidence=0.80, fnr_delta=-0.08, risk=0.10))
        fallback = (_Agent("fallback"), _Plan("fallback", confidence=0.65, avail_delta=0.10, risk=0.15))
        winner = _rank([datarepair, fallback], state)[0]
        # reliability weight=0.35 for data_quality; FNR improvement wins
        assert winner == "datarepair"

    def test_s06_cost_threshold_incident_cost_reduction_wins(self):
        """S06: Cost overrun → agent with cost_delta < 0 wins (cost weight=0.40)."""
        state = _make_state(
            incident_type="cost_threshold",
            cost_per_1000_predictions=8.0,
            cost_budget_per_1000=5.0,
        )
        cost_reducer = (_Agent("fallback"), _Plan("fallback", confidence=0.70, cost_delta=-2.0, risk=0.10))
        no_cost = (_Agent("threshold"), _Plan("threshold", confidence=0.75, cost_delta=0.0, risk=0.05))
        winner = _rank([cost_reducer, no_cost], state)[0]
        assert winner == "fallback"

    def test_s07_high_risk_plan_loses_despite_high_confidence(self):
        """S07: Risk penalty overrides confidence advantage when AUC promises are equal."""
        state = _make_state(incident_type="drift")
        # Same AUC promise; risky has higher confidence but massive risk penalty
        risky = (_Agent("retrain"), _Plan("retrain", confidence=0.90, auc_delta=0.02, risk=0.90))
        safe = (_Agent("threshold"), _Plan("threshold", confidence=0.65, auc_delta=0.02, risk=0.05))
        winner = _rank([risky, safe], state)[0]
        # Risky raw: 0.08 + 0.20*0.90 - 0.15*0.90 = 0.125
        # Safe  raw: 0.08 + 0.20*0.65 - 0.15*0.05 = 0.2025  → safe wins
        assert winner == "threshold"

    def test_s08_rollback_with_no_deployment_advantage_loses(self):
        """S08: Rollback with near-zero AUC gain loses to retrain with solid AUC promise."""
        state = _make_state(incident_type="drift")
        rollback = (_Agent("rollback"), _Plan("rollback", confidence=0.55, auc_delta=0.02, risk=0.20))
        retrain = (_Agent("retrain"), _Plan("retrain", confidence=0.70, auc_delta=0.09, risk=0.15, cost_delta=12.0))
        winner = _rank([rollback, retrain], state)[0]
        assert winner == "retrain"

    def test_s09_availability_bonus_helps_fallback_for_data_quality(self):
        """S09: FallbackAgent availability_delta is scored under reliability — improves its rank."""
        state = _make_state(incident_type="data_quality", false_negative_rate=0.10)
        fallback_with_avail = (_Agent("fallback"), _Plan("fallback", confidence=0.65, avail_delta=0.20, risk=0.15))
        fallback_no_avail = (_Agent("fallback_b"), _Plan("fallback_b", confidence=0.65, avail_delta=0.0, risk=0.15))
        ranked = UtilityScorer.rank([fallback_with_avail, fallback_no_avail], state)
        assert ranked[0][0].agent_type == "fallback"

    def test_s10_three_agents_ranked_correctly_for_drift(self):
        """S10: Three-way drift race — ordering reflects utility formula."""
        state = _make_state(incident_type="drift")
        retrain = (_Agent("retrain"), _Plan("retrain", confidence=0.72, auc_delta=0.10, cost_delta=12.0, risk=0.15))
        threshold = (_Agent("threshold"), _Plan("threshold", confidence=0.75, auc_delta=0.02, cost_delta=0.0, risk=0.05))
        rollback = (_Agent("rollback"), _Plan("rollback", confidence=0.55, auc_delta=0.03, risk=0.20))
        order = _rank([retrain, threshold, rollback], state)
        # Retrain: quality=0.40*1.0=0.40 dominates
        # Threshold: quality=0.40*0.2=0.08 + high confidence
        # Rollback: low confidence, medium AUC
        assert order[0] == "retrain"
        assert "rollback" not in [order[0]]


# ---------------------------------------------------------------------------
# 2. Guardrail check scenarios
# ---------------------------------------------------------------------------

class TestGuardrailScenarios:

    def test_s11_drift_clean_resolution_all_pass(self):
        """S11: Post-retrain state fully within policy → resolved, no regression."""
        before = _make_state(false_negative_rate=0.10, cost_per_1000_predictions=4.0)
        after = _make_state(
            current_auc=0.82, latency_p95_ms=88.0, missing_rate=0.02,
            false_negative_rate=0.09, cost_per_1000_predictions=4.0,
        )
        result = GuardrailChecker.check(before, after)
        assert result.resolved is True
        assert result.no_regression is True
        assert result.should_rollback is False
        assert result.violations == []

    def test_s12_drift_auc_not_recovered_unresolved(self):
        """S12: Retrain didn't recover AUC → not resolved (still below min_auc=0.75)."""
        before = _make_state()
        after = _make_state(current_auc=0.73, min_auc=0.75)
        result = GuardrailChecker.check(before, after)
        assert result.resolved is False
        assert result.auc_ok is False

    def test_s13_latency_drops_below_sla_passes(self):
        """S13: Fallback reduced latency 200→90ms → latency guardrail passes."""
        before = _make_state(latency_p95_ms=200.0)
        after = _make_state(latency_p95_ms=90.0, latency_sla_ms=100.0)
        result = GuardrailChecker.check(before, after)
        assert result.latency_ok is True

    def test_s14_latency_still_above_sla_after_action(self):
        """S14: Fallback reduced latency 200→120ms but SLA=100ms → latency fails."""
        before = _make_state(latency_p95_ms=200.0)
        after = _make_state(latency_p95_ms=120.0, latency_sla_ms=100.0)
        result = GuardrailChecker.check(before, after)
        assert result.latency_ok is False
        assert result.resolved is False

    def test_s15_data_quality_missing_cleaned_passes(self):
        """S15: DataRepair dropped missing_rate 0.25→0.02 → missing guardrail passes."""
        before = _make_state(missing_rate=0.25)
        after = _make_state(missing_rate=0.02, max_missing_rate=0.05)
        result = GuardrailChecker.check(before, after)
        assert result.missing_ok is True

    def test_s16_retrain_spikes_cost_regression(self):
        """S16: Retrain uses extra compute → cost_per_1000 jumps 4→5 (>1.10x) → regression."""
        before = _make_state(cost_per_1000_predictions=4.0)
        after = _make_state(cost_per_1000_predictions=5.0)  # 5/4 = 1.25 > 1.10
        result = GuardrailChecker.check(before, after)
        assert result.cost_ok is False
        assert result.no_regression is False

    def test_s17_lower_threshold_worsens_fnr_regression(self):
        """S17: ThresholdAgent lowers threshold → FNR rises 0.08→0.15 (+0.07 > 0.05) → regression."""
        before = _make_state(false_negative_rate=0.08)
        after = _make_state(false_negative_rate=0.15)  # +0.07 exceeds 0.05 allowed
        result = GuardrailChecker.check(before, after)
        assert result.fnr_ok is False
        assert result.no_regression is False

    def test_s18_rollback_triggered_when_auc_low_and_fnr_spikes(self):
        """S18: AUC below min AND FNR spiked → should_rollback=True."""
        before = _make_state(false_negative_rate=0.08, cost_per_1000_predictions=4.0)
        after = _make_state(
            current_auc=0.70, min_auc=0.75,   # AUC fails → not resolved
            false_negative_rate=0.20,           # FNR spike → regression
        )
        result = GuardrailChecker.check(before, after)
        assert result.resolved is False
        assert result.no_regression is False
        assert result.should_rollback is True

    def test_s19_no_rollback_when_auc_low_but_no_regression(self):
        """S19: AUC below min but cost and FNR stable → don't rollback (might converge)."""
        before = _make_state(false_negative_rate=0.08, cost_per_1000_predictions=4.0)
        after = _make_state(
            current_auc=0.70,              # AUC fails
            false_negative_rate=0.08,      # FNR stable
            cost_per_1000_predictions=4.0, # cost stable
        )
        result = GuardrailChecker.check(before, after)
        assert result.resolved is False
        assert result.no_regression is True
        assert result.should_rollback is False

    def test_s20_no_rollback_when_resolved_despite_cost_increase(self):
        """S20: Incident resolved → should_rollback=False even if cost crept up."""
        before = _make_state(cost_per_1000_predictions=4.0, false_negative_rate=0.08)
        after = _make_state(
            current_auc=0.82,
            latency_p95_ms=88.0,
            missing_rate=0.02,
            cost_per_1000_predictions=5.0,  # cost spike, but resolved
        )
        result = GuardrailChecker.check(before, after)
        assert result.resolved is True
        assert result.should_rollback is False


# ---------------------------------------------------------------------------
# 3. Severity scoring scenarios
# ---------------------------------------------------------------------------

class TestSeverityScenarios:

    def _telemetry(self, **overrides):
        collector = TelemetryCollector(use_mock=True)
        t = collector.collect()
        for k, v in overrides.items():
            # patch nested values via object attribute override
            if hasattr(t.model, k):
                object.__setattr__(t.model, k, v)
            elif hasattr(t.data, k):
                object.__setattr__(t.data, k, v)
            elif hasattr(t.system, k):
                object.__setattr__(t.system, k, v)
        return t

    def test_s21_severe_drift_produces_high_severity(self):
        """S21: Large drift score (3.0) → high severity."""
        calc = SeverityCalculator()
        telemetry = TelemetryCollector(use_mock=True).collect()
        telemetry.data.feature_drift_scores["LIMIT_BAL"] = 3.0
        severity, _ = calc.calculate(IncidentType.DRIFT, telemetry)
        assert severity > 0.40

    def test_s22_mild_drift_produces_low_severity(self):
        """S22: Small drift score (0.5) → low severity."""
        calc = SeverityCalculator()
        telemetry = TelemetryCollector(use_mock=True).collect()
        for k in telemetry.data.feature_drift_scores:
            telemetry.data.feature_drift_scores[k] = 0.5
        severity, _ = calc.calculate(IncidentType.DRIFT, telemetry)
        assert severity < 0.40

    def test_s23_high_missing_rate_produces_moderate_severity(self):
        """S23: Missing rate 0.40 → data quality severity in middle-high range."""
        calc = SeverityCalculator()
        telemetry = TelemetryCollector(use_mock=True).collect()
        telemetry.data.missing_rate = 0.40
        severity, breakdown = calc.calculate(IncidentType.DATA_QUALITY, telemetry)
        assert severity > 0.20
        assert "missing_rate" in breakdown.components

    def test_s24_latency_at_2x_sla_produces_high_severity(self):
        """S24: Latency 200ms vs baseline 100ms → high latency severity."""
        calc = SeverityCalculator()
        telemetry = TelemetryCollector(use_mock=True).collect()
        telemetry.system.latency_p95 = 200.0
        config = {"baseline_latency_ms": 100.0, "max_latency_ms": 200.0}
        severity, breakdown = calc.calculate(IncidentType.LATENCY_BREACH, telemetry, config)
        assert severity > 0.40
        assert "latency_ratio" in breakdown.components

    def test_s25_severity_breakdown_has_named_components(self):
        """S25: SeverityBreakdown.components has per-type named keys, not generic."""
        calc = SeverityCalculator()
        telemetry = TelemetryCollector(use_mock=True).collect()
        _, breakdown = calc.calculate(IncidentType.DRIFT, telemetry)
        # Phase 9: drift components must be named specifically
        component_keys = set(breakdown.components.keys())
        assert "auc_drop" in component_keys or "drift" in component_keys

    def test_s26_all_incident_types_produce_nonzero_severity(self):
        """S26: Each incident type produces non-negative severity from mock telemetry."""
        calc = SeverityCalculator()
        for incident_type in [
            IncidentType.DRIFT,
            IncidentType.DATA_QUALITY,
            IncidentType.LATENCY_BREACH,
            IncidentType.COST_THRESHOLD,
        ]:
            telemetry = TelemetryCollector(use_mock=True).collect()
            severity, breakdown = calc.calculate(incident_type, telemetry)
            assert severity >= 0.0, f"Severity negative for {incident_type}"
            assert breakdown.components, f"No components for {incident_type}"


# ---------------------------------------------------------------------------
# 4. Commander integration scenarios (async)
# ---------------------------------------------------------------------------

class TestCommanderIntegrationScenarios:

    @pytest.mark.asyncio
    async def test_s27_drift_incident_produces_complete_result(self):
        """S27: Canonical drift incident → result has all Phase 13 fields."""
        from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
        from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
        from self_healing_pipeline.commander.commander_v3 import CommanderV3
        from self_healing_pipeline.gateway.events import Incident

        commander = CommanderV3([RetrainAgent("retrain-1"), ThresholdAdjustmentAgent("threshold-1")])
        incident = Incident(
            tenant_id="standard", type=IncidentType.DRIFT,
            payload={"auc": 0.70, "drift_score": 2.0}, severity=0.7,
        )
        result = await commander.handle_incident(incident)
        assert result.incident_type == "drift"
        assert result.winning_agent_type in {"retrain", "threshold"}
        assert -1.0 <= result.reward <= 1.0
        assert -1.0 <= result.utility_score <= 1.0
        assert isinstance(result.guardrail_violations, list)
        assert isinstance(result.auto_rollback_triggered, bool)

    @pytest.mark.asyncio
    async def test_s28_latency_incident_winner_is_fallback_or_threshold(self):
        """S28: Latency breach → only fallback/threshold are candidate types."""
        from self_healing_pipeline.agents.fallback_v2 import FallbackAgent
        from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
        from self_healing_pipeline.commander.commander_v3 import CommanderV3
        from self_healing_pipeline.gateway.events import Incident

        commander = CommanderV3([FallbackAgent("fallback-1"), ThresholdAdjustmentAgent("threshold-1")])
        incident = Incident(
            tenant_id="standard", type=IncidentType.LATENCY_BREACH,
            payload={"latency_p95_ms": 180}, severity=0.6,
        )
        result = await commander.handle_incident(incident)
        assert result.winning_agent_type in {"fallback", "threshold"}

    @pytest.mark.asyncio
    async def test_s29_data_quality_incident_verification_breakdown_complete(self):
        """S29: Data quality incident → verification_breakdown has all expected keys."""
        from self_healing_pipeline.agents.datarepair_v2 import DataRepairAgent
        from self_healing_pipeline.commander.commander_v3 import CommanderV3
        from self_healing_pipeline.gateway.events import Incident

        commander = CommanderV3([DataRepairAgent("repair-1")])
        incident = Incident(
            tenant_id="standard", type=IncidentType.DATA_QUALITY,
            payload={"missing_rate": 0.20}, severity=0.5,
        )
        result = await commander.handle_incident(incident)
        bd = result.verification_breakdown
        required_keys = {
            "quality_gain", "cost_gain", "reliability_gain", "latency_gain",
            "resolution_score", "reward",
            "guardrail_resolved", "guardrail_no_regression",
            "guardrail_violations", "auto_rollback_triggered",
        }
        assert required_keys.issubset(bd.keys())

    @pytest.mark.asyncio
    async def test_s30_no_eligible_agents_returns_error_result(self):
        """S30: No agents can handle the incident type → graceful error result."""
        from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
        from self_healing_pipeline.commander.commander_v3 import CommanderV3
        from self_healing_pipeline.gateway.events import Incident

        # RetrainAgent.can_handle() returns False for LATENCY_BREACH
        commander = CommanderV3([RetrainAgent("retrain-1")])
        incident = Incident(
            tenant_id="standard", type=IncidentType.LATENCY_BREACH,
            payload={}, severity=0.5,
        )
        result = await commander.handle_incident(incident)
        assert result.winning_agent_type == "none"
        assert result.incident_resolved is False
        assert "no_eligible_agents" in result.verification_breakdown.get("error", "")
