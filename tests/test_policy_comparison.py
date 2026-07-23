"""Policy comparison: adaptive commander vs. always-retrain, fixed-priority, highest-confidence.

12 canonical incident scenarios × 4 selection policies = 48 trial results.
Each scenario defines concrete before/after IncidentState pairs per agent type so
reward and guardrail outcomes are deterministic regardless of mock-telemetry noise.

Eligibility constraints (from actual can_handle implementations):
  Threshold  — ALWAYS eligible (fn_cost=500 > 100 hardcoded)
  Retrain    — auc_drop > 0.05 OR drift > 1.0,  AND missing_rate < 0.50
  Rollback   — last_training_age_days < 1.0 (hours < 24) AND current_auc < baseline_auc
  Fallback   — error_rate > 0.15 OR latency_p95_ms > 500 OR missing_rate > 0.30
  DataRepair — missing > 0.15 OR duplicate > 0.05 OR schema_rate*500 > 10
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from self_healing_pipeline.agents.datarepair_v2 import DataRepairAgent
from self_healing_pipeline.agents.fallback_v2 import FallbackAgent
from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
from self_healing_pipeline.agents.rollback_v2 import RollbackAgent
from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
from self_healing_pipeline.commander.utility import UtilityScorer
from self_healing_pipeline.gateway.events import IncidentType
from self_healing_pipeline.observability import IncidentState, StateConstructor
from self_healing_pipeline.verification.guardrails import GuardrailChecker
from self_healing_pipeline.verification.reward import RewardCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _ExecResult:
    success: bool = True
    duration: float = 2.0


_INC_TYPE_MAP = {
    "drift":         IncidentType.DRIFT,
    "latency_breach": IncidentType.LATENCY_BREACH,
    "data_quality":  IncidentType.DATA_QUALITY,
    "cost_threshold": IncidentType.COST_THRESHOLD,
}

_AGENT_POOL = [
    RetrainAgent("retrain-1"),
    ThresholdAdjustmentAgent("threshold-1"),
    RollbackAgent("rollback-1"),
    FallbackAgent("fallback-1"),
    DataRepairAgent("datarepair-1"),
]

_SC = StateConstructor()

_FIXED_PRIORITY = ["threshold", "retrain", "rollback", "fallback", "data_repair"]


def _state(incident_type: str = "drift", **overrides) -> IncidentState:
    defaults = dict(
        tenant_id="standard",
        incident_type=incident_type,
        current_auc=0.80,
        baseline_auc=0.85,
        current_precision=0.80,
        baseline_precision=0.85,
        current_recall=0.78,
        baseline_recall=0.83,
        auc_drop=0.05,
        false_positive_rate=0.05,
        false_negative_rate=0.10,
        max_feature_drift=0.8,
        drifted_features=[],
        missing_rate=0.03,
        duplicate_rate=0.01,
        schema_violation_rate=0.0,
        latency_p95_ms=88.0,
        latency_p99_ms=130.0,
        latency_sla_ms=100.0,
        cost_per_1000_predictions=4.0,
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
        severity_components={"auc_drop": 0.3, "drift": 0.3, "affected_volume": 0.2},
    )
    defaults["incident_type"] = incident_type
    defaults.update(overrides)
    return IncidentState(**defaults)


def _after(before: IncidentState, **overrides) -> IncidentState:
    """Post-action state: start from before, apply only the changed fields."""
    d = {f: getattr(before, f) for f in before.__dataclass_fields__}
    d.update(overrides)
    return IncidentState(**d)


def _agent_state_dict(s: IncidentState) -> dict:
    """Merge all relevant agent state dicts for the incident type (mirrors commander_v3)."""
    inc = _INC_TYPE_MAP[s.incident_type]
    if inc == IncidentType.DRIFT:
        state = _SC.retrain_state_from_incident(s).to_dict()
        state.update(_SC.threshold_state_from_incident(s).to_dict())
        state.update(_SC.rollback_state_from_incident(s).to_dict())
    elif inc == IncidentType.DATA_QUALITY:
        state = _SC.datarepair_state_from_incident(s).to_dict()
        state.update(_SC.fallback_state_from_incident(s).to_dict())
    elif inc == IncidentType.LATENCY_BREACH:
        state = _SC.threshold_state_from_incident(s).to_dict()
        state.update(_SC.fallback_state_from_incident(s).to_dict())
    elif inc == IncidentType.COST_THRESHOLD:
        state = _SC.threshold_state_from_incident(s).to_dict()
        state.update(_SC.fallback_state_from_incident(s).to_dict())
    else:
        state = {}
    state["tenant_id"] = s.tenant_id
    return state


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    description: str
    incident_type: str
    state_before: IncidentState
    # per agent_type: post-action IncidentState; missing keys default to state_before
    outcomes: dict[str, IncidentState]


def _build_scenarios() -> list[Scenario]:
    # -------------------------------------------------------------------------
    # DRIFT scenarios (1-3, 9, 11)
    # -------------------------------------------------------------------------

    # S1: Severe drift — retrain is only full recovery
    # Eligible: threshold (always), retrain (drop=0.20>0.05), fallback (error=0.21>0.15)
    b1 = _state("drift", current_auc=0.65, baseline_auc=0.85, auc_drop=0.20,
                max_feature_drift=2.8, drifted_features=["LIMIT_BAL", "PAY_0", "BILL_AMT1"],
                last_training_age_days=30.0, false_positive_rate=0.05, false_negative_rate=0.16,
                current_recall=0.65, baseline_recall=0.83, missing_rate=0.03)
    s1 = Scenario("severe_drift", "AUC −20pp, model 30d old; retrain is only full fix", "drift",
                  b1, {
                      "retrain":   _after(b1, current_auc=0.83, auc_drop=0.02, max_feature_drift=0.8,
                                         false_negative_rate=0.09, cost_per_1000_predictions=5.5),
                      "threshold": _after(b1, current_auc=0.67, auc_drop=0.18, false_negative_rate=0.14),
                      "fallback":  _after(b1, current_auc=0.60, auc_drop=0.25, false_negative_rate=0.20),
                  })

    # S2: Mild drift — threshold sufficient; retrain wasteful but resolves
    # Eligible: threshold (always), retrain (drop=0.08), fallback (error=0.16>0.15)
    b2 = _state("drift", current_auc=0.77, baseline_auc=0.85, auc_drop=0.08,
                max_feature_drift=1.3, last_training_age_days=5.0,
                false_positive_rate=0.05, false_negative_rate=0.11,
                current_recall=0.76, baseline_recall=0.83, missing_rate=0.03)
    s2 = Scenario("mild_drift", "AUC −8pp, model 5d old; threshold sufficient, retrain wastes $19", "drift",
                  b2, {
                      "retrain":   _after(b2, current_auc=0.80, auc_drop=0.05, false_negative_rate=0.09,
                                         cost_per_1000_predictions=5.8),
                      "threshold": _after(b2, current_auc=0.79, auc_drop=0.06, false_negative_rate=0.10),
                      "fallback":  _after(b2, current_auc=0.75, auc_drop=0.10, false_negative_rate=0.12),
                  })

    # S3: Recent deployment regression — rollback fastest fix
    # Eligible: threshold (always), retrain (drop=0.16), rollback (0.3d<1d AND AUC<baseline),
    #           fallback (error=0.20>0.15)
    b3 = _state("drift", current_auc=0.69, baseline_auc=0.85, auc_drop=0.16,
                max_feature_drift=0.9, last_training_age_days=0.3,
                false_positive_rate=0.05, false_negative_rate=0.15,
                current_recall=0.70, baseline_recall=0.83, missing_rate=0.03)
    s3 = Scenario("deployment_regression", "New deploy broke AUC; rollback to v2 is fastest fix", "drift",
                  b3, {
                      "retrain":   _after(b3, current_auc=0.74, auc_drop=0.11, false_negative_rate=0.12,
                                         cost_per_1000_predictions=5.5),
                      "threshold": _after(b3, current_auc=0.71, auc_drop=0.14, false_negative_rate=0.14),
                      "rollback":  _after(b3, current_auc=0.80, auc_drop=0.05, max_feature_drift=0.5,
                                         false_negative_rate=0.09),
                      "fallback":  _after(b3, current_auc=0.65, auc_drop=0.20, false_negative_rate=0.18),
                  })

    # S9: Drift + corrupted data — no agent fully resolves both dimensions
    # Eligible: threshold (always), retrain (drop=0.15, quality=0.88≥0.50),
    #           fallback (error=0.20>0.15)
    b9 = _state("drift", current_auc=0.70, baseline_auc=0.85, auc_drop=0.15,
                max_feature_drift=1.9, last_training_age_days=20.0,
                false_positive_rate=0.05, false_negative_rate=0.15,
                current_recall=0.72, baseline_recall=0.83, missing_rate=0.12,
                max_missing_rate=0.05)
    s9 = Scenario("drift_with_data_issues",
                  "Drift + 12% missing rate; no single agent fully resolves both", "drift",
                  b9, {
                      "retrain":   _after(b9, current_auc=0.73, auc_drop=0.12, max_feature_drift=1.0,
                                         false_negative_rate=0.12, cost_per_1000_predictions=5.5),
                      "threshold": _after(b9, current_auc=0.71, auc_drop=0.14, false_negative_rate=0.14),
                      "fallback":  _after(b9, current_auc=0.65, auc_drop=0.20, false_negative_rate=0.18),
                  })

    # S11: Near-threshold drift — AUC=0.78 already above min; retrain wasteful
    # Eligible: threshold (always), retrain (drop=0.07>0.05 AND drift=1.2>1.0)
    # Fallback NOT eligible: error=0.05+0.10=0.15 NOT >0.15
    b11 = _state("drift", current_auc=0.78, baseline_auc=0.85, auc_drop=0.07,
                 max_feature_drift=1.2, last_training_age_days=3.0,
                 false_positive_rate=0.05, false_negative_rate=0.10,
                 current_recall=0.76, baseline_recall=0.83, missing_rate=0.03)
    s11 = Scenario("near_threshold_drift",
                   "AUC 0.78 above min 0.75; threshold works, retrain adds $16 cost", "drift",
                   b11, {
                       "retrain":   _after(b11, current_auc=0.81, auc_drop=0.04, false_negative_rate=0.08,
                                          cost_per_1000_predictions=5.8),
                       "threshold": _after(b11, current_auc=0.79, auc_drop=0.06, false_negative_rate=0.09),
                   })

    # -------------------------------------------------------------------------
    # LATENCY scenarios (4, 5, 10)
    # -------------------------------------------------------------------------

    # S4: Mild latency breach — fallback wins on speed weight
    # Eligible: threshold (always), fallback (error=0.05+0.12=0.17>0.15)
    # Retrain NOT eligible: auc_drop=0.05 NOT >0.05, drift=0.8 NOT >1.0
    b4 = _state("latency_breach", latency_p95_ms=130.0, latency_sla_ms=100.0,
                current_auc=0.80, baseline_auc=0.85, auc_drop=0.05,
                max_feature_drift=0.8,
                false_positive_rate=0.06, false_negative_rate=0.12,
                current_recall=0.78, baseline_recall=0.83, missing_rate=0.03)
    s4 = Scenario("latency_mild", "Latency 130ms vs 100ms SLA; fallback wins (speed weight 0.40)", "latency_breach",
                  b4, {
                      "fallback":  _after(b4, latency_p95_ms=72.0, false_negative_rate=0.12),
                      "threshold": _after(b4, latency_p95_ms=118.0, false_negative_rate=0.11),
                  })

    # S5: Severe latency breach — fallback eligible via latency>500
    # Eligible: threshold (always), fallback (latency=520>500)
    b5 = _state("latency_breach", latency_p95_ms=520.0, latency_sla_ms=100.0,
                current_auc=0.80, baseline_auc=0.85, auc_drop=0.05,
                max_feature_drift=0.8,
                false_positive_rate=0.05, false_negative_rate=0.10,
                current_recall=0.78, baseline_recall=0.83, missing_rate=0.03)
    s5 = Scenario("latency_severe", "Latency 520ms vs 100ms SLA; only fallback can resolve it", "latency_breach",
                  b5, {
                      "fallback":  _after(b5, latency_p95_ms=68.0, false_negative_rate=0.12),
                      "threshold": _after(b5, latency_p95_ms=495.0, false_negative_rate=0.10),
                  })

    # S10: Post-upgrade latency — model fine, pure infrastructure issue
    # Eligible: threshold (always), fallback (latency=520>500)
    # Retrain NOT eligible: auc_drop=0.03<0.05, drift=0.6<1.0
    b10 = _state("latency_breach", latency_p95_ms=520.0, latency_sla_ms=100.0,
                 current_auc=0.82, baseline_auc=0.85, auc_drop=0.03,
                 max_feature_drift=0.6,
                 false_positive_rate=0.05, false_negative_rate=0.10,
                 current_recall=0.80, baseline_recall=0.83, missing_rate=0.03)
    s10 = Scenario("post_upgrade_latency",
                   "Good model, infra upgrade caused latency spike; fallback needed", "latency_breach",
                   b10, {
                       "fallback":  _after(b10, latency_p95_ms=75.0, false_negative_rate=0.12),
                       "threshold": _after(b10, latency_p95_ms=500.0, false_negative_rate=0.10),
                   })

    # -------------------------------------------------------------------------
    # DATA QUALITY scenarios (6, 7)
    # -------------------------------------------------------------------------

    # S6: High missing rate — only datarepair fixes it
    # Eligible: threshold (always), fallback (missing=0.32>0.30), datarepair (missing=0.32>0.15)
    # Retrain NOT eligible: auc_drop=0.04<0.05, drift=0.8<1.0
    b6 = _state("data_quality", missing_rate=0.32, max_missing_rate=0.05,
                current_auc=0.80, baseline_auc=0.85, auc_drop=0.04,
                max_feature_drift=0.8, false_positive_rate=0.05, false_negative_rate=0.10,
                current_recall=0.78, baseline_recall=0.83)
    s6 = Scenario("high_missing_rate", "32% rows missing; datarepair restores quality, threshold/fallback don't",
                  "data_quality", b6, {
                      "data_repair": _after(b6, missing_rate=0.01, false_negative_rate=0.09),
                      "fallback":   _after(b6, missing_rate=0.32, false_negative_rate=0.10),
                      "threshold":  _after(b6, missing_rate=0.32, false_negative_rate=0.09),
                  })

    # S7: Schema violations — datarepair fixes schema; fallback/threshold can't
    # Eligible: threshold (always), fallback (error=0.19>0.15), datarepair (schema_count=90>10)
    b7 = _state("data_quality", schema_violation_rate=0.18, missing_rate=0.03,
                max_missing_rate=0.05, current_auc=0.80, baseline_auc=0.85,
                auc_drop=0.04, max_feature_drift=0.8,
                false_positive_rate=0.05, false_negative_rate=0.14,
                current_recall=0.78, baseline_recall=0.83)
    s7 = Scenario("schema_violations", "18% schema violations; datarepair fixes pipeline, others can't",
                  "data_quality", b7, {
                      "data_repair": _after(b7, schema_violation_rate=0.001, missing_rate=0.02,
                                          false_negative_rate=0.09),
                      "fallback":   _after(b7, schema_violation_rate=0.18, false_negative_rate=0.14),
                      "threshold":  _after(b7, schema_violation_rate=0.18, false_negative_rate=0.12),
                  })

    # -------------------------------------------------------------------------
    # COST scenarios (8, 12)
    # -------------------------------------------------------------------------

    # S8: Cost overrun — fallback reduces load, retrain makes it worse
    # Eligible: threshold (always), fallback (error=0.17>0.15)
    # Retrain NOT eligible: auc_drop=0.04<0.05, drift=0.8<1.0
    b8 = _state("cost_threshold", cost_per_1000_predictions=8.5, cost_budget_per_1000=3.0,
                current_auc=0.80, baseline_auc=0.85, auc_drop=0.04,
                max_feature_drift=0.8,
                false_positive_rate=0.05, false_negative_rate=0.12,
                current_recall=0.78, baseline_recall=0.83, missing_rate=0.03)
    s8 = Scenario("cost_overrun",
                  "Cost 8.5 vs budget 3.0; fallback drops cost to 2.5, threshold leaves it at 7.0",
                  "cost_threshold", b8, {
                      "fallback":  _after(b8, cost_per_1000_predictions=2.5),
                      "threshold": _after(b8, cost_per_1000_predictions=7.0),
                  })

    # S12: Cost + quality conflict — retrain eligible but makes cost worse
    # Eligible: threshold (always), retrain (drop=0.13>0.05), fallback (error=0.17>0.15)
    b12 = _state("cost_threshold", cost_per_1000_predictions=6.0, cost_budget_per_1000=3.0,
                 current_auc=0.72, baseline_auc=0.85, auc_drop=0.13,
                 max_feature_drift=0.8, false_positive_rate=0.06, false_negative_rate=0.11,
                 current_recall=0.70, baseline_recall=0.83, missing_rate=0.03)
    s12 = Scenario("cost_quality_conflict",
                   "Cost 2x over budget AND AUC below min; retrain fixes quality but 3x cost",
                   "cost_threshold", b12, {
                       "retrain":   _after(b12, current_auc=0.83, auc_drop=0.02,
                                          cost_per_1000_predictions=17.0),
                       "threshold": _after(b12, current_auc=0.74, auc_drop=0.11,
                                          cost_per_1000_predictions=5.0),
                       "fallback":  _after(b12, current_auc=0.70, cost_per_1000_predictions=2.8),
                   })

    return [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]


# ---------------------------------------------------------------------------
# Policy selection functions
# ---------------------------------------------------------------------------

def _select(policy: str, plans: list[tuple], state: IncidentState) -> str | None:
    """Return selected agent_type string for the given policy."""
    if not plans:
        return None

    if policy == "adaptive_commander":
        ranked = UtilityScorer.rank(plans, state)
        return ranked[0][0].agent_type if ranked else None

    if policy == "always_retrain":
        for agent, _ in plans:
            if agent.agent_type == "retrain":
                return "retrain"
        # Retrain not eligible — fall back to highest confidence
        return max(plans, key=lambda x: x[1].confidence)[0].agent_type

    if policy == "fixed_priority":
        agent_map = {a.agent_type: a for a, _ in plans}
        for t in _FIXED_PRIORITY:
            if t in agent_map:
                return t
        return None

    if policy == "highest_confidence":
        return max(plans, key=lambda x: x[1].confidence)[0].agent_type

    if policy == "cheapest_eligible":
        def _dollar_cost(plan) -> float:
            try:
                return float(plan.cost.lstrip("$"))
            except ValueError:
                return float("inf")
        return min(plans, key=lambda x: _dollar_cost(x[1]))[0].agent_type

    raise ValueError(f"Unknown policy: {policy}")


# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    scenario: str
    incident_type: str
    policy: str
    selected_agent: str
    resolved: bool
    reward: float
    guardrail_violations: list[str]
    no_regression: bool
    unnecessary_retrain: bool  # retrain selected when incident NOT drift


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------

POLICIES = ["adaptive_commander", "always_retrain", "fixed_priority", "highest_confidence", "cheapest_eligible"]


async def _run_trial(scenario: Scenario, policy: str) -> TrialResult:
    state_dict = _agent_state_dict(scenario.state_before)

    plans = []
    for agent in _AGENT_POOL:
        if agent.can_handle(state_dict):
            plan = await agent.analyze(state_dict)
            plans.append((agent, plan))

    selected_type = _select(policy, plans, scenario.state_before) or "none"

    # Look up outcome for the selected agent; default = no change
    state_after = scenario.outcomes.get(selected_type, scenario.state_before)

    exec_result = _ExecResult(success=True, duration=2.0)
    reward, _ = RewardCalculator.calculate_from_incident_states(
        scenario.state_before, state_after, selected_type, exec_result
    )
    gr = GuardrailChecker.check(scenario.state_before, state_after)

    return TrialResult(
        scenario=scenario.name,
        incident_type=scenario.incident_type,
        policy=policy,
        selected_agent=selected_type,
        resolved=gr.resolved,
        reward=reward,
        guardrail_violations=gr.violations,
        no_regression=gr.no_regression,
        unnecessary_retrain=(
            selected_type == "retrain" and scenario.incident_type != "drift"
        ),
    )


def run_comparison() -> list[TrialResult]:
    """Run all scenarios × all policies synchronously. Returns list of TrialResult."""
    scenarios = _build_scenarios()
    results = []
    for scenario in scenarios:
        for policy in POLICIES:
            results.append(asyncio.run(_run_trial(scenario, policy)))
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class PolicyMetrics:
    policy: str
    resolution_rate: float
    mean_reward: float
    unnecessary_retrain_rate: float
    guardrail_violation_rate: float
    n_trials: int


def aggregate(results: list[TrialResult]) -> dict[str, PolicyMetrics]:
    by_policy: dict[str, list[TrialResult]] = {}
    for r in results:
        by_policy.setdefault(r.policy, []).append(r)

    metrics = {}
    for policy, trials in by_policy.items():
        n = len(trials)
        metrics[policy] = PolicyMetrics(
            policy=policy,
            resolution_rate=round(sum(t.resolved for t in trials) / n, 3),
            mean_reward=round(sum(t.reward for t in trials) / n, 3),
            unnecessary_retrain_rate=round(sum(t.unnecessary_retrain for t in trials) / n, 3),
            guardrail_violation_rate=round(sum(bool(t.guardrail_violations) for t in trials) / n, 3),
            n_trials=n,
        )
    return metrics


def print_comparison_table(metrics: dict[str, PolicyMetrics]) -> None:
    policy_order = ["adaptive_commander", "always_retrain", "fixed_priority", "highest_confidence", "cheapest_eligible"]
    header = f"{'Policy':<22} {'Resolution':>10} {'Mean Reward':>12} {'Unnec Retrain':>14} {'Guardrail Viol':>15}"
    print(header)
    print("-" * len(header))
    for p in policy_order:
        m = metrics[p]
        print(
            f"{p:<22} {m.resolution_rate:>9.0%} {m.mean_reward:>12.3f} "
            f"{m.unnecessary_retrain_rate:>13.0%} {m.guardrail_violation_rate:>14.0%}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPolicyComparison:

    @pytest.fixture(scope="class")
    def metrics(self) -> dict[str, PolicyMetrics]:
        results = run_comparison()
        return aggregate(results)

    def test_adaptive_beats_always_retrain_on_resolution(self, metrics):
        assert metrics["adaptive_commander"].resolution_rate > metrics["always_retrain"].resolution_rate, (
            f"Adaptive {metrics['adaptive_commander'].resolution_rate:.0%} should beat "
            f"always-retrain {metrics['always_retrain'].resolution_rate:.0%}"
        )

    def test_adaptive_beats_fixed_priority_on_reward(self, metrics):
        assert metrics["adaptive_commander"].mean_reward > metrics["fixed_priority"].mean_reward, (
            f"Adaptive reward {metrics['adaptive_commander'].mean_reward:.3f} should beat "
            f"fixed-priority {metrics['fixed_priority'].mean_reward:.3f}"
        )

    def test_adaptive_beats_highest_confidence_on_reward(self, metrics):
        assert metrics["adaptive_commander"].mean_reward >= metrics["highest_confidence"].mean_reward, (
            f"Adaptive {metrics['adaptive_commander'].mean_reward:.3f} >= "
            f"highest-conf {metrics['highest_confidence'].mean_reward:.3f}"
        )

    def test_always_retrain_has_highest_unnecessary_retrain_rate(self, metrics):
        ar_rate = metrics["always_retrain"].unnecessary_retrain_rate
        for other_policy in ["adaptive_commander", "fixed_priority", "highest_confidence"]:
            assert ar_rate >= metrics[other_policy].unnecessary_retrain_rate, (
                f"always_retrain ({ar_rate:.0%}) should have >= unnecessary retrains than {other_policy}"
            )

    def test_adaptive_resolution_rate_at_least_60pct(self, metrics):
        assert metrics["adaptive_commander"].resolution_rate >= 0.60

    def test_adaptive_mean_reward_positive(self, metrics):
        assert metrics["adaptive_commander"].mean_reward > 0.0

    def test_all_policies_run_same_number_of_trials(self, metrics):
        counts = {p: m.n_trials for p, m in metrics.items()}
        assert len(set(counts.values())) == 1, f"Unequal trial counts: {counts}"

    def test_results_printed(self, metrics, capsys):
        print_comparison_table(metrics)
        captured = capsys.readouterr()
        assert "adaptive_commander" in captured.out
        assert "Resolution" in captured.out
