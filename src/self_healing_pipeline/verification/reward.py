"""Reward calculation: measure actual improvement after agent execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from self_healing_pipeline.observability.telemetry import Telemetry

if TYPE_CHECKING:
    from self_healing_pipeline.observability.incident_state import IncidentState


@dataclass(slots=True)
class RewardBreakdown:
    """Component breakdown of reward calculation (legacy Telemetry-based)."""

    metric_improvement: float
    incident_resolution: float
    cost_efficiency: float
    risk_penalty: float
    reward: float


@dataclass(slots=True)
class OutcomeReward:
    """Component breakdown for IncidentState-based outcome reward (Phase 14).

    All component values are in [-1, +1]. Final reward is clipped to [-1, +1].
    Positive = improvement, negative = regression.
    """

    # Per-dimension gains (positive = improved)
    quality_gain: float       # AUC improvement relative to 10% ideal
    cost_gain: float          # cost reduction relative to 20% ideal
    reliability_gain: float   # missing-rate reduction relative to baseline
    latency_gain: float       # latency reduction relative to SLA

    # Resolution and penalties
    resolution_score: float   # 1.0 if incident resolved, 0.0 otherwise
    exec_cost_penalty: float  # agent-specific execution cost (0-1)
    time_penalty: float       # penalty for slow execution (0-0.10)
    regression_penalty: float # latency/cost got worse (0-1)

    # Final
    reward: float             # clipped to [-1, +1]

    # Context
    incident_type: str
    agent_type: str
    auc_before: float | None
    auc_after: float | None


# Per-incident-type reward weights (quality, cost, reliability, latency, resolution)
_INCIDENT_WEIGHTS: dict[str, tuple[float, float, float, float, float]] = {
    "drift":           (0.45, 0.10, 0.10, 0.10, 0.25),
    "data_quality":    (0.10, 0.10, 0.35, 0.10, 0.35),
    "latency_breach":  (0.05, 0.10, 0.10, 0.45, 0.30),
    "cost_threshold":  (0.10, 0.40, 0.10, 0.10, 0.30),
}
_DEFAULT_WEIGHTS: tuple[float, float, float, float, float] = (0.30, 0.15, 0.15, 0.15, 0.25)

# Execution cost penalties per agent type (fraction of reward subtracted)
_EXEC_COST: dict[str, float] = {
    "threshold":   0.00,
    "retrain":     0.10,
    "rollback":    0.05,
    "fallback":    0.05,
    "data_repair": 0.08,
}


class RewardCalculator:
    """Calculate reward from before/after telemetry."""

    @staticmethod
    def calculate_drift_reward(
        state_before: Telemetry,
        state_after: Telemetry,
        agent_type: str,
        incident_resolved: bool = False,
    ) -> tuple[float, RewardBreakdown]:
        """Calculate reward for drift incident remediation.

        Reward = 0.4*metric_improvement + 0.3*incident_resolution + 0.2*cost_efficiency - 0.1*risk

        Args:
            state_before: telemetry before agent execution
            state_after: telemetry after agent execution
            agent_type: which agent executed
            incident_resolved: whether drift was actually fixed

        Returns:
            (reward: -1 to +1, breakdown: component breakdown)
        """
        # Metric improvement: AUC recovery
        auc_gain = state_after.model.auc - state_before.model.auc
        auc_reward = min(auc_gain / 0.10, 1.0)  # 10% improvement = perfect

        # Incident resolution bonus
        resolution_bonus = 1.0 if incident_resolved else 0.0

        # Cost efficiency: did we stay within budget?
        cost_increase = state_after.system.cost_per_prediction - state_before.system.cost_per_prediction
        cost_reward = 1.0 if cost_increase <= 0 else max(0, 1.0 - (cost_increase / 0.01))

        # Risk penalty: agent-specific
        risk_penalty = {
            "threshold": 0.05,
            "retrain": 0.10,
            "rollback": 0.05,
            "fallback": 0.15,
            "data_repair": 0.12,
        }.get(agent_type, 0.10)

        reward = (
            0.4 * auc_reward + 0.3 * resolution_bonus + 0.2 * cost_reward - 0.1 * risk_penalty
        )

        return reward, RewardBreakdown(
            metric_improvement=auc_reward,
            incident_resolution=resolution_bonus,
            cost_efficiency=cost_reward,
            risk_penalty=risk_penalty,
            reward=reward,
        )

    @staticmethod
    def calculate_data_quality_reward(
        state_before: Telemetry,
        state_after: Telemetry,
        agent_type: str,
        incident_resolved: bool = False,
    ) -> tuple[float, RewardBreakdown]:
        """Calculate reward for data quality incident remediation.

        Args:
            state_before: telemetry before execution
            state_after: telemetry after execution
            agent_type: which agent executed
            incident_resolved: whether quality issues fixed

        Returns:
            (reward: -1 to +1, breakdown: component breakdown)
        """
        # Metric improvement: missing rate reduction
        missing_improvement = state_before.data.missing_rate - state_after.data.missing_rate
        missing_reward = min(missing_improvement / 0.25, 1.0)  # 25% reduction = good

        # Schema violation improvement
        schema_improvement = state_before.data.schema_violations - state_after.data.schema_violations
        schema_reward = min(schema_improvement / 50, 1.0)

        # Combined metric improvement
        metric_reward = (0.6 * missing_reward + 0.4 * schema_reward)

        # Incident resolution
        resolution_bonus = 1.0 if incident_resolved else 0.0

        # Cost efficiency
        cost_increase = state_after.system.cost_per_prediction - state_before.system.cost_per_prediction
        cost_reward = 1.0 if cost_increase <= 0 else max(0, 1.0 - (cost_increase / 0.01))

        # Risk penalty
        risk_penalty = {
            "threshold": 0.10,
            "retrain": 0.15,
            "rollback": 0.10,
            "fallback": 0.05,
            "data_repair": 0.20,
        }.get(agent_type, 0.12)

        reward = (
            0.4 * metric_reward + 0.3 * resolution_bonus + 0.2 * cost_reward - 0.1 * risk_penalty
        )

        return reward, RewardBreakdown(
            metric_improvement=metric_reward,
            incident_resolution=resolution_bonus,
            cost_efficiency=cost_reward,
            risk_penalty=risk_penalty,
            reward=reward,
        )

    @staticmethod
    def calculate_latency_reward(
        state_before: Telemetry,
        state_after: Telemetry,
        agent_type: str,
        incident_resolved: bool = False,
    ) -> tuple[float, RewardBreakdown]:
        """Calculate reward for latency incident remediation.

        Args:
            state_before: telemetry before execution
            state_after: telemetry after execution
            agent_type: which agent executed
            incident_resolved: whether latency fixed

        Returns:
            (reward: -1 to +1, breakdown: component breakdown)
        """
        # Metric improvement: latency reduction
        p95_improvement = state_before.system.latency_p95 - state_after.system.latency_p95
        p95_reward = min(p95_improvement / 100, 1.0)  # 100ms improvement = good

        p99_improvement = state_before.system.latency_p99 - state_after.system.latency_p99
        p99_reward = min(p99_improvement / 200, 1.0)

        metric_reward = 0.6 * p95_reward + 0.4 * p99_reward

        # Incident resolution
        resolution_bonus = 1.0 if incident_resolved else 0.0

        # Cost efficiency (latency fixes might use more compute)
        cost_increase = state_after.system.cost_per_prediction - state_before.system.cost_per_prediction
        cost_reward = 1.0 if cost_increase <= 0 else max(0, 1.0 - (cost_increase / 0.01))

        # Risk penalty
        risk_penalty = {
            "threshold": 0.05,
            "retrain": 0.08,
            "rollback": 0.07,
            "fallback": 0.15,
            "data_repair": 0.10,
        }.get(agent_type, 0.08)

        reward = (
            0.4 * metric_reward + 0.3 * resolution_bonus + 0.2 * cost_reward - 0.1 * risk_penalty
        )

        return reward, RewardBreakdown(
            metric_improvement=metric_reward,
            incident_resolution=resolution_bonus,
            cost_efficiency=cost_reward,
            risk_penalty=risk_penalty,
            reward=reward,
        )

    @staticmethod
    def calculate_cost_reward(
        state_before: Telemetry,
        state_after: Telemetry,
        agent_type: str,
        incident_resolved: bool = False,
    ) -> tuple[float, RewardBreakdown]:
        """Calculate reward for cost threshold incident remediation.

        Args:
            state_before: telemetry before execution
            state_after: telemetry after execution
            agent_type: which agent executed
            incident_resolved: whether cost threshold respected

        Returns:
            (reward: -1 to +1, breakdown: component breakdown)
        """
        # Metric improvement: cost reduction
        cost_reduction = state_before.system.cost_per_prediction - state_after.system.cost_per_prediction
        cost_reward = min(cost_reduction / 0.005, 1.0)  # $0.005 reduction = good

        # Incident resolution: cost back within budget
        resolution_bonus = 1.0 if incident_resolved else 0.0

        # Latency efficiency: did we reduce cost without blowing latency?
        latency_increase = state_after.system.latency_p95 - state_before.system.latency_p95
        latency_penalty = min(latency_increase / 50, 1.0)  # 50ms increase = penalty

        # Risk penalty
        risk_penalty = {
            "threshold": 0.05,
            "retrain": 0.15,
            "rollback": 0.08,
            "fallback": 0.10,
            "data_repair": 0.20,
        }.get(agent_type, 0.10)

        reward = (
            0.4 * cost_reward + 0.3 * resolution_bonus + 0.2 * (1.0 - latency_penalty) - 0.1 * risk_penalty
        )

        return reward, RewardBreakdown(
            metric_improvement=cost_reward,
            incident_resolution=resolution_bonus,
            cost_efficiency=1.0 - latency_penalty,
            risk_penalty=risk_penalty,
            reward=reward,
        )

    # ------------------------------------------------------------------
    # Phase 14: IncidentState-based outcome reward
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_from_incident_states(
        state_before: IncidentState,
        state_after: IncidentState,
        agent_type: str,
        execution_result: Any,
    ) -> tuple[float, OutcomeReward]:
        """Compute outcome-based reward from real before/after IncidentState snapshots.

        All deltas come from Prometheus-sourced IncidentState, not agent estimates.
        Reward is clipped to [-1, +1] and fully decomposed for explainability.

        Args:
            state_before: IncidentState snapshot taken before agent execution
            state_after:  IncidentState snapshot taken after stabilization window
            agent_type:   winning agent type (for exec cost lookup)
            execution_result: ExecutionResult (for duration + success flag)

        Returns:
            (reward, OutcomeReward breakdown)
        """
        inc_type = state_before.incident_type
        wq, wc, wr, wl, wres = _INCIDENT_WEIGHTS.get(inc_type, _DEFAULT_WEIGHTS)

        # --- AUC quality ---
        auc_b = state_before.current_auc if state_before.current_auc is not None else state_before.baseline_auc
        auc_a = state_after.current_auc if state_after.current_auc is not None else state_after.baseline_auc
        auc_delta = auc_a - auc_b
        quality_gain = _clip(auc_delta / 0.10, -1.0, 1.0)  # 10% AUC gain = perfect

        # --- Cost ---
        cost_b = state_before.cost_per_1000_predictions
        cost_a = state_after.cost_per_1000_predictions
        cost_ref = max(cost_b * 0.20, 0.10)  # 20% reduction = perfect
        cost_gain = _clip((cost_b - cost_a) / cost_ref, -1.0, 1.0)

        # --- Reliability (missing rate) ---
        miss_b = state_before.missing_rate
        miss_a = state_after.missing_rate
        miss_ref = max(miss_b, state_before.max_missing_rate, 0.01)
        reliability_gain = _clip((miss_b - miss_a) / miss_ref, -1.0, 1.0)

        # --- Latency ---
        lat_b = state_before.latency_p95_ms
        lat_a = state_after.latency_p95_ms
        lat_ref = max(state_before.latency_sla_ms, 50.0)
        latency_gain = _clip((lat_b - lat_a) / lat_ref, -1.0, 1.0)

        # --- Resolution ---
        resolved = RewardCalculator.check_resolved(state_after)
        resolution_score = 1.0 if resolved else 0.0

        # --- Penalties ---
        exec_cost_penalty = _EXEC_COST.get(agent_type, 0.08)
        duration = getattr(execution_result, "duration", 0.0)
        time_penalty = min(duration / 300.0, 1.0) * 0.10
        # Regression: latency or cost got significantly worse
        lat_regression = _clip((lat_a - lat_b) / max(lat_ref, 1.0), 0.0, 1.0)
        cost_regression = _clip((cost_a - cost_b) / max(cost_ref, 0.01), 0.0, 1.0)
        regression_penalty = max(lat_regression, cost_regression)

        raw = (
            wq * quality_gain
            + wc * cost_gain
            + wr * reliability_gain
            + wl * latency_gain
            + wres * resolution_score
            - exec_cost_penalty
            - time_penalty
            - 0.10 * regression_penalty
        )
        reward = _clip(raw, -1.0, 1.0)

        breakdown = OutcomeReward(
            quality_gain=quality_gain,
            cost_gain=cost_gain,
            reliability_gain=reliability_gain,
            latency_gain=latency_gain,
            resolution_score=resolution_score,
            exec_cost_penalty=exec_cost_penalty,
            time_penalty=time_penalty,
            regression_penalty=regression_penalty,
            reward=reward,
            incident_type=inc_type,
            agent_type=agent_type,
            auc_before=state_before.current_auc,
            auc_after=state_after.current_auc,
        )
        return reward, breakdown

    @staticmethod
    def check_resolved(state: IncidentState) -> bool:
        """Check if incident is resolved based on post-action IncidentState vs policy limits.

        Intentionally lenient: any major dimension being within limits counts as resolved.
        """
        inc_type = state.incident_type

        if inc_type == "drift":
            auc_ok = (state.current_auc is None) or (state.current_auc >= state.min_auc)
            drift_ok = state.max_feature_drift < 1.5
            return auc_ok and drift_ok

        if inc_type == "data_quality":
            return (
                state.missing_rate <= state.max_missing_rate
                and state.schema_violation_rate < 0.01
            )

        if inc_type == "latency_breach":
            return state.latency_p95_ms <= state.latency_sla_ms

        if inc_type == "cost_threshold":
            return state.cost_per_1000_predictions <= state.cost_budget_per_1000

        # Unknown type: unresolved
        return False


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
