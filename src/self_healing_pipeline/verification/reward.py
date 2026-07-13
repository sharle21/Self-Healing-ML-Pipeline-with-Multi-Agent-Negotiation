"""Reward calculation: measure actual improvement after agent execution."""

from __future__ import annotations

from dataclasses import dataclass

from self_healing_pipeline.observability.telemetry import Telemetry


@dataclass(slots=True)
class RewardBreakdown:
    """Component breakdown of reward calculation."""

    metric_improvement: float
    incident_resolution: float
    cost_efficiency: float
    risk_penalty: float
    reward: float


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
