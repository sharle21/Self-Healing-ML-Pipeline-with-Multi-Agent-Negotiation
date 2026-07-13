"""Rollback Model Remediation Policy: revert to previous known-good version."""

from __future__ import annotations

from typing import Any

from self_healing_pipeline.agents.remediation_policy import RemediationPlan, RemediationPolicyAgent


class RollbackAgent(RemediationPolicyAgent):
    """Revert to previous model version when recent deployment caused issues.

    Cares about: recent deployment, AUC regression, rollback history.

    Confidence = 0.40*deployment_recency + 0.30*auc_regression + 0.20*previous_health + 0.10*historical_success
    """

    agent_type = "rollback"

    def can_handle(self, state: dict[str, Any]) -> bool:
        """Can handle if recent deployment + AUC regression."""
        deployment_hours = state.get("deployment_age_hours", 24)
        auc_regression = state.get("current_auc", 0.68) < state.get("previous_auc", 0.77)

        return deployment_hours < 24 and auc_regression

    async def analyze(self, state: dict[str, Any]) -> RemediationPlan:
        """Analyze state and propose rollback.

        Args:
            state: RollbackAgentState dict

        Returns:
            RemediationPlan with state-based confidence
        """
        current_auc = state.get("current_auc", 0.68)
        previous_auc = state.get("previous_auc", 0.77)
        deployment_hours = state.get("deployment_age_hours", 6)
        current_error = state.get("current_error_rate", 0.18)
        previous_error = state.get("previous_error_rate", 0.09)
        incident_prob = state.get("deployment_related_incident_probability", 0.8)
        historical_success = state.get("historical_rollback_success", 0.91)

        # Compute confidence from state
        auc_regression = max(0, previous_auc - current_auc)
        error_regression = max(0, current_error - previous_error)

        state_features = {
            "deployment_recency": max(0, 1.0 - (deployment_hours / 24)),  # Very recent = high
            "auc_regression": min(auc_regression / 0.20, 1.0),  # Bigger regression = stronger
            "previous_health": previous_auc,  # Previous version quality
            "error_regression": min(error_regression / 0.20, 1.0),
            "historical_success": historical_success,
        }

        weights = {
            "deployment_recency": 0.40,
            "auc_regression": 0.25,
            "previous_health": 0.10,
            "error_regression": 0.10,
            "historical_success": 0.15,
        }

        confidence = self._compute_confidence_from_state(state_features, weights)

        return RemediationPlan(
            agent_type=self.agent_type,
            action="rollback",
            confidence=confidence,
            expected_effect={
                "auc_recovery": auc_regression * 0.9,  # Recover most regression
                "error_rate_recovery": error_regression * 0.95,
                "latency_change": 0,
            },
            reasoning=(
                f"Recent deployment ({deployment_hours:.1f}h) + AUC regression ({auc_regression:.3f}) + "
                f"error rate increase ({error_regression:.3f}) + incident probability ({incident_prob:.2f}) → "
                f"rollback to previous version (v12, AUC={previous_auc:.2f})"
            ),
            cost="$2",
            execution_time="15 seconds",
            risk=0.05,
        )

    async def execute(self, plan: RemediationPlan) -> Any:
        """Execute rollback (simulated)."""
        import asyncio

        await asyncio.sleep(0.01)

        from self_healing_pipeline.agents.remediation_policy import ExecutionResult

        return ExecutionResult(
            success=True,
            actual_improvement={
                "auc_recovery": 0.07,
                "error_rate_recovery": 0.08,
                "latency_change": 0,
            },
            duration=15.0,
            logs=[f"rollback completed: {plan.reasoning}"],
        )
