"""Threshold Adjustment Remediation Policy: adjust decision boundaries for business tradeoffs."""

from __future__ import annotations

import logging
from typing import Any

from self_healing_pipeline.agents.remediation_policy import RemediationPlan, RemediationPolicyAgent

logger = logging.getLogger(__name__)


class ThresholdAdjustmentAgent(RemediationPolicyAgent):
    """Adjust decision threshold to optimize false positive/negative tradeoff.

    Cares about: recall degradation, false negative cost, prediction distribution shift,
    historical threshold success.

    Confidence = 0.35*recall_health + 0.30*fn_cost_impact + 0.20*distribution_shift + 0.15*historical_success
    """

    agent_type = "threshold"

    def can_handle(self, state: dict[str, Any]) -> bool:
        """Can handle if recall is degraded or FN cost is high."""
        recall_drop = abs(state.get("recall_drop", 0))
        fn_cost = state.get("cost_false_negative", 0)

        return recall_drop > 0.05 or fn_cost > 100

    async def analyze(self, state: dict[str, Any]) -> RemediationPlan:
        """Analyze state and propose threshold adjustment.

        Args:
            state: ThresholdAgentState dict with all metrics

        Returns:
            RemediationPlan with confidence computed from state features
        """
        # Extract state
        current_threshold = state.get("current_threshold", 0.50)
        recall_drop = abs(state.get("recall_drop", 0))
        fn_cost = state.get("cost_false_negative", 500)
        distribution_shift = state.get("prediction_distribution_shift", 0.15)
        historical_success = state.get("historical_threshold_success", 0.75)

        # Compute confidence from state features
        state_features = {
            "recall_health": 1.0 - min(recall_drop / 0.30, 1.0),  # Higher = better recall
            "fn_cost_impact": min(fn_cost / 1000, 1.0),  # Higher cost = more important to fix
            "distribution_stability": 1.0 - min(distribution_shift / 0.50, 1.0),  # Stable = better
            "historical_success": historical_success,
        }

        weights = {
            "recall_health": 0.35,
            "fn_cost_impact": 0.30,
            "distribution_stability": 0.20,
            "historical_success": 0.15,
        }

        confidence = self._compute_confidence_from_state(state_features, weights)

        # Propose new threshold
        # Logic: if FN cost is high, lower threshold (more positives → lower FN)
        threshold_adjustment = 0.0
        if recall_drop > 0.10:
            threshold_adjustment = -0.05 if fn_cost > 300 else -0.03
        elif recall_drop > 0.05:
            threshold_adjustment = -0.02

        new_threshold = max(0.1, min(current_threshold + threshold_adjustment, 0.9))

        return RemediationPlan(
            agent_type=self.agent_type,
            action="change_threshold",
            confidence=confidence,
            expected_effect={
                "false_negative_rate": -recall_drop * 0.5,  # Recover some recall
                "false_positive_rate": +distribution_shift * 0.3,  # Trade-off: more FP
                "availability": "maintained",
            },
            reasoning=(
                f"Recall degradation ({recall_drop:.2f}) + high FN cost (${fn_cost}) → "
                f"shift threshold from {current_threshold:.2f} to {new_threshold:.2f} to recover recall"
            ),
            cost="$0",
            execution_time="5 seconds",
            risk=0.05,
        )

    async def execute(self, plan: RemediationPlan) -> Any:
        """Execute threshold adjustment (simulated)."""
        import asyncio

        await asyncio.sleep(0.01)

        from self_healing_pipeline.agents.remediation_policy import ExecutionResult

        return ExecutionResult(
            success=True,
            actual_improvement={
                "false_negative_rate": -0.08,
                "false_positive_rate": 0.03,
            },
            duration=5.0,
            logs=[f"threshold adjusted: {plan.reasoning}"],
        )
