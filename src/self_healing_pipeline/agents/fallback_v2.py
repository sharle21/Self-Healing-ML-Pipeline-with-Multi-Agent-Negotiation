"""Fallback Model Remediation Policy: maintain availability with degraded ML."""

from __future__ import annotations

from typing import Any

from self_healing_pipeline.agents.remediation_policy import RemediationPlan, RemediationPolicyAgent


class FallbackAgent(RemediationPolicyAgent):
    """Switch to simpler model or rule-based fallback when ML unreliable.

    Emergency policy for keeping system available with reduced capability.

    Cares about: error rate, latency, prediction failure rate, data quality, fallback quality.

    Confidence = 0.30*error_rate + 0.25*latency_failure + 0.20*data_quality_failure + 0.15*fallback_quality + 0.10*historical_success
    """

    agent_type = "fallback"

    def can_handle(self, state: dict[str, Any]) -> bool:
        """Can handle if model unreliable or latency critical."""
        error_rate = state.get("error_rate", 0.12)
        latency = state.get("latency_p95", 85)
        missing_rate = state.get("missing_rate", 0.08)

        return error_rate > 0.15 or latency > 500 or missing_rate > 0.30

    async def analyze(self, state: dict[str, Any]) -> RemediationPlan:
        """Analyze state and propose fallback activation.

        Args:
            state: FallbackAgentState dict

        Returns:
            RemediationPlan with state-based confidence
        """
        error_rate = state.get("error_rate", 0.12)
        latency_p95 = state.get("latency_p95", 85)
        pred_failure = state.get("prediction_failure_rate", 0.15)
        missing_rate = state.get("missing_rate", 0.08)
        confidence_mean = state.get("confidence_distribution_mean", 0.62)
        fallback_quality = state.get("fallback_quality", 0.70)
        historical_success = state.get("historical_fallback_success", 0.85)

        # Phase 10: latency ratio (vs assumed 100ms SLA) drives urgency
        latency_sla_ms = 100.0
        latency_ratio_norm = min(latency_p95 / latency_sla_ms - 1.0, 1.0) if latency_p95 > latency_sla_ms else 0.0

        state_features = {
            "error_severity": min(error_rate / 0.25, 1.0),
            "latency_breach_ratio": max(0.0, latency_ratio_norm),
            "data_quality_issue": min(missing_rate / 0.50, 1.0),
            "model_confidence_low": 1.0 - min(confidence_mean, 1.0),
            "fallback_availability": fallback_quality,
            "historical_success": historical_success,
        }
        weights = {
            "error_severity": 0.25,
            "latency_breach_ratio": 0.25,
            "data_quality_issue": 0.20,
            "model_confidence_low": 0.15,
            "fallback_availability": 0.10,
            "historical_success": 0.05,
        }
        confidence = self._compute_confidence_from_state(state_features, weights)

        accuracy_loss_pct = (1.0 - fallback_quality) * 100
        return RemediationPlan(
            agent_type=self.agent_type,
            action="activate_fallback",
            confidence=confidence,
            expected_effect={
                "auc_delta": -(1.0 - fallback_quality) * 0.10,
                "latency_p95_delta_ms": -max(0.0, latency_p95 - latency_sla_ms),
                "cost_delta_usd": 0.0,
                "availability_delta": 0.15,
            },
            reasoning=(
                f"error={error_rate:.2f} latency={latency_p95:.0f}ms "
                f"({latency_p95/latency_sla_ms:.1f}x SLA) missing={missing_rate:.2f} "
                f"→ fallback (quality={fallback_quality:.2f}, -{accuracy_loss_pct:.0f}% accuracy)"
            ),
            cost="$0.10",
            execution_time="2 seconds",
            risk=0.15,
        )

    async def execute(self, plan: RemediationPlan) -> Any:
        """Execute fallback activation (simulated)."""
        import asyncio

        await asyncio.sleep(0.01)

        from self_healing_pipeline.agents.remediation_policy import ExecutionResult

        return ExecutionResult(
            success=True,
            actual_improvement={
                "availability": "+18%",
                "accuracy_change": "-3%",
                "latency_change": -80,
            },
            duration=2.0,
            logs=[f"fallback activated: {plan.reasoning}"],
        )
