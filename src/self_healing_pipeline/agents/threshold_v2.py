"""Threshold Adjustment Remediation Policy: adjust decision boundaries for business tradeoffs."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from self_healing_pipeline.agents.remediation_policy import ExecutionResult, RemediationPlan, RemediationPolicyAgent

logger = logging.getLogger(__name__)

# Search space for threshold candidates
_THRESHOLD_CANDIDATES = np.linspace(0.20, 0.80, 61)
# Linear sensitivity: delta-threshold → delta-FPR/FNR scales
# Calibrated so a 0.40 change in threshold brings FPR→0 or FNR→2x
_SENSITIVITY = 0.40


class ThresholdAdjustmentAgent(RemediationPolicyAgent):
    """Adjust decision threshold to minimize expected business cost.

    Phase 10: searches 61 candidate thresholds using a linear FPR/FNR cost model
    calibrated from real incident state (false_positive_rate, false_negative_rate,
    cost_false_positive, cost_false_negative).

    When session_factory is provided, execute() writes the new threshold to
    TenantThresholdOverride and ModelServer reads it on the next request.
    """

    agent_type = "threshold"

    def __init__(self, agent_id: str, session_factory: Any | None = None) -> None:
        super().__init__(agent_id)
        self._session_factory = session_factory

    def can_handle(self, state: dict[str, Any]) -> bool:
        recall_drop = abs(state.get("recall_drop", 0))
        fn_cost = state.get("cost_false_negative", 0)
        return recall_drop > 0.05 or fn_cost > 100

    async def analyze(self, state: dict[str, Any]) -> RemediationPlan:
        current_threshold = state.get("current_threshold", 0.50)
        tenant_id = state.get("tenant_id", "unknown")
        current_fpr = max(0.001, state.get("false_positive_rate", 0.05))
        current_fnr = max(0.001, state.get("false_negative_rate", 0.10))
        cost_fp = state.get("cost_false_positive", 20.0)
        cost_fn = state.get("cost_false_negative", 500.0)
        recall_drop = abs(state.get("recall_drop", 0))
        historical_success = state.get("historical_threshold_success", 0.75)

        # --- Confidence ---
        state_features = {
            "recall_health": 1.0 - min(recall_drop / 0.30, 1.0),
            "fn_cost_impact": min(cost_fn / 1000.0, 1.0),
            "fpr_stability": 1.0 - min(current_fpr / 0.20, 1.0),
            "historical_success": historical_success,
        }
        weights = {
            "recall_health": 0.35,
            "fn_cost_impact": 0.30,
            "fpr_stability": 0.20,
            "historical_success": 0.15,
        }
        confidence = self._compute_confidence_from_state(state_features, weights)

        # --- Threshold search (Phase 10) ---
        # Linear model: FPR decreases, FNR increases as threshold rises.
        # At delta=+_SENSITIVITY from current_threshold, FPR→0; FNR doubles.
        best_threshold = current_threshold
        best_cost = current_fpr * cost_fp + current_fnr * cost_fn

        for t in _THRESHOLD_CANDIDATES:
            delta = t - current_threshold
            fpr_t = max(0.001, current_fpr * (1.0 - delta / _SENSITIVITY))
            fnr_t = max(0.001, current_fnr * (1.0 + delta / _SENSITIVITY))
            expected_cost = fpr_t * cost_fp + fnr_t * cost_fn
            if expected_cost < best_cost:
                best_cost = expected_cost
                best_threshold = round(float(t), 4)

        new_threshold = round(max(0.10, min(best_threshold, 0.90)), 4)
        delta_final = new_threshold - current_threshold
        new_fpr = max(0.001, current_fpr * (1.0 - delta_final / _SENSITIVITY))
        new_fnr = max(0.001, current_fnr * (1.0 + delta_final / _SENSITIVITY))

        return RemediationPlan(
            agent_type=self.agent_type,
            action="change_threshold",
            confidence=confidence,
            expected_effect={
                "new_threshold": new_threshold,
                "tenant_id": tenant_id,
                "false_negative_rate_delta": new_fnr - current_fnr,
                "false_positive_rate_delta": new_fpr - current_fpr,
                "estimated_cost_per_prediction": best_cost,
            },
            reasoning=(
                f"threshold search (61 candidates, cost_fp=${cost_fp}, cost_fn=${cost_fn}) → "
                f"{current_threshold:.4f} → {new_threshold:.4f} "
                f"[fpr {current_fpr:.3f}→{new_fpr:.3f}, fnr {current_fnr:.3f}→{new_fnr:.3f}]"
            ),
            cost="$0",
            execution_time="<1 second",
            risk=0.05,
        )

    async def execute(self, plan: RemediationPlan) -> ExecutionResult:
        t0 = time.time()
        new_threshold = plan.expected_effect.get("new_threshold")
        tenant_id = plan.expected_effect.get("tenant_id", "unknown")

        if new_threshold is None:
            return ExecutionResult(
                success=False,
                actual_improvement={},
                duration=time.time() - t0,
                error="new_threshold missing from plan",
            )

        if self._session_factory is not None:
            try:
                self._write_threshold(tenant_id, float(new_threshold))
                logger.info(
                    "threshold written: tenant=%s threshold=%.4f", tenant_id, new_threshold
                )
            except Exception as exc:
                logger.error("threshold write failed: %s", exc)
                return ExecutionResult(
                    success=False,
                    actual_improvement={},
                    duration=time.time() - t0,
                    error=str(exc),
                )

        return ExecutionResult(
            success=True,
            actual_improvement={"new_threshold": new_threshold, "tenant_id": tenant_id},
            duration=time.time() - t0,
            logs=[plan.reasoning],
        )

    def _write_threshold(self, tenant_id: str, threshold: float) -> None:
        from self_healing_pipeline.db.models import TenantThresholdOverride

        with self._session_factory() as session:
            override = session.get(TenantThresholdOverride, tenant_id)
            if override is None:
                override = TenantThresholdOverride(
                    tenant_id=tenant_id,
                    threshold=threshold,
                    updated_by=self.agent_id,
                )
                session.add(override)
            else:
                override.threshold = threshold
                override.updated_by = self.agent_id
            session.commit()
