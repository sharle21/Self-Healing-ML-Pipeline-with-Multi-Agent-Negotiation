"""Rollback Model Remediation Policy: revert to previous known-good version."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

from self_healing_pipeline.agents.remediation_policy import ExecutionResult, RemediationPlan, RemediationPolicyAgent

logger = logging.getLogger(__name__)


class RollbackAgent(RemediationPolicyAgent):
    """Revert to previous model version when recent deployment caused issues.

    Cares about: recent deployment, AUC regression, rollback history.

    Confidence = 0.40*deployment_recency + 0.30*auc_regression + 0.20*previous_health + 0.10*historical_success

    When model_path is provided, execute() restores the backup model on disk
    and reloads the serving API.
    """

    agent_type = "rollback"

    def __init__(
        self,
        agent_id: str,
        model_path: Path | None = None,
        session_factory: Any | None = None,
        api_url: str = "http://localhost:8000",
    ) -> None:
        super().__init__(agent_id)
        self._model_path = model_path
        self._session_factory = session_factory
        self._api_url = api_url

    def can_handle(self, state: dict[str, Any]) -> bool:
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

        # Phase 10: confidence weighted by deployment_probability (low when old deployment)
        auc_regression = max(0, previous_auc - current_auc)
        error_regression = max(0, current_error - previous_error)
        current_model = state.get("current_model", "unknown")
        previous_model = state.get("previous_model", "unknown")

        state_features = {
            "deployment_prob": incident_prob,            # primary signal: deployment-related?
            "auc_regression": min(auc_regression / 0.20, 1.0),
            "previous_health": previous_auc,
            "error_regression": min(error_regression / 0.20, 1.0),
            "historical_success": historical_success,
        }
        weights = {
            "deployment_prob": 0.40,   # Phase 10: high weight — only roll back if likely deployment-caused
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
                "auc_delta": auc_regression * 0.90,
                "false_negative_rate_delta": -error_regression * 0.95,
                "latency_p95_delta_ms": 0,
                "cost_delta_usd": 0.0,
            },
            reasoning=(
                f"deployment_age={deployment_hours:.1f}h deployment_prob={incident_prob:.2f} "
                f"auc_regression={auc_regression:.3f} error_regression={error_regression:.3f} "
                f"→ rollback {current_model} → {previous_model} (AUC={previous_auc:.2f})"
            ),
            cost="$2",
            execution_time="15 seconds",
            risk=0.05,
        )

    async def execute(self, plan: RemediationPlan) -> ExecutionResult:
        t0 = time.time()

        if self._model_path is None:
            return ExecutionResult(
                success=True,
                actual_improvement={"auc_recovery": 0.07, "error_rate_recovery": 0.08},
                duration=time.time() - t0,
                logs=[f"[simulated] {plan.reasoning}"],
            )

        backup = self._model_path.with_suffix(".backup.joblib")
        if not backup.exists():
            return ExecutionResult(
                success=False,
                actual_improvement={},
                duration=time.time() - t0,
                error=f"no backup found at {backup}; cannot rollback",
            )

        try:
            shutil.copy2(backup, self._model_path)
            logger.info("rollback: restored %s from %s", self._model_path, backup)
        except Exception as exc:
            return ExecutionResult(
                success=False,
                actual_improvement={},
                duration=time.time() - t0,
                error=str(exc),
            )

        self._reload_api()

        return ExecutionResult(
            success=True,
            actual_improvement={"restored_from": str(backup)},
            duration=time.time() - t0,
            logs=[plan.reasoning],
        )

    def _reload_api(self) -> None:
        try:
            import httpx
            r = httpx.post(f"{self._api_url}/internal/reload-model", timeout=10)
            r.raise_for_status()
            logger.info("API model reloaded after rollback")
        except Exception as exc:
            logger.warning("API reload failed: %s", exc)
