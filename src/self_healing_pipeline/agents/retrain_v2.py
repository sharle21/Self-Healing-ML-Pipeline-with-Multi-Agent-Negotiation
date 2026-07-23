"""Retrain Model Remediation Policy: refit model on recent data to address distribution shift."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

from self_healing_pipeline.agents.remediation_policy import ExecutionResult, RemediationPlan, RemediationPolicyAgent

logger = logging.getLogger(__name__)


class RetrainAgent(RemediationPolicyAgent):
    """Refit model on recent data to address drift and degradation.

    Cares about: drift magnitude, AUC drop, data freshness (model age), historical retrain success.

    Confidence = 0.35*drift_score + 0.25*auc_degradation + 0.25*model_staleness + 0.15*historical_success

    When model_path is provided, execute() runs real LightGBM training, saves the
    model to disk, and posts /internal/reload-model to the serving API.
    """

    agent_type = "retrain"

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
        drift = state.get("drift_score", 0)
        auc_drop = abs(state.get("auc_drop", 0))
        data_quality = state.get("data_quality_score", 1.0)
        # Don't retrain when data is severely corrupted (quality < 0.50)
        return (drift > 1.0 or auc_drop > 0.05) and data_quality >= 0.50

    async def analyze(self, state: dict[str, Any]) -> RemediationPlan:
        drift_score = state.get("drift_score", 0)
        auc_drop = abs(state.get("auc_drop", 0.08))
        data_quality = state.get("data_quality_score", 0.92)
        model_age = state.get("model_age_days", 30)
        historical_success = state.get("historical_retrain_success", 0.72)
        affected_features = state.get("affected_features", [])

        # Phase 10: higher confidence when data is fresh + drift is large + model is stale
        # Lower confidence when data quality is borderline (0.50-0.70 range)
        data_quality_signal = max(0.0, (data_quality - 0.50) / 0.50)  # 0 at quality=0.50, 1 at quality=1.0
        state_features = {
            "drift_magnitude": min(drift_score / 3.0, 1.0),
            "auc_degradation": min(auc_drop / 0.15, 1.0),   # 15% drop = certain retrain needed
            "model_staleness": min(model_age / 60.0, 1.0),
            "data_freshness": data_quality_signal,
            "historical_success": historical_success,
        }
        weights = {
            "drift_magnitude": 0.35,
            "auc_degradation": 0.25,
            "model_staleness": 0.10,
            "data_freshness": 0.15,
            "historical_success": 0.15,
        }
        confidence = self._compute_confidence_from_state(state_features, weights)

        # Estimated training cost: proportional to model age + drift magnitude
        estimated_training_cost_usd = 10.0 + model_age * 0.50 + drift_score * 5.0

        return RemediationPlan(
            agent_type=self.agent_type,
            action="retrain_model",
            confidence=confidence,
            expected_effect={
                "auc_delta": min(auc_drop * 0.80, 0.15),
                "false_negative_rate_delta": -auc_drop * 0.50,
                "latency_p95_delta_ms": 0,
                "cost_delta_usd": 0.0,
                "estimated_training_cost_usd": round(estimated_training_cost_usd, 2),
            },
            reasoning=(
                f"drift={drift_score:.2f}σ auc_drop={auc_drop:.3f} "
                f"model_age={model_age}d data_quality={data_quality:.2f} "
                f"→ retrain (affected: {', '.join(affected_features) or 'none'})"
            ),
            cost=f"${estimated_training_cost_usd:.0f}",
            execution_time="~30 seconds",
            risk=0.15,
        )

    async def execute(self, plan: RemediationPlan) -> ExecutionResult:
        t0 = time.time()

        if self._model_path is None:
            # Simulated path (no model_path configured — e.g. unit tests)
            return ExecutionResult(
                success=True,
                actual_improvement={"auc_recovery": 0.07, "drift_reduction": 1.2},
                duration=time.time() - t0,
                logs=[f"[simulated] {plan.reasoning}"],
            )

        try:
            new_auc = await self._real_retrain()
        except Exception as exc:
            logger.error("retrain failed: %s", exc)
            return ExecutionResult(
                success=False,
                actual_improvement={},
                duration=time.time() - t0,
                error=str(exc),
            )

        duration = time.time() - t0
        logger.info("retrain complete: auc=%.4f duration=%.1fs", new_auc, duration)

        # Write ModelValidationReport to DB
        if self._session_factory is not None:
            try:
                self._write_validation_report(new_auc)
            except Exception as exc:
                logger.warning("validation report write failed: %s", exc)

        # Reload the serving model
        self._reload_api()

        return ExecutionResult(
            success=True,
            actual_improvement={"overall_auc": new_auc},
            duration=duration,
            logs=[plan.reasoning, f"new_model_path={self._model_path}"],
        )

    async def _real_retrain(self) -> float:
        """Run real LightGBM retraining. Returns new overall AUC."""
        import asyncio
        import numpy as np

        from self_healing_pipeline.pipeline.loader import fetch_uci_credit_default, split_by_tenant
        from self_healing_pipeline.pipeline.trainer import persist_model, train_model

        logger.info("fetching training data")
        df = await asyncio.get_event_loop().run_in_executor(
            None, fetch_uci_credit_default
        )
        df = split_by_tenant(df, rng=np.random.default_rng(42))

        # Back up current model before overwriting
        if self._model_path and self._model_path.exists():
            backup = self._model_path.with_suffix(".backup.joblib")
            shutil.copy2(self._model_path, backup)
            logger.info("backup saved to %s", backup)

        logger.info("training LightGBM on %d rows", len(df))
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: train_model(df, random_state=42)
        )
        persist_model(result, self._model_path)
        return result.overall_auc

    def _write_validation_report(self, auc: float) -> None:
        from datetime import UTC, datetime
        from self_healing_pipeline.db.models import ModelRegistry, ModelValidationReport

        version = f"retrain-{int(time.time())}"
        with self._session_factory() as session:
            # One report per tenant (shared model)
            for tenant_id in ("standard", "enterprise", "free"):
                session.add(ModelValidationReport(
                    model_version=version,
                    tenant_id=tenant_id,
                    auc=auc,
                    precision=0.0,  # not computed per-tenant here
                    recall=0.0,
                    f1_score=0.0,
                    optimal_threshold=0.5,
                    calibration_error=0.0,
                    validated_at=datetime.now(UTC),
                ))
            session.add(ModelRegistry(
                model_version=version,
                artifact_path=str(self._model_path),
                backup_path=str(self._model_path.with_suffix(".backup.joblib")) if self._model_path else None,
                overall_auc=auc,
                status="active",
            ))
            session.commit()

    def _reload_api(self) -> None:
        try:
            import httpx
            r = httpx.post(f"{self._api_url}/internal/reload-model", timeout=10)
            r.raise_for_status()
            logger.info("API model reloaded via %s", self._api_url)
        except Exception as exc:
            logger.warning("API reload failed (API may not be running): %s", exc)
