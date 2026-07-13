"""State constructor: build agent-specific state dicts from telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from self_healing_pipeline.observability.telemetry import Telemetry


@dataclass(slots=True)
class ThresholdAgentState:
    """State for threshold adjustment remediation policy."""

    current_threshold: float
    precision_drop: float
    recall_drop: float
    false_positive_rate: float
    false_negative_rate: float
    cost_false_positive: float
    cost_false_negative: float
    latency: float
    historical_threshold_success: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_threshold": self.current_threshold,
            "precision_drop": self.precision_drop,
            "recall_drop": self.recall_drop,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "cost_false_positive": self.cost_false_positive,
            "cost_false_negative": self.cost_false_negative,
            "latency": self.latency,
            "historical_threshold_success": self.historical_threshold_success,
        }


@dataclass(slots=True)
class RetrainAgentState:
    """State for model retraining remediation policy."""

    drift_score: float
    auc_drop: float
    data_quality_score: float
    model_age_days: int
    historical_retrain_success: float
    affected_features: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_score": self.drift_score,
            "auc_drop": self.auc_drop,
            "data_quality_score": self.data_quality_score,
            "model_age_days": self.model_age_days,
            "historical_retrain_success": self.historical_retrain_success,
            "affected_features": self.affected_features,
        }


@dataclass(slots=True)
class RollbackAgentState:
    """State for model rollback remediation policy."""

    current_model: str
    previous_model: str
    deployment_age_hours: float
    current_auc: float
    previous_auc: float
    current_error_rate: float
    previous_error_rate: float
    deployment_related_incident_probability: float
    historical_rollback_success: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_model": self.current_model,
            "previous_model": self.previous_model,
            "deployment_age_hours": self.deployment_age_hours,
            "current_auc": self.current_auc,
            "previous_auc": self.previous_auc,
            "current_error_rate": self.current_error_rate,
            "previous_error_rate": self.previous_error_rate,
            "deployment_related_incident_probability": self.deployment_related_incident_probability,
            "historical_rollback_success": self.historical_rollback_success,
        }


@dataclass(slots=True)
class FallbackAgentState:
    """State for fallback logic remediation policy."""

    error_rate: float
    latency_p95: float
    prediction_failure_rate: float
    confidence_distribution_mean: float
    missing_rate: float
    acceptable_accuracy_loss: float
    fallback_quality: float
    historical_fallback_success: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_rate": self.error_rate,
            "latency_p95": self.latency_p95,
            "prediction_failure_rate": self.prediction_failure_rate,
            "confidence_distribution_mean": self.confidence_distribution_mean,
            "missing_rate": self.missing_rate,
            "acceptable_accuracy_loss": self.acceptable_accuracy_loss,
            "fallback_quality": self.fallback_quality,
            "historical_fallback_success": self.historical_fallback_success,
        }


@dataclass(slots=True)
class DataRepairAgentState:
    """State for data repair remediation policy."""

    missing_rate: float
    duplicate_rate: float
    schema_error_count: int
    affected_features: list[str]
    available_backup_data: bool
    data_pipeline_health: float
    historical_repair_success: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_rate": self.missing_rate,
            "duplicate_rate": self.duplicate_rate,
            "schema_error_count": self.schema_error_count,
            "affected_features": self.affected_features,
            "available_backup_data": self.available_backup_data,
            "data_pipeline_health": self.data_pipeline_health,
            "historical_repair_success": self.historical_repair_success,
        }


class StateConstructor:
    """Construct agent-specific state dicts from telemetry."""

    @staticmethod
    def threshold_state(
        telemetry: Telemetry,
        current_threshold: float = 0.50,
        cost_fp: float = 20,
        cost_fn: float = 500,
        historical_success: float = 0.75,
    ) -> ThresholdAgentState:
        """Build state for threshold adjustment agent."""
        return ThresholdAgentState(
            current_threshold=current_threshold,
            precision_drop=0.05,  # Assume baseline 87% precision → 82%
            recall_drop=0.12,  # Assume baseline 83% recall → 71%
            false_positive_rate=telemetry.model.error_rate * 0.75,
            false_negative_rate=telemetry.model.error_rate * 1.5,
            cost_false_positive=cost_fp,
            cost_false_negative=cost_fn,
            latency=telemetry.system.latency_p95,
            historical_threshold_success=historical_success,
        )

    @staticmethod
    def retrain_state(
        telemetry: Telemetry,
        model_age_days: int = 30,
        historical_success: float = 0.72,
    ) -> RetrainAgentState:
        """Build state for model retraining agent."""
        max_drift = max(telemetry.data.feature_drift_scores.values()) if telemetry.data.feature_drift_scores else 0.0
        affected_features = [
            feat for feat, score in telemetry.data.feature_drift_scores.items() if score > 1.0
        ]

        return RetrainAgentState(
            drift_score=max_drift,
            auc_drop=0.08,  # Assume 8% AUC drop
            data_quality_score=1.0 - telemetry.data.missing_rate,
            model_age_days=model_age_days,
            historical_retrain_success=historical_success,
            affected_features=affected_features,
        )

    @staticmethod
    def rollback_state(
        telemetry: Telemetry,
        current_model: str = "v13",
        previous_model: str = "v12",
        deployment_age_hours: float = 6,
        current_auc: float = 0.68,
        previous_auc: float = 0.77,
        historical_success: float = 0.91,
    ) -> RollbackAgentState:
        """Build state for rollback agent."""
        return RollbackAgentState(
            current_model=current_model,
            previous_model=previous_model,
            deployment_age_hours=deployment_age_hours,
            current_auc=current_auc,
            previous_auc=previous_auc,
            current_error_rate=telemetry.model.error_rate,
            previous_error_rate=0.09,  # Assume v12 was healthier
            deployment_related_incident_probability=0.8 if deployment_age_hours < 24 else 0.2,
            historical_rollback_success=historical_success,
        )

    @staticmethod
    def fallback_state(
        telemetry: Telemetry,
        acceptable_accuracy_loss: float = 0.05,
        fallback_quality: float = 0.70,
        historical_success: float = 0.85,
    ) -> FallbackAgentState:
        """Build state for fallback agent."""
        return FallbackAgentState(
            error_rate=telemetry.model.error_rate,
            latency_p95=telemetry.system.latency_p95,
            prediction_failure_rate=telemetry.data.missing_rate,
            confidence_distribution_mean=0.62,  # Assume average model confidence
            missing_rate=telemetry.data.missing_rate,
            acceptable_accuracy_loss=acceptable_accuracy_loss,
            fallback_quality=fallback_quality,
            historical_fallback_success=historical_success,
        )

    @staticmethod
    def datarepair_state(
        telemetry: Telemetry,
        available_backup: bool = True,
        pipeline_health: float = 0.55,
        historical_success: float = 0.70,
    ) -> DataRepairAgentState:
        """Build state for data repair agent."""
        # Identify affected features from drift
        affected = [
            feat for feat, score in telemetry.data.feature_drift_scores.items() if score > 1.5
        ]

        return DataRepairAgentState(
            missing_rate=telemetry.data.missing_rate,
            duplicate_rate=telemetry.data.duplicate_rate,
            schema_error_count=telemetry.data.schema_violations,
            affected_features=affected,
            available_backup_data=available_backup,
            data_pipeline_health=pipeline_health,
            historical_repair_success=historical_success,
        )
