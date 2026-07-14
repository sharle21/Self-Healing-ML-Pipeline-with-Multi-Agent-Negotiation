"""Tenant configuration from measurable sources (validation metrics, deployment profiles)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class ValidationMetrics:
    """Model validation results from training pipeline."""

    auc: float
    optimal_threshold: float
    precision: float
    recall: float
    f1: float
    validation_timestamp: datetime


@dataclass(slots=True)
class DeploymentProfile:
    """Measured deployment characteristics."""

    model_version: str
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_rps: int
    deployment_timestamp: datetime


def initialize_tenant_config(
    tenant_id: str,
    validation_metrics: ValidationMetrics,
    deployment_profile: DeploymentProfile,
    daily_cost_budget: float,
    latency_sla_ms: float,
    auc_degradation_tolerance: float = 0.03,
    latency_multiplier: float = 1.2,
    missing_rate_threshold: float = 0.05,
) -> dict[str, float | str | datetime]:
    """Initialize tenant config from validation + deployment metrics.

    All values are traceable to their source, not hardcoded.

    Args:
        tenant_id: tenant identifier
        validation_metrics: offline validation results
        deployment_profile: measured runtime characteristics
        daily_cost_budget: operator-defined daily cost limit
        latency_sla_ms: operator-defined SLA (ms)
        auc_degradation_tolerance: alert if AUC drops below baseline - this (default 3%)
        latency_multiplier: alert if latency exceeds baseline * this (default 1.2x)
        missing_rate_threshold: alert if missing data rate exceeds this (domain knowledge)

    Returns:
        Dict ready for TenantConfig table
    """
    baseline_auc = validation_metrics.auc
    baseline_latency = deployment_profile.latency_p95_ms

    return {
        "tenant_id": tenant_id,
        # Deployment (from training)
        "model_version": deployment_profile.model_version,
        "decision_threshold": validation_metrics.optimal_threshold,
        "last_training_time": validation_metrics.validation_timestamp,
        # Baselines (measured)
        "baseline_auc": baseline_auc,
        "baseline_latency_ms": baseline_latency,
        # Alert thresholds (derived)
        "min_auc": baseline_auc - auc_degradation_tolerance,
        "max_latency_ms": baseline_latency * latency_multiplier,
        "max_missing_rate": missing_rate_threshold,
        # Business constraints
        "daily_cost_budget": daily_cost_budget,
        "latency_sla_ms": latency_sla_ms,
    }


# Default tier configurations (fallback if DB not available)
DEFAULT_TIER_CONFIGS = {
    "standard": {
        "threshold_enabled": True,
        "retrain_enabled": True,
        "rollback_enabled": True,
        "fallback_enabled": True,
        "datarepair_enabled": True,
        "business_value_weight": 0.30,
        "confidence_weight": 0.20,
        "risk_inverse_weight": 0.20,
        "cost_efficiency_weight": 0.10,
        "time_inverse_weight": 0.05,
        "historical_success_weight": 0.15,
    },
    "enterprise": {
        "threshold_enabled": False,
        "retrain_enabled": True,
        "rollback_enabled": True,
        "fallback_enabled": False,
        "datarepair_enabled": True,
        "business_value_weight": 0.25,
        "confidence_weight": 0.35,
        "risk_inverse_weight": 0.20,
        "cost_efficiency_weight": 0.05,
        "time_inverse_weight": 0.05,
        "historical_success_weight": 0.10,
    },
    "free": {
        "threshold_enabled": True,
        "retrain_enabled": False,
        "rollback_enabled": False,
        "fallback_enabled": True,
        "datarepair_enabled": False,
        "business_value_weight": 0.15,
        "confidence_weight": 0.15,
        "risk_inverse_weight": 0.10,
        "cost_efficiency_weight": 0.40,
        "time_inverse_weight": 0.10,
        "historical_success_weight": 0.10,
    },
}


def get_eligible_agents(tenant_id: str, failed_agents: list[str], db_session: Any = None) -> list[str]:
    """Get agents eligible for tenant, excluding failed ones.

    Args:
        tenant_id: tenant identifier
        failed_agents: list of agent types that failed
        db_session: optional DB session to load from TenantTierConfig

    Returns:
        List of eligible agent types
    """
    tier_config = None

    # Try to load from DB if session provided
    if db_session is not None:
        try:
            from self_healing_pipeline.db.models import TenantTierConfig
            db_tier = db_session.query(TenantTierConfig).filter_by(tenant_id=tenant_id).first()
            if db_tier:
                tier_config = {
                    "threshold": db_tier.threshold_enabled,
                    "retrain": db_tier.retrain_enabled,
                    "rollback": db_tier.rollback_enabled,
                    "fallback": db_tier.fallback_enabled,
                    "data_repair": db_tier.datarepair_enabled,
                }
        except Exception:
            pass

    # Fallback to defaults if DB lookup failed
    if tier_config is None:
        defaults = DEFAULT_TIER_CONFIGS.get(tenant_id)
        if defaults is None:
            # Unknown tenant: return failed agents as-is (fallback behavior)
            return failed_agents
        tier_config = {
            "threshold": defaults["threshold_enabled"],
            "retrain": defaults["retrain_enabled"],
            "rollback": defaults["rollback_enabled"],
            "fallback": defaults["fallback_enabled"],
            "data_repair": defaults["datarepair_enabled"],
        }

    eligible = [agent for agent, enabled in tier_config.items() if enabled and agent not in failed_agents]
    return eligible


def get_weight_overrides(tenant_id: str, db_session: Any = None) -> dict[str, float]:
    """Get weight overrides for tenant scoring.

    Args:
        tenant_id: tenant identifier
        db_session: optional DB session to load from TenantTierConfig

    Returns:
        Weight dict for scoring
    """
    weights = None

    # Try to load from DB if session provided
    if db_session is not None:
        try:
            from self_healing_pipeline.db.models import TenantTierConfig
            db_tier = db_session.query(TenantTierConfig).filter_by(tenant_id=tenant_id).first()
            if db_tier:
                weights = {
                    "business_value": db_tier.business_value_weight,
                    "confidence": db_tier.confidence_weight,
                    "risk_inverse": db_tier.risk_inverse_weight,
                    "cost_efficiency": db_tier.cost_efficiency_weight,
                    "time_inverse": db_tier.time_inverse_weight,
                    "historical_success": db_tier.historical_success_weight,
                }
        except Exception:
            pass

    # Fallback to defaults if DB lookup failed
    if weights is None:
        defaults = DEFAULT_TIER_CONFIGS.get(tenant_id, DEFAULT_TIER_CONFIGS["standard"])
        weights = {
            "business_value": defaults["business_value_weight"],
            "confidence": defaults["confidence_weight"],
            "risk_inverse": defaults["risk_inverse_weight"],
            "cost_efficiency": defaults["cost_efficiency_weight"],
            "time_inverse": defaults["time_inverse_weight"],
            "historical_success": defaults["historical_success_weight"],
        }

    return weights


def apply_tenant_weights(tenant_id: str, base_weights: Any, db_session: Any = None) -> Any:
    """Apply tenant-specific weight overrides to ScoringWeights.

    Args:
        tenant_id: tenant identifier
        base_weights: ScoringWeights object to modify
        db_session: optional DB session to load from TenantTierConfig

    Returns:
        New ScoringWeights with tenant overrides applied
    """
    from self_healing_pipeline.meta_harness.tuner import ScoringWeights

    overrides = get_weight_overrides(tenant_id, db_session=db_session)

    # Create new weights with tenant overrides
    weighted = ScoringWeights(
        business_value=overrides.get("business_value", base_weights.business_value),
        confidence=overrides.get("confidence", base_weights.confidence),
        risk_inverse=overrides.get("risk_inverse", base_weights.risk_inverse),
        cost_efficiency=overrides.get("cost_efficiency", base_weights.cost_efficiency),
        time_inverse=overrides.get("time_inverse", base_weights.time_inverse),
        historical_success=overrides.get("historical_success", base_weights.historical_success),
    )

    # Normalize to ensure weights still sum to 1.0
    total = weighted.total()
    if total > 0:
        weighted.business_value /= total
        weighted.confidence /= total
        weighted.risk_inverse /= total
        weighted.cost_efficiency /= total
        weighted.time_inverse /= total
        weighted.historical_success /= total

    return weighted
