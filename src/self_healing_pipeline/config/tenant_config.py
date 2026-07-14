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


# Tier-based agent eligibility (backward compatible for existing tests)
TENANT_AGENT_TIERS = {
    "standard": {
        "threshold": True,
        "retrain": True,
        "rollback": True,
        "fallback": True,
        "data_repair": True,
    },
    "enterprise": {
        "threshold": False,
        "retrain": True,
        "rollback": True,
        "fallback": False,
        "data_repair": True,
    },
    "free": {
        "threshold": True,
        "retrain": False,
        "rollback": False,
        "fallback": True,
        "data_repair": False,
    },
}

TENANT_WEIGHT_OVERRIDES = {
    "standard": {
        "business_value": 0.30,
        "confidence": 0.20,
        "risk_inverse": 0.20,
        "cost_efficiency": 0.10,
        "time_inverse": 0.05,
        "historical_success": 0.15,
    },
    "enterprise": {
        "business_value": 0.25,
        "confidence": 0.35,
        "risk_inverse": 0.20,
        "cost_efficiency": 0.05,
        "time_inverse": 0.05,
        "historical_success": 0.10,
    },
    "free": {
        "business_value": 0.15,
        "confidence": 0.15,
        "risk_inverse": 0.10,
        "cost_efficiency": 0.40,
        "time_inverse": 0.10,
        "historical_success": 0.10,
    },
}


def get_eligible_agents(tenant_id: str, failed_agents: list[str]) -> list[str]:
    """Get agents eligible for tenant, excluding failed ones.

    Args:
        tenant_id: tenant identifier
        failed_agents: list of agent types that failed

    Returns:
        List of eligible agent types
    """
    tier_config = TENANT_AGENT_TIERS.get(tenant_id)
    if tier_config is None:
        # Unknown tenant: return failed agents as-is (fallback behavior)
        return failed_agents

    eligible = [agent for agent, enabled in tier_config.items() if enabled and agent not in failed_agents]
    return eligible


def get_weight_overrides(tenant_id: str) -> dict[str, float]:
    """Get weight overrides for tenant scoring.

    Args:
        tenant_id: tenant identifier

    Returns:
        Weight dict for scoring
    """
    return TENANT_WEIGHT_OVERRIDES.get(tenant_id, TENANT_WEIGHT_OVERRIDES["standard"])


def apply_tenant_weights(tenant_id: str, base_weights: Any) -> Any:
    """Apply tenant-specific weight overrides to ScoringWeights.

    Args:
        tenant_id: tenant identifier
        base_weights: ScoringWeights object to modify

    Returns:
        New ScoringWeights with tenant overrides applied
    """
    from self_healing_pipeline.meta_harness.tuner import ScoringWeights

    overrides = get_weight_overrides(tenant_id)

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
