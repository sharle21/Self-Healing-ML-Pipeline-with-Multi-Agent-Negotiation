"""Tenant onboarding: initialize new tenant in system with separate policy/validation/runtime records."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from self_healing_pipeline.config.tenant_config import (
    DeploymentProfile,
    ValidationMetrics,
)
from self_healing_pipeline.db.models import (
    TenantPolicy,
    ModelValidationReport,
    RuntimeDeploymentProfile,
)


def onboard_tenant(
    db_session: Session,
    tenant_id: str,
    validation_metrics: ValidationMetrics,
    deployment_profile: DeploymentProfile,
    daily_cost_budget: float,
    latency_sla_ms: float,
    auc_degradation_tolerance: float = 0.03,
    latency_multiplier: float = 1.2,
) -> dict[str, Any]:
    """Onboard new tenant: create policy + validation report + runtime profile.

    Args:
        db_session: database session
        tenant_id: tenant identifier
        validation_metrics: offline validation results
        deployment_profile: measured deployment characteristics
        daily_cost_budget: operator-defined daily budget
        latency_sla_ms: operator-defined SLA
        auc_degradation_tolerance: threshold for AUC alerts
        latency_multiplier: multiplier for latency alerts

    Returns:
        Dict with policy, validation_report, runtime_profile
    """
    # Create TenantPolicy (governance)
    policy = TenantPolicy(
        tenant_id=tenant_id,
        min_acceptable_auc=validation_metrics.auc - auc_degradation_tolerance,
        max_acceptable_latency_ms=deployment_profile.latency_p95_ms * latency_multiplier,
        daily_cost_budget=daily_cost_budget,
        latency_sla_ms=latency_sla_ms,
    )
    db_session.add(policy)

    # Create ModelValidationReport (immutable)
    validation_report = ModelValidationReport(
        model_version=deployment_profile.model_version,
        tenant_id=tenant_id,
        auc=validation_metrics.auc,
        precision=validation_metrics.precision,
        recall=validation_metrics.recall,
        f1_score=validation_metrics.f1,
        optimal_threshold=validation_metrics.optimal_threshold,
        calibration_error=0.05,
        validated_at=validation_metrics.validation_timestamp,
    )
    db_session.add(validation_report)

    # Create RuntimeDeploymentProfile (continuous)
    runtime_profile = RuntimeDeploymentProfile(
        tenant_id=tenant_id,
        model_version=deployment_profile.model_version,
        latency_p95_ms=deployment_profile.latency_p95_ms,
        latency_p99_ms=deployment_profile.latency_p99_ms,
        throughput_rps=float(deployment_profile.throughput_rps),
        measured_at=deployment_profile.deployment_timestamp,
    )
    db_session.add(runtime_profile)
    db_session.commit()

    return {
        "tenant_id": tenant_id,
        "policy": policy,
        "validation_report": validation_report,
        "runtime_profile": runtime_profile,
    }


def get_tenant_policy(db_session: Session, tenant_id: str) -> TenantPolicy | None:
    """Retrieve tenant policy from DB.

    Args:
        db_session: database session
        tenant_id: tenant identifier

    Returns:
        TenantPolicy or None if not found
    """
    return db_session.query(TenantPolicy).filter_by(tenant_id=tenant_id).first()


def get_latest_validation_report(
    db_session: Session, tenant_id: str
) -> ModelValidationReport | None:
    """Get latest model validation report for tenant.

    Args:
        db_session: database session
        tenant_id: tenant identifier

    Returns:
        Latest ModelValidationReport or None
    """
    return (
        db_session.query(ModelValidationReport)
        .filter_by(tenant_id=tenant_id)
        .order_by(ModelValidationReport.validated_at.desc())
        .first()
    )


def list_tenants(db_session: Session) -> list[dict[str, Any]]:
    """List all configured tenants with their policies.

    Args:
        db_session: database session

    Returns:
        List of tenant info dicts
    """
    policies = db_session.query(TenantPolicy).all()
    return [
        {
            "tenant_id": p.tenant_id,
            "min_acceptable_auc": p.min_acceptable_auc,
            "max_acceptable_latency_ms": p.max_acceptable_latency_ms,
            "latency_sla_ms": p.latency_sla_ms,
            "daily_cost_budget": p.daily_cost_budget,
            "risk_tolerance": p.risk_tolerance,
            "updated_at": p.updated_at.isoformat(),
        }
        for p in policies
    ]
