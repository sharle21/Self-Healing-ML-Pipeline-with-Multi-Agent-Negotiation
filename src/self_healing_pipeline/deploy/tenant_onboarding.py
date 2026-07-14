"""Tenant onboarding: initialize new tenant in system."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from self_healing_pipeline.config.tenant_config import (
    DeploymentProfile,
    ValidationMetrics,
    initialize_tenant_config,
)
from self_healing_pipeline.db.models import TenantConfig


def onboard_tenant(
    db_session: Session,
    tenant_id: str,
    validation_metrics: ValidationMetrics,
    deployment_profile: DeploymentProfile,
    daily_cost_budget: float,
    latency_sla_ms: float,
) -> TenantConfig:
    """Onboard new tenant with measured validation + deployment data.

    Args:
        db_session: database session
        tenant_id: tenant identifier
        validation_metrics: offline validation results
        deployment_profile: measured deployment characteristics
        daily_cost_budget: operator-defined daily budget
        latency_sla_ms: operator-defined SLA

    Returns:
        TenantConfig row in DB
    """
    config_dict = initialize_tenant_config(
        tenant_id=tenant_id,
        validation_metrics=validation_metrics,
        deployment_profile=deployment_profile,
        daily_cost_budget=daily_cost_budget,
        latency_sla_ms=latency_sla_ms,
    )

    # Check if tenant already exists
    existing = db_session.query(TenantConfig).filter_by(tenant_id=tenant_id).first()
    if existing:
        # Update existing
        for key, value in config_dict.items():
            if key != "tenant_id":
                setattr(existing, key, value)
        db_session.commit()
        return existing

    # Create new
    config_row = TenantConfig(**config_dict)
    db_session.add(config_row)
    db_session.commit()
    db_session.refresh(config_row)
    return config_row


def get_tenant_config(db_session: Session, tenant_id: str) -> TenantConfig | None:
    """Retrieve tenant config from DB.

    Args:
        db_session: database session
        tenant_id: tenant identifier

    Returns:
        TenantConfig or None if not found
    """
    return db_session.query(TenantConfig).filter_by(tenant_id=tenant_id).first()


def list_tenants(db_session: Session) -> list[dict[str, Any]]:
    """List all configured tenants.

    Args:
        db_session: database session

    Returns:
        List of tenant configs as dicts
    """
    configs = db_session.query(TenantConfig).all()
    return [
        {
            "tenant_id": c.tenant_id,
            "model_version": c.model_version,
            "baseline_auc": c.baseline_auc,
            "baseline_latency_ms": c.baseline_latency_ms,
            "latency_sla_ms": c.latency_sla_ms,
            "daily_cost_budget": c.daily_cost_budget,
            "updated_at": c.updated_at.isoformat(),
        }
        for c in configs
    ]
