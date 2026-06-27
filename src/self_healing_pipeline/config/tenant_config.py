"""Tenant-specific configuration helpers."""

from typing import Any

from self_healing_pipeline.config import load_tenants_config
from self_healing_pipeline.meta_harness.tuner import ScoringWeights


def get_eligible_agents(tenant_id: str, default_agents: list[str]) -> list[str]:
    """Get eligible agents for a tenant.

    Args:
        tenant_id: tenant identifier
        default_agents: agent types to use if no tenant config

    Returns:
        List of agent types allowed for this tenant
    """
    config = load_tenants_config()
    tenant = config.get(tenant_id, {})
    return tenant.get("eligible_agents", default_agents)


def get_weight_overrides(tenant_id: str) -> dict[str, float] | None:
    """Get weight overrides for a tenant.

    Args:
        tenant_id: tenant identifier

    Returns:
        Weight overrides dict, or None if using defaults
    """
    config = load_tenants_config()
    tenant = config.get(tenant_id, {})
    return tenant.get("weight_overrides")


def apply_tenant_weights(tenant_id: str, base_weights: ScoringWeights) -> ScoringWeights:
    """Apply tenant-specific weight overrides.

    Args:
        tenant_id: tenant identifier
        base_weights: default weights

    Returns:
        Weights with tenant overrides applied
    """
    overrides = get_weight_overrides(tenant_id)
    if not overrides:
        return base_weights

    return ScoringWeights.from_dict({
        **base_weights.to_dict(),
        **overrides,
    })
