"""Apply tuned ScoringWeights to a tenant's live TenantTierConfig row.

Closes the loop between the offline meta-harness (analyze -> tune -> version)
and the live commander, which reads weights from TenantTierConfig via
UtilityScorer.weights_from_tier_config(). Without this, tuned weights were
only ever saved to versioned JSON and never reached a live decision.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from self_healing_pipeline.db.models import TenantTierConfig
from self_healing_pipeline.meta_harness.tuner import ScoringWeights

# ScoringWeights field -> TenantTierConfig column. historical_success is the
# only ScoringWeights dimension with a direct 1:1 column; the others already
# share a name prefix by convention.
_FIELD_TO_COLUMN = {
    "business_value": "business_value_weight",
    "confidence": "confidence_weight",
    "risk_inverse": "risk_inverse_weight",
    "cost_efficiency": "cost_efficiency_weight",
    "time_inverse": "time_inverse_weight",
    "historical_success": "historical_success_weight",
}


def apply_weights_to_tier_config(weights: ScoringWeights, tier_config: TenantTierConfig) -> None:
    """Mutate a TenantTierConfig row in place with tuned weight values."""
    for field, column in _FIELD_TO_COLUMN.items():
        setattr(tier_config, column, getattr(weights, field))


def sync_tuned_weights(session: Session, tenant_id: str, weights: ScoringWeights) -> TenantTierConfig:
    """Apply tuned weights to `tenant_id`'s TenantTierConfig row and commit.

    Creates the row with defaults first if the tenant has none yet.

    Args:
        session: active DB session
        tenant_id: tenant to update
        weights: tuned ScoringWeights to persist

    Returns:
        The updated (and committed) TenantTierConfig row.
    """
    tier_config = session.query(TenantTierConfig).filter_by(tenant_id=tenant_id).first()
    if tier_config is None:
        tier_config = TenantTierConfig(tenant_id=tenant_id)
        session.add(tier_config)

    apply_weights_to_tier_config(weights, tier_config)
    session.commit()
    return tier_config
