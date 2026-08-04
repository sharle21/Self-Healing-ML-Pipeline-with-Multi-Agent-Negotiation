"""Meta-harness -> live commander wiring: tuned ScoringWeights must reach
TenantTierConfig, since that is what UtilityScorer.weights_from_tier_config
actually reads at decision time.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from self_healing_pipeline.commander.utility import UtilityScorer
from self_healing_pipeline.db.models import Base, TenantTierConfig
from self_healing_pipeline.meta_harness.apply import (
    apply_weights_to_tier_config,
    sync_tuned_weights,
)
from self_healing_pipeline.meta_harness.tuner import ScoringWeights


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine, future=True)


def test_apply_weights_to_tier_config_mutates_matching_columns() -> None:
    tier_config = TenantTierConfig(tenant_id="enterprise")
    weights = ScoringWeights(
        business_value=0.45,
        confidence=0.10,
        risk_inverse=0.15,
        cost_efficiency=0.05,
        time_inverse=0.05,
        historical_success=0.20,
    )

    apply_weights_to_tier_config(weights, tier_config)

    assert tier_config.business_value_weight == 0.45
    assert tier_config.confidence_weight == 0.10
    assert tier_config.risk_inverse_weight == 0.15
    assert tier_config.cost_efficiency_weight == 0.05
    assert tier_config.time_inverse_weight == 0.05
    assert tier_config.historical_success_weight == 0.20


def test_sync_tuned_weights_creates_row_when_tenant_has_none() -> None:
    session = _make_session()
    weights = ScoringWeights(business_value=0.50)

    tier_config = sync_tuned_weights(session, "new_tenant", weights)

    assert tier_config.tenant_id == "new_tenant"
    assert tier_config.business_value_weight == 0.50

    reloaded = session.query(TenantTierConfig).filter_by(tenant_id="new_tenant").first()
    assert reloaded is not None
    assert reloaded.business_value_weight == 0.50


def test_sync_tuned_weights_updates_existing_row() -> None:
    session = _make_session()
    session.add(TenantTierConfig(tenant_id="standard", business_value_weight=0.30))
    session.commit()

    sync_tuned_weights(session, "standard", ScoringWeights(business_value=0.60))

    reloaded = session.query(TenantTierConfig).filter_by(tenant_id="standard").first()
    assert reloaded.business_value_weight == 0.60


def test_synced_weights_are_visible_to_live_utility_scorer() -> None:
    """The actual bug this closes: tuned weights must change commander scoring."""
    session = _make_session()
    tuned = ScoringWeights(
        business_value=0.70, confidence=0.05, risk_inverse=0.05,
        cost_efficiency=0.05, time_inverse=0.05, historical_success=0.10,
    )
    sync_tuned_weights(session, "enterprise", tuned)

    tier_config = session.query(TenantTierConfig).filter_by(tenant_id="enterprise").first()
    utility_weights = UtilityScorer.weights_from_tier_config(tier_config, "drift")

    assert utility_weights.quality == 0.70
    assert utility_weights.confidence == 0.05
    assert utility_weights.risk == 0.05
