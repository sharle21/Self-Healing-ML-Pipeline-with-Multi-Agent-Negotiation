"""Tests for tenant-specific configuration."""

import pytest

from self_healing_pipeline.config.tenant_config import (
    apply_tenant_weights,
    get_eligible_agents,
    get_weight_overrides,
)
from self_healing_pipeline.meta_harness.tuner import ScoringWeights


class TestTenantConfig:
    """Tenant configuration tests."""

    def test_get_eligible_agents_standard(self):
        """Test standard tenant allows all agents."""
        agents = get_eligible_agents("standard", [])
        assert len(agents) == 5
        assert "threshold" in agents
        assert "retrain" in agents

    def test_get_eligible_agents_enterprise(self):
        """Test enterprise tenant restricts to expensive agents."""
        agents = get_eligible_agents("enterprise", [])
        assert len(agents) == 3
        assert "retrain" in agents
        assert "data_repair" in agents
        assert "threshold" not in agents  # Fast but cheap

    def test_get_eligible_agents_free(self):
        """Test free tenant restricts to fast/cheap agents."""
        agents = get_eligible_agents("free", [])
        assert len(agents) == 2
        assert "threshold" in agents
        assert "fallback" in agents
        assert "retrain" not in agents  # Slow and expensive

    def test_get_eligible_agents_unknown_tenant(self):
        """Test unknown tenant uses defaults."""
        agents = get_eligible_agents("unknown-tenant", ["default1", "default2"])
        assert agents == ["default1", "default2"]

    def test_get_weight_overrides_standard(self):
        """Test standard tenant weight overrides."""
        overrides = get_weight_overrides("standard")
        assert overrides is not None
        assert overrides["confidence"] == 0.20
        assert overrides["business_value"] == 0.30

    def test_get_weight_overrides_enterprise(self):
        """Test enterprise tenant weight overrides (accuracy-focused)."""
        overrides = get_weight_overrides("enterprise")
        assert overrides is not None
        # Enterprise prioritizes confidence + business_value
        assert overrides["confidence"] > 0.20
        assert overrides["cost_efficiency"] < 0.10

    def test_get_weight_overrides_free(self):
        """Test free tenant weight overrides (cost-focused)."""
        overrides = get_weight_overrides("free")
        assert overrides is not None
        # Free prioritizes cost_efficiency
        assert overrides["cost_efficiency"] > 0.20
        assert overrides["time_inverse"] > 0.05

    def test_apply_tenant_weights(self):
        """Test applying tenant weights to base weights."""
        base = ScoringWeights()
        weighted = apply_tenant_weights("enterprise", base)

        # Enterprise should have higher confidence
        assert weighted.confidence > base.confidence
        # Enterprise should have lower cost_efficiency
        assert weighted.cost_efficiency < base.cost_efficiency

    def test_apply_tenant_weights_normalizes(self):
        """Test that applied weights still sum to 1.0."""
        base = ScoringWeights()
        weighted = apply_tenant_weights("free", base)

        # Weights should normalize
        assert weighted.total() == pytest.approx(1.0)

    def test_tenant_weight_differences(self):
        """Test that different tenants get different weights."""
        standard = apply_tenant_weights("standard", ScoringWeights())
        enterprise = apply_tenant_weights("enterprise", ScoringWeights())
        free = apply_tenant_weights("free", ScoringWeights())

        # All should have different confidence weights
        assert standard.confidence != enterprise.confidence
        assert enterprise.confidence != free.confidence
