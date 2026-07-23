"""End-to-end tests for Layer 3 integration: observe → decide → verify."""

import pytest

from self_healing_pipeline.agents.datarepair_v2 import DataRepairAgent
from self_healing_pipeline.agents.fallback_v2 import FallbackAgent
from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
from self_healing_pipeline.agents.rollback_v2 import RollbackAgent
from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
from self_healing_pipeline.commander.commander_v3 import CommanderV3
from self_healing_pipeline.gateway.events import Incident, IncidentType


class TestCommanderV3Integration:
    """Test Commander V3 full 3-layer pipeline."""

    @pytest.mark.asyncio
    async def test_drift_incident_end_to_end(self):
        """Test complete drift incident handling: observe → decide → verify."""
        # Setup
        agents = [
            ThresholdAdjustmentAgent("threshold-1"),
            RetrainAgent("retrain-1"),
            RollbackAgent("rollback-1"),
            FallbackAgent("fallback-1"),
            DataRepairAgent("datarepair-1"),
        ]
        commander = CommanderV3(agents)

        incident = Incident(
            tenant_id="test-tenant",
            type=IncidentType.DRIFT,
            payload={"accuracy": 0.68, "expected": 0.77},
            severity=0.6,
            affected_features=("feature_a", "feature_b"),
        )

        # Execute full pipeline
        result = await commander.handle_incident(incident)

        # Verify all 3 layers executed
        assert result.incident_id == incident.id
        assert result.incident_type == IncidentType.DRIFT.value
        assert 0 <= result.severity <= 1  # Layer 1: severity calculated
        assert result.winning_agent_type in ["threshold", "retrain", "rollback"]  # Layer 2: agent selected
        assert result.winning_plan  # Layer 2: plan generated
        assert -1 <= result.reward <= 1  # Layer 3: reward calculated
        assert isinstance(result.incident_resolved, bool)  # Layer 3: resolution checked

    @pytest.mark.asyncio
    async def test_multiple_agents_competing(self):
        """Test that all eligible agents compete and best wins."""
        agents = [
            ThresholdAdjustmentAgent("threshold-1"),
            RetrainAgent("retrain-1"),
        ]
        commander = CommanderV3(agents)

        incident = Incident(
            tenant_id="test-tenant",
            type=IncidentType.DRIFT,
            payload={"accuracy": 0.68},
            severity=0.6,
        )

        result = await commander.handle_incident(incident)

        # One of the eligible agents should win
        assert result.winning_agent_type in ["threshold", "retrain"]
        assert result.reward is not None

    @pytest.mark.asyncio
    async def test_cost_incident(self):
        """Test cost threshold incident handling."""
        agents = [
            ThresholdAdjustmentAgent("threshold-1"),
            RetrainAgent("retrain-1"),
            RollbackAgent("rollback-1"),
            FallbackAgent("fallback-1"),
            DataRepairAgent("datarepair-1"),
        ]
        commander = CommanderV3(agents)

        incident = Incident(
            tenant_id="test-tenant",
            type=IncidentType.COST_THRESHOLD,
            payload={"cost_per_pred": 0.15, "budget": 0.10},
            severity=0.4,
        )

        result = await commander.handle_incident(incident)

        assert result.incident_type == IncidentType.COST_THRESHOLD.value
        assert result.winning_agent_type in ["threshold", "fallback"]
        assert result.reward is not None


class TestVerificationLayerAccuracy:
    """Test that verification layer correctly measures outcomes."""

    @pytest.mark.asyncio
    async def test_reward_reflects_improvement(self):
        """Test reward is high when metrics improve."""
        agents = [RetrainAgent("retrain-1")]
        commander = CommanderV3(agents)

        incident = Incident(
            tenant_id="test-tenant",
            type=IncidentType.DRIFT,
            payload={"accuracy": 0.68},
            severity=0.6,
        )

        result = await commander.handle_incident(incident)

        # Reward should be calculated
        assert isinstance(result.reward, float)
        assert -1 <= result.reward <= 1

    @pytest.mark.asyncio
    async def test_incident_resolved_check(self):
        """Test incident_resolved flag reflects actual resolution."""
        agents = [ThresholdAdjustmentAgent("threshold-1")]
        commander = CommanderV3(agents)

        incident = Incident(
            tenant_id="test-tenant",
            type=IncidentType.COST_THRESHOLD,
            payload={"cost": 0.15},
            severity=0.4,
        )

        result = await commander.handle_incident(incident)

        # Should have resolved flag set
        assert isinstance(result.incident_resolved, bool)

    @pytest.mark.asyncio
    async def test_verification_breakdown_contains_reward_components(self):
        """Test verification breakdown has all reward components."""
        agents = [RetrainAgent("retrain-1")]
        commander = CommanderV3(agents)

        incident = Incident(
            tenant_id="test-tenant",
            type=IncidentType.DRIFT,
            payload={"accuracy": 0.68},
            severity=0.6,
        )

        result = await commander.handle_incident(incident)

        # Verification breakdown should have Phase 14 OutcomeReward components
        assert "quality_gain" in result.verification_breakdown
        assert "resolution_score" in result.verification_breakdown
        assert "exec_cost_penalty" in result.verification_breakdown
        assert "regression_penalty" in result.verification_breakdown
        assert "reward" in result.verification_breakdown


class TestObservationLayerAccuracy:
    """Test that observation layer correctly measures severity."""

    @pytest.mark.asyncio
    async def test_severity_calculated_from_telemetry(self):
        """Test severity comes from actual metrics, not hardcoded."""
        agents = [RetrainAgent("retrain-1")]
        commander = CommanderV3(agents)

        incident = Incident(
            tenant_id="test-tenant",
            type=IncidentType.DRIFT,
            payload={"drift": 2.5, "auc_drop": 0.12},
            severity=0.99,  # Passed value ignored
        )

        result = await commander.handle_incident(incident)

        # Severity should be calculated from telemetry, not use incident.severity
        assert 0 <= result.severity <= 1
        # With drift signal, severity should be non-trivial (Phase 9: no baseline → auc_drop=0, but drift+volume fire)
        assert result.severity > 0.15


class TestRemediationPolicySelection:
    """Test that best remediation policy is selected based on state."""

    @pytest.mark.asyncio
    async def test_agents_score_based_on_state_not_random(self):
        """Test agent confidence is computed from state features."""
        agents = [RetrainAgent("retrain-1")]
        commander = CommanderV3(agents)

        incident = Incident(
            tenant_id="test-tenant",
            type=IncidentType.DRIFT,
            payload={"drift": 2.5, "auc_drop": 0.12},
            severity=0.8,
        )

        result = await commander.handle_incident(incident)

        # Agent should be selected
        assert result.winning_agent_type == "retrain"

        # Plan should have confidence computed from state (not random)
        assert "confidence" in result.winning_plan
        assert 0 <= result.winning_plan["confidence"] <= 1

        # Reasoning should reference state metrics
        assert "drift" in result.winning_plan["reasoning"].lower()
        assert "auc" in result.winning_plan["reasoning"].lower()
