"""Tests for reconciliation debate logic."""

import pytest

from self_healing_pipeline.agents.base import Proposal
from self_healing_pipeline.commander.reconciliation import Reconciliation
from self_healing_pipeline.gateway.events import Incident, IncidentType


@pytest.fixture
def incident():
    """Test incident."""
    return Incident(
        tenant_id="standard",
        type=IncidentType.DRIFT,
        payload={},
        severity=0.5,
    )


@pytest.fixture
def reconciliation():
    """Reconciliation instance (heuristic mode, no API key)."""
    return Reconciliation(model_name=None)


class TestReconciliation:
    """Reconciliation debate tests."""

    def test_debate_heuristic_picks_safer_option(self, incident, reconciliation):
        """Test that heuristic debate picks safer option when risk differs significantly."""
        safe = Proposal(
            agent_id="safe-1",
            agent_type="rollback",
            confidence=0.8,
            estimated_business_savings=2000.0,
            estimated_risk=0.05,  # Much safer
            estimated_compute_cost=2.0,
            estimated_time=15.0,
            rationale="Safe rollback",
        )
        risky = Proposal(
            agent_id="risky-1",
            agent_type="retrain",
            confidence=0.75,
            estimated_business_savings=8000.0,
            estimated_risk=0.30,  # Much riskier
            estimated_compute_cost=50.0,
            estimated_time=180.0,
            rationale="Risky retrain",
        )

        result = reconciliation._debate_heuristic(safe, risky, incident)
        assert "risk" in result.rationale.lower() or "safer" in result.rationale.lower()

    def test_debate_heuristic_picks_faster_option(self, incident, reconciliation):
        """Test that heuristic debate picks faster option when time differs significantly."""
        fast = Proposal(
            agent_id="fast-1",
            agent_type="threshold",
            confidence=0.7,
            estimated_business_savings=1500.0,
            estimated_risk=0.10,
            estimated_compute_cost=0.5,
            estimated_time=5.0,  # Much faster
            rationale="Fast threshold",
        )
        slow = Proposal(
            agent_id="slow-1",
            agent_type="retrain",
            confidence=0.75,
            estimated_business_savings=8000.0,
            estimated_risk=0.30,
            estimated_compute_cost=50.0,
            estimated_time=180.0,  # Much slower
            rationale="Slow retrain",
        )

        result = reconciliation._debate_heuristic(fast, slow, incident)
        assert "speed" in result.rationale.lower() or "faster" in result.rationale.lower()

    def test_debate_heuristic_uses_confidence_tiebreaker(self, incident, reconciliation):
        """Test that heuristic uses confidence as tiebreaker when other factors are close."""
        similar1 = Proposal(
            agent_id="agent-1",
            agent_type="threshold",
            confidence=0.85,  # Higher confidence
            estimated_business_savings=1500.0,
            estimated_risk=0.10,
            estimated_compute_cost=0.5,
            estimated_time=5.0,
            rationale="Threshold",
        )
        similar2 = Proposal(
            agent_id="agent-2",
            agent_type="rollback",
            confidence=0.65,  # Lower confidence
            estimated_business_savings=2000.0,
            estimated_risk=0.05,
            estimated_compute_cost=2.0,
            estimated_time=15.0,
            rationale="Rollback",
        )

        result = reconciliation._debate_heuristic(similar1, similar2, incident)
        assert result.winner_type == "threshold"

    def test_debate_heuristic_generates_log(self, incident, reconciliation):
        """Test that debate generates a log of reasoning."""
        prop1 = Proposal(
            agent_id="a1",
            agent_type="agent_a",
            confidence=0.8,
            estimated_business_savings=5000.0,
            estimated_risk=0.20,
            estimated_compute_cost=30.0,
            estimated_time=100.0,
            rationale="Agent A",
        )
        prop2 = Proposal(
            agent_id="a2",
            agent_type="agent_b",
            confidence=0.75,
            estimated_business_savings=4000.0,
            estimated_risk=0.15,
            estimated_compute_cost=25.0,
            estimated_time=80.0,
            rationale="Agent B",
        )

        result = reconciliation._debate_heuristic(prop1, prop2, incident)
        assert len(result.debate_log) > 0
        assert result.debate_log[0].startswith("Debating")
        assert "Tiebreaker" in "\n".join(result.debate_log)

    def test_debate_heuristic_sets_winner_type(self, incident, reconciliation):
        """Test that result includes winner type."""
        prop1 = Proposal(
            agent_id="a1",
            agent_type="threshold",
            confidence=0.9,
            estimated_business_savings=1500.0,
            estimated_risk=0.10,
            estimated_compute_cost=0.5,
            estimated_time=5.0,
            rationale="Threshold",
        )
        prop2 = Proposal(
            agent_id="a2",
            agent_type="retrain",
            confidence=0.7,
            estimated_business_savings=8000.0,
            estimated_risk=0.30,
            estimated_compute_cost=50.0,
            estimated_time=180.0,
            rationale="Retrain",
        )

        result = reconciliation._debate_heuristic(prop1, prop2, incident)
        assert result.winner_type in ("threshold", "retrain")
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_debate_async_heuristic(self, incident):
        """Test async debate method (falls back to heuristic)."""
        reconciliation = Reconciliation(model_name=None)
        prop1 = Proposal(
            agent_id="a1",
            agent_type="threshold",
            confidence=0.8,
            estimated_business_savings=1500.0,
            estimated_risk=0.10,
            estimated_compute_cost=0.5,
            estimated_time=5.0,
            rationale="Threshold",
        )
        prop2 = Proposal(
            agent_id="a2",
            agent_type="rollback",
            confidence=0.75,
            estimated_business_savings=2000.0,
            estimated_risk=0.05,
            estimated_compute_cost=2.0,
            estimated_time=15.0,
            rationale="Rollback",
        )

        result = await reconciliation.debate(prop1, prop2, incident)
        assert result.winner_type in ("threshold", "rollback")
        assert len(result.debate_log) > 0
