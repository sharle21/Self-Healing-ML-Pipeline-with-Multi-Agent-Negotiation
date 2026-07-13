"""Tests for LangGraph-based multi-turn reconciliation."""

import pytest

from self_healing_pipeline.agents.base import Proposal
from self_healing_pipeline.commander.reconciliation_langgraph import (
    LangGraphReconciliation,
)
from self_healing_pipeline.gateway.events import Incident, IncidentType


@pytest.fixture
def proposals():
    """Create test proposals."""
    top1 = Proposal(
        agent_id="threshold-1",
        agent_type="threshold",
        confidence=0.65,
        estimated_business_savings=450.0,
        estimated_risk=0.10,
        estimated_compute_cost=0.50,
        estimated_time=5.0,
        rationale="Adjust decision boundary quickly.",
    )

    top2 = Proposal(
        agent_id="retrain-1",
        agent_type="retrain",
        confidence=0.55,
        estimated_business_savings=2400.0,
        estimated_risk=0.15,
        estimated_compute_cost=50.0,
        estimated_time=180.0,
        rationale="Refit model on recent data.",
    )

    return top1, top2


@pytest.fixture
def incident():
    """Create test incident."""
    return Incident(
        tenant_id="test-tenant",
        type=IncidentType.DRIFT,
        payload={"accuracy": 0.78, "expected": 0.92},
        severity=0.3,
        affected_features=("feature_a", "feature_b"),
    )


class TestLangGraphReconciliation:
    """LangGraph reconciliation tests."""

    def test_heuristic_fallback_no_api_key(self, proposals, incident):
        """Test fallback to heuristic when no API key."""
        top1, top2 = proposals
        reconciliation = LangGraphReconciliation(model_name=None)

        result = reconciliation._heuristic_fallback(top1, top2)

        assert result.winner_type in ["threshold", "retrain"]
        assert result.confidence > 0.0
        assert len(result.debate_log) > 0

    def test_heuristic_chooses_safer_agent(self):
        """Test heuristic picks safer agent when risk differs significantly."""
        safe_proposal = Proposal(
            agent_id="rollback-1",
            agent_type="rollback",
            confidence=0.60,
            estimated_business_savings=600.0,
            estimated_risk=0.05,
            estimated_compute_cost=2.0,
            estimated_time=15.0,
            rationale="Rollback to previous version.",
        )

        risky_proposal = Proposal(
            agent_id="retrain-1",
            agent_type="retrain",
            confidence=0.55,
            estimated_business_savings=2400.0,
            estimated_risk=0.25,
            estimated_compute_cost=50.0,
            estimated_time=180.0,
            rationale="Refit model.",
        )

        reconciliation = LangGraphReconciliation()
        result = reconciliation._heuristic_fallback(safe_proposal, risky_proposal)

        assert result.winner_type == "rollback"
        assert "safe" in result.rationale.lower()

    def test_heuristic_chooses_confident_agent(self):
        """Test heuristic picks more confident agent on tie."""
        prop1 = Proposal(
            agent_id="a-1",
            agent_type="agent_a",
            confidence=0.70,
            estimated_business_savings=1000.0,
            estimated_risk=0.15,
            estimated_compute_cost=10.0,
            estimated_time=60.0,
            rationale="Option A.",
        )

        prop2 = Proposal(
            agent_id="b-1",
            agent_type="agent_b",
            confidence=0.50,
            estimated_business_savings=1000.0,
            estimated_risk=0.15,
            estimated_compute_cost=10.0,
            estimated_time=60.0,
            rationale="Option B.",
        )

        reconciliation = LangGraphReconciliation()
        result = reconciliation._heuristic_fallback(prop1, prop2)

        assert result.winner_type == "agent_a"

    @pytest.mark.asyncio
    async def test_debate_returns_result(self, proposals, incident):
        """Test debate returns valid ReconciliationResult."""
        top1, top2 = proposals
        reconciliation = LangGraphReconciliation(model_name=None)

        result = await reconciliation.debate(top1, top2, incident)

        assert result.winner_type in ["threshold", "retrain"]
        assert result.confidence > 0.0
        assert result.rationale is not None
        assert len(result.debate_log) > 0

    def test_debate_state_initialization(self, proposals, incident):
        """Test DebateState initialized correctly."""
        top1, top2 = proposals
        reconciliation = LangGraphReconciliation()

        # Would need to extract state init logic for unit test
        # For now, test graph building doesn't crash
        graph = reconciliation._build_graph()
        assert graph is not None
