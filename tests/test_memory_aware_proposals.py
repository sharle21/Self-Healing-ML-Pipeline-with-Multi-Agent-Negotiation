"""Tests for memory-aware agent proposals (confidence scaling)."""

import pytest

from self_healing_pipeline.agents.threshold import ThresholdAgent
from self_healing_pipeline.gateway.events import Incident, IncidentType
from self_healing_pipeline.memory.tier2_store import AgentStats


@pytest.fixture
def incident():
    """Create a test incident."""
    return Incident(
        tenant_id="standard",
        type=IncidentType.DRIFT,
        payload={"drift_percentage": 0.5},
        severity=0.5,
        affected_features=(),
    )


class TestMemoryAwareProposals:
    """Memory-aware proposal tests."""

    def test_proposal_confidence_baseline(self, incident):
        """Test proposal confidence without memory context."""
        agent = ThresholdAgent("agent-1")
        proposal = agent._analyze_heuristic(incident, {})
        # Baseline: 0.55 + 0.25 * 0.5 = 0.675
        assert proposal.confidence == pytest.approx(0.675)

    def test_proposal_confidence_cold_start(self, incident):
        """Test proposal with cold start (empty agents context)."""
        agent = ThresholdAgent("agent-1")
        memory = {"agents": {}, "cold_start": True}
        proposal = agent._analyze_heuristic(incident, memory)
        # Should be baseline since agent not in memory
        assert proposal.confidence == pytest.approx(0.675)

    def test_proposal_confidence_with_perfect_history(self, incident):
        """Test confidence scaling with 100% success rate."""
        agent = ThresholdAgent("agent-1")
        # Agent has perfect recent history
        agent_stats = AgentStats(
            agent_type="threshold",
            attempts=5,
            successes=5,
            success_rate=1.0,
            recent_success_rate=1.0,  # All 5 recent outcomes succeeded
            avg_business_savings=1000.0,
            avg_duration=5.0,
        )
        memory = {"agents": {"threshold": agent_stats}}
        proposal = agent._analyze_heuristic(incident, memory)
        # Scaled: 0.675 * (0.5 + 0.5 * 1.0) = 0.675 * 1.0 = 0.675
        assert proposal.confidence == pytest.approx(0.675)

    def test_proposal_confidence_with_poor_history(self, incident):
        """Test confidence scaling with 0% success rate."""
        agent = ThresholdAgent("agent-1")
        agent_stats = AgentStats(
            agent_type="threshold",
            attempts=5,
            successes=0,
            success_rate=0.0,
            recent_success_rate=0.0,  # All 5 recent outcomes failed
            avg_business_savings=0.0,
            avg_duration=5.0,
        )
        memory = {"agents": {"threshold": agent_stats}}
        proposal = agent._analyze_heuristic(incident, memory)
        # Scaled: 0.675 * (0.5 + 0.5 * 0.0) = 0.675 * 0.5 = 0.3375
        assert proposal.confidence == pytest.approx(0.3375)

    def test_proposal_confidence_with_moderate_history(self, incident):
        """Test confidence scaling with 50% success rate."""
        agent = ThresholdAgent("agent-1")
        agent_stats = AgentStats(
            agent_type="threshold",
            attempts=5,
            successes=2,
            success_rate=0.4,
            recent_success_rate=0.5,  # 2-3 successes in recent outcomes
            avg_business_savings=500.0,
            avg_duration=5.0,
        )
        memory = {"agents": {"threshold": agent_stats}}
        proposal = agent._analyze_heuristic(incident, memory)
        # Scaled: 0.675 * (0.5 + 0.5 * 0.5) = 0.675 * 0.75 = 0.50625
        assert proposal.confidence == pytest.approx(0.50625)

    def test_proposal_confidence_stays_in_bounds(self, incident):
        """Test that scaling keeps confidence in [0, 1]."""
        agent = ThresholdAgent("agent-1")
        # Even with low baseline and low history, should be in bounds
        agent_stats = AgentStats(
            agent_type="threshold",
            attempts=1,
            successes=0,
            success_rate=0.0,
            recent_success_rate=0.0,
            avg_business_savings=0.0,
            avg_duration=5.0,
        )
        memory = {"agents": {"threshold": agent_stats}}
        proposal = agent._analyze_heuristic(incident, memory)
        assert 0.0 <= proposal.confidence <= 1.0

    def test_proposal_preserves_other_fields(self, incident):
        """Test that confidence scaling doesn't affect other proposal fields."""
        agent = ThresholdAgent("agent-1")
        agent_stats = AgentStats(
            agent_type="threshold",
            attempts=5,
            successes=5,
            success_rate=1.0,
            recent_success_rate=1.0,
            avg_business_savings=1000.0,
            avg_duration=5.0,
        )
        memory = {"agents": {"threshold": agent_stats}}
        proposal = agent._analyze_heuristic(incident, memory)

        # Other fields should be unchanged
        assert proposal.agent_type == "threshold"
        assert proposal.estimated_risk == 0.10
        assert proposal.estimated_compute_cost == 0.50
        assert proposal.estimated_time == 5.0
        assert proposal.memory_context == memory

    def test_different_agents_different_scaling(self, incident):
        """Test that agents scale independently based on their own history."""
        from self_healing_pipeline.agents.retrain import RetrainAgent

        # Threshold agent with poor history
        threshold_stats = AgentStats(
            agent_type="threshold",
            attempts=5,
            successes=0,
            success_rate=0.0,
            recent_success_rate=0.0,
            avg_business_savings=0.0,
            avg_duration=5.0,
        )
        # Retrain agent with good history
        retrain_stats = AgentStats(
            agent_type="retrain",
            attempts=5,
            successes=5,
            success_rate=1.0,
            recent_success_rate=1.0,
            avg_business_savings=8000.0,
            avg_duration=180.0,
        )
        memory = {
            "agents": {
                "threshold": threshold_stats,
                "retrain": retrain_stats,
            }
        }

        threshold_agent = ThresholdAgent("t-1")
        retrain_agent = RetrainAgent("r-1")

        t_proposal = threshold_agent._analyze_heuristic(incident, memory)
        r_proposal = retrain_agent._analyze_heuristic(incident, memory)

        # Threshold should be scaled down (poor history)
        # Retrain should be unscaled or scaled up (good history)
        assert t_proposal.confidence < r_proposal.confidence
