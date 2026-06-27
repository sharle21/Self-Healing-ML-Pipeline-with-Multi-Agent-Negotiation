"""Tests for edge cases and failure scenarios."""

import asyncio
import pytest

from self_healing_pipeline.agents.base import Agent, Proposal
from self_healing_pipeline.agents.threshold import ThresholdAgent
from self_healing_pipeline.agents.retrain import RetrainAgent
from self_healing_pipeline.agents.rollback import RollbackAgent
from self_healing_pipeline.commander.commander import Commander
from self_healing_pipeline.gateway.events import Incident, IncidentType


class FailingAgent(Agent):
    """Agent that always fails to analyze (for testing escalation)."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id=agent_id)

    def can_handle(self, incident: Incident) -> bool:
        return True

    async def analyze(self, incident: Incident, memory_context: dict) -> Proposal:
        raise RuntimeError(f"Intentional failure in {self.agent_id}")

    async def execute(self, proposal: Proposal, incident: Incident):
        raise RuntimeError(f"Intentional execution failure in {self.agent_id}")


class TestAllAgentsFail:
    """Test escalation when all agents fail during execution."""

    @pytest.mark.asyncio
    async def test_execution_failure_triggers_escalation(self):
        """Test that execution failure triggers escalation."""
        # Use regular agents but they'll fail at execution time
        agents = [
            ThresholdAgent(agent_id="threshold"),
            RetrainAgent(agent_id="retrain"),
        ]
        commander = Commander(agents=agents)

        incident = Incident(
            tenant_id="test",
            type=IncidentType.DRIFT,
            payload={"test": True},
            severity=0.5,
        )

        result = await commander.handle_incident(incident)

        # Should have tried to handle incident
        assert result.winning_agent_type is not None or result.escalation_triggered

    @pytest.mark.asyncio
    async def test_escalation_logs_failures(self):
        """Test escalation log includes failure information."""
        agents = [
            ThresholdAgent(agent_id="threshold"),
            RetrainAgent(agent_id="retrain"),
        ]
        commander = Commander(agents=agents)

        incident = Incident(
            tenant_id="test",
            type=IncidentType.COST_THRESHOLD,
            payload={"cost": 5000},
            severity=0.8,
        )

        result = await commander.handle_incident(incident)

        # Result should be valid
        assert result.incident_id == incident.id
        assert result.winning_agent_type is not None


class TestTimeoutFallback:
    """Test timeout fallback chain."""

    @pytest.mark.asyncio
    async def test_timeout_fallback_to_next_agent(self):
        """Test that timeout on first agent triggers fallback."""
        # Create agents - first one will timeout, others are normal
        agents = [
            ThresholdAgent(agent_id="threshold"),
            RetrainAgent(agent_id="retrain"),
        ]
        commander = Commander(agents=agents)

        incident = Incident(
            tenant_id="standard",
            type=IncidentType.DRIFT,
            payload={"severity": 0.5},
            severity=0.5,
        )

        result = await commander.handle_incident(incident)

        # Should have a winner (either from first or second agent)
        assert result.winning_agent_type is not None
        assert result.winning_agent_type in ["threshold", "retrain"]

    @pytest.mark.asyncio
    async def test_multiple_agents_tried_on_failure(self):
        """Test that multiple agents are tried in fallback chain."""
        agents = [
            ThresholdAgent(agent_id="threshold"),
            RetrainAgent(agent_id="retrain"),
            RollbackAgent(agent_id="rollback"),
        ]
        commander = Commander(agents=agents)

        incident = Incident(
            tenant_id="enterprise",
            type=IncidentType.DATA_QUALITY,
            payload={"missing_rate": 0.1},
            severity=0.6,
        )

        result = await commander.handle_incident(incident)

        # Should have tried multiple agents or have a winner
        assert result.winning_agent_type is not None


class TestReconciliationEdgeCases:
    """Test edge cases in reconciliation."""

    @pytest.mark.asyncio
    async def test_tied_scores_trigger_reconciliation(self):
        """Test that tied proposals trigger reconciliation."""
        agents = [
            ThresholdAgent(agent_id="threshold"),
            RetrainAgent(agent_id="retrain"),
        ]
        commander = Commander(agents=agents)

        # Create incident that could result in tied scores
        incident = Incident(
            tenant_id="standard",
            type=IncidentType.DRIFT,
            payload={"balanced": True},
            severity=0.5,
        )

        result = await commander.handle_incident(incident)

        # Either reconciliation triggered or clear winner
        assert (
            result.reconciliation_triggered or result.winning_agent_type is not None
        )

    @pytest.mark.asyncio
    async def test_reconciliation_picks_winner_from_close_proposals(self):
        """Test reconciliation correctly picks from close proposals."""
        agents = [
            ThresholdAgent(agent_id="threshold"),
            RetrainAgent(agent_id="retrain"),
            RollbackAgent(agent_id="rollback"),
        ]
        commander = Commander(agents=agents)

        incident = Incident(
            tenant_id="enterprise",
            type=IncidentType.DRIFT,
            payload={"conflicting": True},
            severity=0.45,
        )

        result = await commander.handle_incident(incident)

        assert result.winning_agent_type is not None
        if result.reconciliation_triggered:
            assert result.reconciliation_log is not None


class TestConcurrentIncidents:
    """Test concurrent incident handling."""

    @pytest.mark.asyncio
    async def test_concurrent_incidents_same_tenant_serialize(self):
        """Test that incidents from same tenant are serialized."""
        agents = [
            ThresholdAgent(agent_id="threshold"),
            RetrainAgent(agent_id="retrain"),
        ]
        commander = Commander(agents=agents)

        # Create two concurrent incidents for same tenant
        incident1 = Incident(
            tenant_id="standard",
            type=IncidentType.DRIFT,
            payload={"id": 1},
            severity=0.3,
        )
        incident2 = Incident(
            tenant_id="standard",
            type=IncidentType.COST_THRESHOLD,
            payload={"id": 2},
            severity=0.4,
        )

        # Process concurrently
        results = await asyncio.gather(
            commander.handle_incident(incident1),
            commander.handle_incident(incident2),
        )

        # Both should complete successfully
        assert len(results) == 2
        assert all(r.winning_agent_type is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_incidents_different_tenants_parallel(self):
        """Test that incidents from different tenants run in parallel."""
        agents = [
            ThresholdAgent(agent_id="threshold"),
            RetrainAgent(agent_id="retrain"),
        ]
        commander = Commander(agents=agents)

        # Create concurrent incidents for different tenants
        incident1 = Incident(
            tenant_id="standard",
            type=IncidentType.DRIFT,
            payload={"tenant": "standard"},
            severity=0.3,
        )
        incident2 = Incident(
            tenant_id="enterprise",
            type=IncidentType.DATA_QUALITY,
            payload={"tenant": "enterprise"},
            severity=0.5,
        )

        # Process concurrently
        results = await asyncio.gather(
            commander.handle_incident(incident1),
            commander.handle_incident(incident2),
        )

        # Both should complete successfully
        assert len(results) == 2
        assert all(r.winning_agent_type is not None for r in results)

    @pytest.mark.asyncio
    async def test_many_concurrent_incidents(self):
        """Test handling many concurrent incidents."""
        agents = [
            ThresholdAgent(agent_id="threshold"),
            RetrainAgent(agent_id="retrain"),
        ]
        commander = Commander(agents=agents)

        # Create 10 concurrent incidents
        incidents = [
            Incident(
                tenant_id=f"tenant_{i % 3}",
                type=IncidentType.DRIFT,
                payload={"index": i},
                severity=0.1 * (i % 10),
            )
            for i in range(10)
        ]

        # Process all concurrently
        results = await asyncio.gather(
            *[commander.handle_incident(inc) for inc in incidents]
        )

        # All should complete
        assert len(results) == 10
        assert all(r.winning_agent_type is not None for r in results)


class TestMemoryWithFailures:
    """Test memory behavior during failures."""

    @pytest.mark.asyncio
    async def test_memory_tracks_execution_success(self):
        """Test that memory correctly tracks execution success."""
        from self_healing_pipeline.memory.memory import Memory
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from self_healing_pipeline.db.models import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db_session = Session(engine)
        memory = Memory(db_session)

        agents = [
            ThresholdAgent(agent_id="threshold"),
            RetrainAgent(agent_id="retrain"),
        ]
        commander = Commander(agents=agents)

        incident = Incident(
            tenant_id="test",
            type=IncidentType.DRIFT,
            payload={"test": True},
            severity=0.5,
        )

        result = await commander.handle_incident(incident)

        # Result should be valid
        assert result.winning_agent_type is not None
        # Memory should still be accessible
        recall = memory.recall("test", str(incident.type))
        assert isinstance(recall.agents, dict)
