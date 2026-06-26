"""Tests for failure handling: timeouts, fallback, escalation."""

import pytest

from self_healing_pipeline.agents.base import ExecutionResult, Proposal
from self_healing_pipeline.commander.escalation import Escalation
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
def proposals():
    """Create 3 test proposals."""
    return [
        Proposal(
            agent_id="a1",
            agent_type="threshold",
            confidence=0.8,
            estimated_business_savings=1500.0,
            estimated_risk=0.10,
            estimated_compute_cost=0.5,
            estimated_time=5.0,
            rationale="Threshold",
        ),
        Proposal(
            agent_id="a2",
            agent_type="retrain",
            confidence=0.75,
            estimated_business_savings=8000.0,
            estimated_risk=0.30,
            estimated_compute_cost=50.0,
            estimated_time=180.0,
            rationale="Retrain",
        ),
        Proposal(
            agent_id="a3",
            agent_type="rollback",
            confidence=0.7,
            estimated_business_savings=2000.0,
            estimated_risk=0.05,
            estimated_compute_cost=2.0,
            estimated_time=15.0,
            rationale="Rollback",
        ),
    ]


class TestEscalation:
    """Escalation handler tests."""

    def test_escalate_single_failure(self, incident, proposals):
        """Test escalation with one failed attempt."""
        failure = ExecutionResult(
            success=False,
            actual_business_savings=0.0,
            duration=600.0,
            error="timeout after 600s",
        )
        attempts = [(proposals[0], failure)]

        result = Escalation.escalate(incident, proposals, attempts)
        assert result.success is False
        assert "All 3 agents failed" in result.reason
        assert len(result.failed_attempts) == 1
        assert result.failed_attempts[0]["agent_type"] == "threshold"
        assert "timeout" in result.failed_attempts[0]["error"]

    def test_escalate_all_failures(self, incident, proposals):
        """Test escalation when all agents fail."""
        failures = [
            ExecutionResult(
                success=False,
                actual_business_savings=0.0,
                duration=600.0,
                error="timeout",
            ),
            ExecutionResult(
                success=False,
                actual_business_savings=0.0,
                duration=30.0,
                error="memory exhausted",
            ),
            ExecutionResult(
                success=False,
                actual_business_savings=0.0,
                duration=15.0,
                error="rollback failed",
            ),
        ]
        attempts = list(zip(proposals, failures))

        result = Escalation.escalate(incident, proposals, attempts)
        assert result.success is False
        assert len(result.failed_attempts) == 3
        assert result.failed_attempts[0]["agent_type"] == "threshold"
        assert result.failed_attempts[1]["agent_type"] == "retrain"
        assert result.failed_attempts[2]["agent_type"] == "rollback"

    def test_escalate_preserves_rationales(self, incident, proposals):
        """Test that escalation preserves agent rationales."""
        failures = [
            ExecutionResult(success=False, actual_business_savings=0.0, duration=5.0, error="err1"),
            ExecutionResult(success=False, actual_business_savings=0.0, duration=5.0, error="err2"),
            ExecutionResult(success=False, actual_business_savings=0.0, duration=5.0, error="err3"),
        ]
        attempts = list(zip(proposals, failures))

        result = Escalation.escalate(incident, proposals, attempts)
        for i, attempt in enumerate(result.failed_attempts):
            assert attempt["rationale"] == proposals[i].rationale

    def test_escalate_mentions_tenant(self, incident, proposals):
        """Test that escalation mentions the tenant."""
        failure = ExecutionResult(
            success=False, actual_business_savings=0.0, duration=0.0, error="test"
        )
        attempts = [(proposals[0], failure)]

        result = Escalation.escalate(incident, proposals, attempts)
        assert "standard" in result.reason


class TestFailureHandling:
    """Failure handling integration tests."""

    def test_execution_result_failure_flag(self):
        """Test execution result failure flag."""
        success = ExecutionResult(
            success=True, actual_business_savings=1000.0, duration=5.0
        )
        failure = ExecutionResult(
            success=False, actual_business_savings=0.0, duration=600.0, error="timeout"
        )
        assert success.success is True
        assert failure.success is False
        assert failure.error == "timeout"

    def test_execution_result_with_logs(self):
        """Test execution result with detailed logs."""
        result = ExecutionResult(
            success=True,
            actual_business_savings=1500.0,
            duration=5.0,
            logs=["step 1", "step 2", "success"],
        )
        assert len(result.logs) == 3
        assert result.logs[0] == "step 1"

    def test_timeout_error_preserved(self):
        """Test that timeout errors are preserved in result."""
        result = ExecutionResult(
            success=False,
            actual_business_savings=0.0,
            duration=600.0,
            error="execution timeout after 600s",
        )
        assert "timeout" in result.error.lower()
        assert "600" in result.error
