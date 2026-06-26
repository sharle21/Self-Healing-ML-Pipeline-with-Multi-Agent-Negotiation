"""Escalation handler: all agents failed, cannot resolve incident."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from self_healing_pipeline.agents.base import ExecutionResult, Proposal
from self_healing_pipeline.gateway.events import Incident


@dataclass(slots=True)
class EscalationResult:
    """Escalation result when all agents fail."""

    success: bool
    reason: str
    failed_attempts: list[dict[str, Any]]


class Escalation:
    """Handle incidents where all agents fail."""

    @staticmethod
    def escalate(
        incident: Incident,
        proposals: list[Proposal],
        execution_results: list[tuple[Proposal, ExecutionResult]],
    ) -> EscalationResult:
        """Escalate when all agents fail.

        Args:
            incident: the incident that couldn't be resolved
            proposals: all proposals that were tried
            execution_results: (proposal, result) for each failed attempt

        Returns:
            EscalationResult with escalation details
        """
        failed_attempts = [
            {
                "agent_type": prop.agent_type,
                "error": result.error,
                "duration": result.duration,
                "rationale": prop.rationale,
            }
            for prop, result in execution_results
        ]

        reason = (
            f"All {len(proposals)} agents failed to resolve "
            f"{incident.type.value} for tenant {incident.tenant_id}. "
            f"Escalation required."
        )

        return EscalationResult(
            success=False,
            reason=reason,
            failed_attempts=failed_attempts,
        )
