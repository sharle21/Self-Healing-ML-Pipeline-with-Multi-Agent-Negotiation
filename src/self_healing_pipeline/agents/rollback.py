from __future__ import annotations

import asyncio
from typing import Any

from self_healing_pipeline.agents.base import Agent, ExecutionResult, Proposal
from self_healing_pipeline.gateway.events import Incident, IncidentType


class RollbackAgent(Agent):
    """Revert to previous model version — fast, safe, low payoff."""

    agent_type = "rollback"

    _ELIGIBLE = frozenset({IncidentType.DRIFT, IncidentType.DATA_QUALITY})

    def can_handle(self, incident: Incident) -> bool:
        return incident.type in self._ELIGIBLE

    def _analyze_heuristic(
        self, incident: Incident, memory_context: dict[str, Any]
    ) -> Proposal:
        sev = max(0.0, min(1.0, incident.severity))
        confidence = 0.60 + 0.20 * sev

        # Scale confidence by memory: recent_success_rate
        agents_context = memory_context.get("agents", {})
        if self.agent_type in agents_context:
            recent_rate = agents_context[self.agent_type].recent_success_rate
            confidence *= 0.5 + 0.5 * recent_rate

        savings = 2000.0 * sev
        rationale = (
            f"Rollback to previous model version for {incident.tenant_id} "
            f"({incident.type.value}, severity={sev:.2f}). "
            "Fast, safe, restores known-good behavior."
        )
        return Proposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=confidence,
            estimated_business_savings=savings,
            estimated_risk=0.05,
            estimated_compute_cost=2.0,
            estimated_time=15.0,
            rationale=rationale,
            memory_context=memory_context,
        )

    async def analyze(
        self, incident: Incident, memory_context: dict[str, Any]
    ) -> Proposal:
        return self._analyze_heuristic(incident, memory_context)

    async def execute(
        self, proposal: Proposal, incident: Incident
    ) -> ExecutionResult:
        await asyncio.sleep(0.01)
        return ExecutionResult(
            success=True,
            actual_business_savings=proposal.estimated_business_savings * 0.92,
            duration=proposal.estimated_time,
            logs=[f"rollback completed for tenant={incident.tenant_id}"],
        )
