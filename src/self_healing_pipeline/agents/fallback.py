from __future__ import annotations

import asyncio
from typing import Any

from self_healing_pipeline.agents.base import Agent, ExecutionResult, Proposal
from self_healing_pipeline.gateway.events import Incident, IncidentType


class FallbackAgent(Agent):
    """Use simpler model or default logic — safe, but degraded UX."""

    agent_type = "fallback"

    _ELIGIBLE = frozenset({IncidentType.DATA_QUALITY, IncidentType.COST_THRESHOLD})

    def can_handle(self, incident: Incident) -> bool:
        return incident.type in self._ELIGIBLE

    async def analyze(
        self, incident: Incident, memory_context: dict[str, Any]
    ) -> Proposal:
        sev = max(0.0, min(1.0, incident.severity))
        confidence = 0.50 + 0.15 * sev
        savings = 500.0 * sev
        rationale = (
            f"Switch to fallback logic for {incident.tenant_id} "
            f"({incident.type.value}, severity={sev:.2f}). "
            "Prevents errors, but UX degraded until root cause fixed."
        )
        return Proposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=confidence,
            estimated_business_savings=savings,
            estimated_risk=0.20,
            estimated_compute_cost=0.10,
            estimated_time=2.0,
            rationale=rationale,
            memory_context=memory_context,
        )

    async def execute(
        self, proposal: Proposal, incident: Incident
    ) -> ExecutionResult:
        await asyncio.sleep(0.01)
        return ExecutionResult(
            success=True,
            actual_business_savings=proposal.estimated_business_savings * 0.80,
            duration=proposal.estimated_time,
            logs=[f"fallback logic activated for tenant={incident.tenant_id}"],
        )
