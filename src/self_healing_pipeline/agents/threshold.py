from __future__ import annotations

import asyncio
from typing import Any

from self_healing_pipeline.agents.base import Agent, ExecutionResult, Proposal
from self_healing_pipeline.gateway.events import Incident, IncidentType


class ThresholdAgent(Agent):
    agent_type = "threshold"

    _ELIGIBLE = frozenset(
        {IncidentType.DRIFT, IncidentType.COST_THRESHOLD, IncidentType.LATENCY_BREACH}
    )

    def can_handle(self, incident: Incident) -> bool:
        return incident.type in self._ELIGIBLE

    async def analyze(
        self, incident: Incident, memory_context: dict[str, Any]
    ) -> Proposal:
        sev = max(0.0, min(1.0, incident.severity))
        confidence = 0.55 + 0.25 * sev
        savings = 1500.0 * sev
        rationale = (
            f"Adjust per-tenant decision threshold for {incident.tenant_id} "
            f"in response to {incident.type.value} (severity={sev:.2f}). "
            "Cheap, instant, low-risk."
        )
        return Proposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=confidence,
            estimated_business_savings=savings,
            estimated_risk=0.10,
            estimated_compute_cost=0.50,
            estimated_time=5.0,
            rationale=rationale,
            memory_context=memory_context,
        )

    async def execute(
        self, proposal: Proposal, incident: Incident
    ) -> ExecutionResult:
        await asyncio.sleep(0.01)
        return ExecutionResult(
            success=True,
            actual_business_savings=proposal.estimated_business_savings * 0.95,
            duration=proposal.estimated_time,
            logs=[f"threshold adjusted for tenant={incident.tenant_id}"],
        )
