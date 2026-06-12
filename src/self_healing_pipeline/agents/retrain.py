from __future__ import annotations

import asyncio
from typing import Any

from self_healing_pipeline.agents.base import Agent, ExecutionResult, Proposal
from self_healing_pipeline.gateway.events import Incident, IncidentType


class RetrainAgent(Agent):
    agent_type = "retrain"

    _ELIGIBLE = frozenset({IncidentType.DRIFT, IncidentType.DATA_QUALITY})

    def can_handle(self, incident: Incident) -> bool:
        return incident.type in self._ELIGIBLE

    async def analyze(
        self, incident: Incident, memory_context: dict[str, Any]
    ) -> Proposal:
        sev = max(0.0, min(1.0, incident.severity))
        confidence = 0.45 + 0.35 * sev
        savings = 8000.0 * sev
        affected = ", ".join(incident.affected_features) or "all features"
        rationale = (
            f"Refit model on recent data for {incident.tenant_id} "
            f"({incident.type.value}, severity={sev:.2f}, affected={affected}). "
            "Slow + costly but high payoff."
        )
        return Proposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=confidence,
            estimated_business_savings=savings,
            estimated_risk=0.30,
            estimated_compute_cost=50.0,
            estimated_time=180.0,
            rationale=rationale,
            memory_context=memory_context,
        )

    async def execute(
        self, proposal: Proposal, incident: Incident
    ) -> ExecutionResult:
        await asyncio.sleep(0.01)
        return ExecutionResult(
            success=True,
            actual_business_savings=proposal.estimated_business_savings * 0.90,
            duration=proposal.estimated_time,
            logs=[f"retrain completed for tenant={incident.tenant_id}"],
        )
