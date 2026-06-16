from __future__ import annotations

import asyncio
from typing import Any

from self_healing_pipeline.agents.base import Agent, ExecutionResult, Proposal
from self_healing_pipeline.gateway.events import Incident, IncidentType


class DataRepairAgent(Agent):
    """Fix data quality issues — expensive, but long-term win."""

    agent_type = "data_repair"

    _ELIGIBLE = frozenset({IncidentType.DATA_QUALITY})

    def can_handle(self, incident: Incident) -> bool:
        return incident.type in self._ELIGIBLE

    async def analyze(
        self, incident: Incident, memory_context: dict[str, Any]
    ) -> Proposal:
        sev = max(0.0, min(1.0, incident.severity))
        confidence = 0.40 + 0.40 * sev
        savings = 5000.0 * sev
        affected = ", ".join(incident.affected_features) or "all features"
        rationale = (
            f"Repair data quality issues for {incident.tenant_id} "
            f"(severity={sev:.2f}, affected={affected}). "
            "High cost but durable fix; prevents future incidents."
        )
        return Proposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=confidence,
            estimated_business_savings=savings,
            estimated_risk=0.25,
            estimated_compute_cost=100.0,
            estimated_time=300.0,
            rationale=rationale,
            memory_context=memory_context,
        )

    async def execute(
        self, proposal: Proposal, incident: Incident
    ) -> ExecutionResult:
        await asyncio.sleep(0.01)
        return ExecutionResult(
            success=True,
            actual_business_savings=proposal.estimated_business_savings * 0.88,
            duration=proposal.estimated_time,
            logs=[f"data repair completed for tenant={incident.tenant_id}"],
        )
