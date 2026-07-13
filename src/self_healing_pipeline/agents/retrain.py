from __future__ import annotations

import asyncio
import json
from typing import Any

from anthropic import Anthropic

from self_healing_pipeline.agents.base import Agent, ExecutionResult, Proposal
from self_healing_pipeline.gateway.events import Incident, IncidentType


class RetrainAgent(Agent):
    """Remediation Policy: Refit model on recent data to address distribution shift.

    Slow, expensive, high-confidence fix. Addresses root cause of drift. Good for sustained incidents.
    """
    agent_type = "retrain"

    _ELIGIBLE = frozenset({IncidentType.DRIFT, IncidentType.DATA_QUALITY})

    def can_handle(self, incident: Incident) -> bool:
        return incident.type in self._ELIGIBLE

    async def analyze(
        self, incident: Incident, memory_context: dict[str, Any]
    ) -> Proposal:
        if self.model_name:
            return await self._analyze_with_llm(incident, memory_context)
        return self._analyze_heuristic(incident, memory_context)

    def _analyze_heuristic(
        self, incident: Incident, memory_context: dict[str, Any]
    ) -> Proposal:
        sev = max(0.0, min(1.0, incident.severity))
        confidence = 0.45 + 0.35 * sev

        # Scale confidence by memory: recent_success_rate
        agents_context = memory_context.get("agents", {})
        if self.agent_type in agents_context:
            recent_rate = agents_context[self.agent_type].recent_success_rate
            confidence *= 0.5 + 0.5 * recent_rate

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

    async def _analyze_with_llm(
        self, incident: Incident, memory_context: dict[str, Any]
    ) -> Proposal:
        client = Anthropic()
        msg = f"""Analyze this incident and propose model retraining.

Tenant: {incident.tenant_id}
Type: {incident.type.value}
Severity: {incident.severity}
Affected features: {', '.join(incident.affected_features) or 'all'}
Memory context: {json.dumps(memory_context, default=str)}

Return JSON with: confidence (0-1), estimated_business_savings (number), estimated_risk (0-1),
estimated_compute_cost (number), estimated_time (seconds), rationale (string)."""

        try:
            response = client.messages.create(
                model=self.model_name,
                max_tokens=500,
                messages=[{"role": "user", "content": msg}],
            )
            data = json.loads(response.content[0].text)
        except Exception:
            proposal = self._analyze_heuristic(incident, memory_context)
            proposal.degraded = True
            return proposal

        return Proposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=float(data.get("confidence", 0.45)),
            estimated_business_savings=float(data.get("estimated_business_savings", 8000.0)),
            estimated_risk=float(data.get("estimated_risk", 0.3)),
            estimated_compute_cost=float(data.get("estimated_compute_cost", 50.0)),
            estimated_time=float(data.get("estimated_time", 180.0)),
            rationale=str(data.get("rationale", "Model retraining")),
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
