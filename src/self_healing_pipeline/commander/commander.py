from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Any

from self_healing_pipeline.agents.base import Agent, ExecutionResult, Proposal
from self_healing_pipeline.gateway.events import Incident

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CommanderResult:
    incident_id: str
    winning_agent_type: str
    winning_proposal: dict[str, Any]
    execution_result: dict[str, Any]
    all_proposals: list[dict[str, Any]]
    scoring_breakdown: list[dict[str, Any]]
    reconciliation_triggered: bool = False
    fallback_used: bool = False


class Commander:
    def __init__(self, agents: list[Agent]) -> None:
        self.agents = agents
        self.last_scoring: list[tuple[Proposal, float]] = []
        self.last_reconciliation: dict[str, Any] | None = None

    def score_proposals(
        self, proposals: list[Proposal], incident: Incident
    ) -> list[Proposal]:
        weights = {
            "business_value": 0.30,
            "confidence": 0.20,
            "risk_inverse": 0.20,
            "cost_efficiency": 0.10,
            "time_inverse": 0.05,
            "historical_success": 0.15,
        }

        scored = []
        for p in proposals:
            hist_score = 0.5
            business_norm = min(p.estimated_business_savings / 10000.0, 1.0)
            cost_ratio = min(
                p.estimated_business_savings / max(p.estimated_compute_cost, 0.01) / 100.0,
                1.0,
            )
            time_norm = 1.0 - min(p.estimated_time / 600.0, 1.0)

            p.score = (
                weights["business_value"] * business_norm
                + weights["confidence"] * p.confidence
                + weights["risk_inverse"] * (1.0 - p.estimated_risk)
                + weights["cost_efficiency"] * cost_ratio
                + weights["time_inverse"] * time_norm
                + weights["historical_success"] * hist_score
            )
            scored.append(p)

        self.last_scoring = [(p, p.score) for p in scored]
        return sorted(scored, key=lambda p: p.score, reverse=True)

    def needs_reconciliation(self, scored: list[Proposal]) -> bool:
        if len(scored) < 2:
            return False
        if scored[0].score == 0:
            return False
        margin = abs(scored[0].score - scored[1].score) / scored[0].score
        return margin < 0.10

    async def execute_with_timeout(
        self, proposal: Proposal, incident: Incident, timeout: float = 600.0
    ) -> ExecutionResult:
        agent = next((a for a in self.agents if a.agent_id == proposal.agent_id), None)
        if agent is None:
            return ExecutionResult(
                success=False,
                actual_business_savings=0.0,
                duration=0.0,
                error=f"agent {proposal.agent_id} not found",
            )

        try:
            result = await asyncio.wait_for(
                agent.execute(proposal, incident), timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                actual_business_savings=0.0,
                duration=timeout,
                error=f"execution timeout after {timeout}s",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                actual_business_savings=0.0,
                duration=0.0,
                error=str(e),
            )

    async def handle_incident(
        self, incident: Incident
    ) -> CommanderResult:
        eligible = [a for a in self.agents if a.can_handle(incident)]
        if not eligible:
            logger.warning(
                f"No agents eligible for incident {incident.id} (type={incident.type.value})"
            )
            return CommanderResult(
                incident_id=incident.id,
                winning_agent_type="none",
                winning_proposal={},
                execution_result={},
                all_proposals=[],
                scoring_breakdown=[],
            )

        proposals = [await a.analyze(incident, {}) for a in eligible]
        scored = self.score_proposals(proposals, incident)

        reconciliation_triggered = self.needs_reconciliation(scored)
        if reconciliation_triggered:
            logger.info(
                f"Reconciliation triggered for incident {incident.id} "
                f"(top 2 scores: {scored[0].score:.3f}, {scored[1].score:.3f})"
            )

        winner = scored[0]
        execution = await self.execute_with_timeout(winner, incident)

        fallback_used = False
        if not execution.success:
            logger.warning(
                f"Winner {winner.agent_type} failed: {execution.error}. Trying next best."
            )
            for backup in scored[1:]:
                execution = await self.execute_with_timeout(backup, incident)
                if execution.success:
                    fallback_used = True
                    winner = backup
                    break

        return CommanderResult(
            incident_id=incident.id,
            winning_agent_type=winner.agent_type,
            winning_proposal=asdict(winner),
            execution_result=asdict(execution),
            all_proposals=[asdict(p) for p in proposals],
            scoring_breakdown=[
                {
                    "agent_type": p.agent_type,
                    "score": p.score,
                    "degraded": p.degraded,
                    "confidence": p.confidence,
                    "savings": p.estimated_business_savings,
                }
                for p in scored
            ],
            reconciliation_triggered=reconciliation_triggered,
            fallback_used=fallback_used,
        )
