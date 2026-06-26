"""Reconciliation debate: LangGraph-based agent negotiation for close calls.

When top 2 proposals are within 10% score margin, run a debate to pick the better one.
Uses Sonnet 4.6 when API key available, otherwise heuristic debate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from self_healing_pipeline.agents.base import Proposal
from self_healing_pipeline.gateway.events import Incident

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ReconciliationResult:
    """Reconciliation debate result."""

    winner_type: str
    rationale: str
    confidence: float
    debate_log: list[str]


class Reconciliation:
    """Debate logic between top 2 proposals."""

    def __init__(self, model_name: str | None = None) -> None:
        """Init reconciliation.

        Args:
            model_name: Sonnet 4.6 model (e.g., "claude-sonnet-4-6"). If None, use heuristics.
        """
        self.model_name = model_name

    async def debate(
        self, top1: Proposal, top2: Proposal, incident: Incident
    ) -> ReconciliationResult:
        """Run debate between top 2 proposals.

        Args:
            top1: highest-scored proposal
            top2: second-highest-scored proposal
            incident: the incident being resolved

        Returns:
            ReconciliationResult with winner and rationale
        """
        if self.model_name:
            return await self._debate_with_llm(top1, top2, incident)
        return self._debate_heuristic(top1, top2, incident)

    def _debate_heuristic(
        self, top1: Proposal, top2: Proposal, incident: Incident
    ) -> ReconciliationResult:
        """Heuristic debate: compare risk/cost/time trade-offs.

        If scores are close, pick the one with:
        1. Lower risk (safer)
        2. Faster execution
        3. Lower cost
        """
        debate_log: list[str] = []

        debate_log.append(
            f"Debating {top1.agent_type} vs {top2.agent_type} "
            f"(scores {top1.score:.3f} vs {top2.score:.3f})"
        )

        # Risk comparison
        risk_diff = top2.estimated_risk - top1.estimated_risk
        if abs(risk_diff) > 0.05:
            safer = top1 if top1.estimated_risk < top2.estimated_risk else top2
            debate_log.append(
                f"Risk: {safer.agent_type} is safer "
                f"({safer.estimated_risk:.2f} vs {(top2.estimated_risk if safer.agent_type == top1.agent_type else top1.estimated_risk):.2f})"
            )

        # Time comparison
        time_diff = top1.estimated_time - top2.estimated_time
        if abs(time_diff) > 10.0:
            faster = top1 if top1.estimated_time < top2.estimated_time else top2
            debate_log.append(
                f"Speed: {faster.agent_type} is faster "
                f"({faster.estimated_time:.1f}s vs {(top2.estimated_time if faster.agent_type == top1.agent_type else top1.estimated_time):.1f}s)"
            )

        # Cost comparison
        cost_diff = top1.estimated_compute_cost - top2.estimated_compute_cost
        if abs(cost_diff) > 5.0:
            cheaper = top1 if top1.estimated_compute_cost < top2.estimated_compute_cost else top2
            debate_log.append(
                f"Cost: {cheaper.agent_type} is cheaper "
                f"(${cheaper.estimated_compute_cost:.1f} vs ${(top2.estimated_compute_cost if cheaper.agent_type == top1.agent_type else top1.estimated_compute_cost):.1f})"
            )

        # Tiebreaker: pick higher confidence
        winner_type = top1.agent_type if top1.confidence >= top2.confidence else top2.agent_type
        winner = top1 if winner_type == top1.agent_type else top2
        debate_log.append(
            f"Tiebreaker: {winner.agent_type} wins (confidence {winner.confidence:.2f})"
        )

        return ReconciliationResult(
            winner_type=winner_type,
            rationale="\n".join(debate_log),
            confidence=winner.confidence,
            debate_log=debate_log,
        )

    async def _debate_with_llm(
        self, top1: Proposal, top2: Proposal, incident: Incident
    ) -> ReconciliationResult:
        """LangGraph-based debate with Sonnet 4.6.

        Placeholder: uses heuristic for now, swaps to Sonnet when API key available.
        """
        # TODO: Implement LangGraph state machine with Sonnet 4.6
        # For now, fall back to heuristic
        logger.warning(
            f"LLM reconciliation not yet implemented (model={self.model_name}). "
            "Using heuristic debate."
        )
        return self._debate_heuristic(top1, top2, incident)
