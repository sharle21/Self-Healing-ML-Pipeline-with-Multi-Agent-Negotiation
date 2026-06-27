"""Reconciliation debate: LangGraph-based agent negotiation for close calls.

When top 2 proposals are within 10% score margin, run a debate to pick the better one.
Uses Sonnet 4.6 when API key available, otherwise heuristic debate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic

from self_healing_pipeline.agents.base import Proposal
from self_healing_pipeline.config import get_settings
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
        """Sonnet 4.6 debate via multi-turn conversation.

        Moderator asks each agent to defend their proposal, then decides winner.
        """
        settings = get_settings()
        if not settings.anthropic_api_key:
            logger.warning("No API key; falling back to heuristic debate")
            return self._debate_heuristic(top1, top2, incident)

        try:
            client = Anthropic(api_key=settings.anthropic_api_key)
            debate_log: list[str] = []

            # System prompt for moderator
            system_prompt = """You are a fair moderator judging a debate between two ML incident response proposals.

Two agents have proposed different solutions to fix an ML system failure. Your job is to evaluate their arguments and declare a winner based on:
1. Feasibility (can it actually be executed?)
2. Impact (how much will it improve the system?)
3. Risk (how likely is it to cause problems?)
4. Cost (resource usage)
5. Speed (time to resolution)

Be concise. Make a clear decision."""

            # Opening: describe incident and proposals
            opening = f"""
Incident: {incident.type.value} (severity={incident.severity:.1f})
Payload: {json.dumps(incident.payload)}

Agent 1 ({top1.agent_type}):
- Estimated savings: ${top1.estimated_business_savings:.0f}
- Risk: {top1.estimated_risk:.2f}
- Time: {top1.estimated_time:.1f}s
- Cost: ${top1.estimated_compute_cost:.1f}
- Confidence: {top1.confidence:.2f}
- Rationale: {top1.rationale}

Agent 2 ({top2.agent_type}):
- Estimated savings: ${top2.estimated_business_savings:.0f}
- Risk: {top2.estimated_risk:.2f}
- Time: {top2.estimated_time:.1f}s
- Cost: ${top2.estimated_compute_cost:.1f}
- Confidence: {top2.confidence:.2f}
- Rationale: {top2.rationale}

Which agent's proposal is better overall? Declare the winner and your reasoning in 1-2 sentences."""

            debate_log.append(f"Moderator: {opening[:100]}...")

            # Call Sonnet for debate
            response = client.messages.create(
                model=self.model_name or "claude-sonnet-4-6",
                max_tokens=300,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": opening}
                ],
            )

            debate_text = response.content[0].text
            debate_log.append(f"Sonnet decision: {debate_text}")

            # Parse decision: which agent won?
            winner_type = top1.agent_type
            confidence = 0.7
            if top2.agent_type.lower() in debate_text.lower():
                winner_type = top2.agent_type
            if top1.agent_type.lower() in debate_text.lower() and winner_type == top2.agent_type:
                # Both mentioned, use heuristic tiebreaker
                winner_type = top1.agent_type if top1.confidence >= top2.confidence else top2.agent_type

            winner = top1 if winner_type == top1.agent_type else top2
            confidence = winner.confidence

            logger.info(f"Sonnet reconciliation: {winner_type} wins")

            return ReconciliationResult(
                winner_type=winner_type,
                rationale=debate_text,
                confidence=confidence,
                debate_log=debate_log,
            )

        except Exception as e:
            logger.warning(f"LLM debate failed ({e}); falling back to heuristic")
            return self._debate_heuristic(top1, top2, incident)
