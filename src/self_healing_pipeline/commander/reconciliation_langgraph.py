"""Multi-turn LangGraph-based reconciliation debate."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic
from langgraph.graph import StateGraph
from pydantic import BaseModel

from self_healing_pipeline.agents.base import Proposal
from self_healing_pipeline.config import get_settings
from self_healing_pipeline.gateway.events import Incident

logger = logging.getLogger(__name__)


class DebateState(BaseModel):
    """State for multi-turn debate."""

    incident: dict[str, Any]
    top1_proposal: dict[str, Any]
    top2_proposal: dict[str, Any]
    debate_history: list[dict[str, str]] = field(default_factory=list)
    agent1_defense: str = ""
    agent2_defense: str = ""
    moderator_decision: str = ""
    winner_type: str = ""
    confidence: float = 0.0


@dataclass(slots=True)
class ReconciliationResult:
    """Reconciliation debate result."""

    winner_type: str
    rationale: str
    confidence: float
    debate_log: list[str]


class LangGraphReconciliation:
    """Multi-turn debate using LangGraph state graph."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name
        self.client = Anthropic()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph state graph for debate."""
        graph = StateGraph(DebateState)

        # Add nodes
        graph.add_node("moderator_opening", self._moderator_opening)
        graph.add_node("agent1_defense", self._agent1_defense)
        graph.add_node("agent2_defense", self._agent2_defense)
        graph.add_node("moderator_decision", self._moderator_decision)

        # Add edges
        graph.add_edge("moderator_opening", "agent1_defense")
        graph.add_edge("agent1_defense", "agent2_defense")
        graph.add_edge("agent2_defense", "moderator_decision")
        graph.set_entry_point("moderator_opening")
        graph.set_finish_point("moderator_decision")

        return graph

    def _moderator_opening(self, state: DebateState) -> DebateState:
        """Moderator describes incident and proposals."""
        incident = state.incident
        top1 = state.top1_proposal
        top2 = state.top2_proposal

        opening = f"""
Incident: {incident['type']} (severity={incident['severity']:.1f})
Tenant: {incident['tenant_id']}
Payload: {json.dumps(incident['payload'])}

Agent 1 ({top1['agent_type']}):
- Savings: ${top1['estimated_business_savings']:.0f}
- Confidence: {top1['confidence']:.2f}
- Risk: {top1['estimated_risk']:.2f}
- Cost: ${top1['estimated_compute_cost']:.1f}
- Time: {top1['estimated_time']:.0f}s

Agent 2 ({top2['agent_type']}):
- Savings: ${top2['estimated_business_savings']:.0f}
- Confidence: {top2['confidence']:.2f}
- Risk: {top2['estimated_risk']:.2f}
- Cost: ${top2['estimated_compute_cost']:.1f}
- Time: {top2['estimated_time']:.0f}s

Agent 1, defend your proposal."""

        state.debate_history.append({"role": "moderator", "content": opening})
        return state

    def _agent1_defense(self, state: DebateState) -> DebateState:
        """Agent 1 defends its proposal."""
        top1 = state.top1_proposal
        incident = state.incident

        prompt = f"""You are {top1['agent_type']} proposing to fix a {incident['type']} incident.

Your proposal:
- Estimated savings: ${top1['estimated_business_savings']:.0f}
- Confidence: {top1['confidence']:.2f}
- Risk: {top1['estimated_risk']:.2f}
- Rationale: {top1['rationale']}

Defend your approach in 1-2 sentences. Focus on why your solution is best for this incident."""

        try:
            response = self.client.messages.create(
                model=self.model_name or "claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            defense = response.content[0].text
            state.agent1_defense = defense
            state.debate_history.append({"role": "agent1", "content": defense})
        except Exception as e:
            logger.warning(f"Agent1 defense failed: {e}")
            state.agent1_defense = state.top1_proposal["rationale"]
            state.debate_history.append(
                {"role": "agent1", "content": state.top1_proposal["rationale"]}
            )

        return state

    def _agent2_defense(self, state: DebateState) -> DebateState:
        """Agent 2 counters Agent 1's proposal."""
        top2 = state.top2_proposal
        top1 = state.top1_proposal
        incident = state.incident

        prompt = f"""You are {top2['agent_type']} proposing an alternative fix for a {incident['type']} incident.

Your proposal:
- Estimated savings: ${top2['estimated_business_savings']:.0f}
- Confidence: {top2['confidence']:.2f}
- Risk: {top2['estimated_risk']:.2f}
- Rationale: {top2['rationale']}

{top1['agent_type']} just argued: "{state.agent1_defense}"

Counter their argument and explain why YOUR approach is better. Focus on risk, cost, or confidence trade-offs. 1-2 sentences."""

        try:
            response = self.client.messages.create(
                model=self.model_name or "claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            defense = response.content[0].text
            state.agent2_defense = defense
            state.debate_history.append({"role": "agent2", "content": defense})
        except Exception as e:
            logger.warning(f"Agent2 defense failed: {e}")
            state.agent2_defense = state.top2_proposal["rationale"]
            state.debate_history.append(
                {"role": "agent2", "content": state.top2_proposal["rationale"]}
            )

        return state

    def _moderator_decision(self, state: DebateState) -> DebateState:
        """Moderator makes final decision."""
        top1 = state.top1_proposal
        top2 = state.top2_proposal

        prompt = f"""You are a fair moderator. {top1['agent_type']} and {top2['agent_type']} debated fixing an incident.

Agent 1 ({top1['agent_type']}): "{state.agent1_defense}"
Agent 2 ({top2['agent_type']}): "{state.agent2_defense}"

Criteria:
1. Feasibility (can it be executed?)
2. Impact (savings + confidence)
3. Risk (how safe is it?)
4. Cost efficiency (savings/cost ratio)
5. Speed (time to resolution)

Declare ONLY the winner name (e.g., "threshold" or "retrain") and your reasoning in 1-2 sentences."""

        try:
            response = self.client.messages.create(
                model=self.model_name or "claude-sonnet-4-6",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            decision = response.content[0].text
            state.moderator_decision = decision
            state.debate_history.append({"role": "moderator_decision", "content": decision})

            # Parse winner from response
            if top2["agent_type"].lower() in decision.lower():
                state.winner_type = top2["agent_type"]
                state.confidence = top2["confidence"]
            else:
                state.winner_type = top1["agent_type"]
                state.confidence = top1["confidence"]

        except Exception as e:
            logger.warning(f"Moderator decision failed: {e}")
            state.winner_type = top1["agent_type"]
            state.confidence = top1["confidence"]
            state.moderator_decision = f"Default to {top1['agent_type']} (API error)"
            state.debate_history.append(
                {"role": "moderator_decision", "content": state.moderator_decision}
            )

        return state

    async def debate(
        self, top1: Proposal, top2: Proposal, incident: Incident
    ) -> ReconciliationResult:
        """Run multi-turn debate.

        Args:
            top1: highest-scored proposal
            top2: second-highest-scored proposal
            incident: the incident being resolved

        Returns:
            ReconciliationResult with winner and debate log
        """
        settings = get_settings()
        if not settings.anthropic_api_key:
            logger.warning("No API key; using heuristic debate")
            return self._heuristic_fallback(top1, top2)

        try:
            # Build and compile graph
            graph = self._build_graph()
            compiled_graph = graph.compile()

            # Initialize state
            state = DebateState(
                incident={
                    "type": incident.type.value,
                    "tenant_id": incident.tenant_id,
                    "severity": incident.severity,
                    "payload": incident.payload,
                },
                top1_proposal={
                    "agent_type": top1.agent_type,
                    "confidence": top1.confidence,
                    "estimated_business_savings": top1.estimated_business_savings,
                    "estimated_risk": top1.estimated_risk,
                    "estimated_compute_cost": top1.estimated_compute_cost,
                    "estimated_time": top1.estimated_time,
                    "rationale": top1.rationale,
                },
                top2_proposal={
                    "agent_type": top2.agent_type,
                    "confidence": top2.confidence,
                    "estimated_business_savings": top2.estimated_business_savings,
                    "estimated_risk": top2.estimated_risk,
                    "estimated_compute_cost": top2.estimated_compute_cost,
                    "estimated_time": top2.estimated_time,
                    "rationale": top2.rationale,
                },
            )

            # Run debate
            final_state = compiled_graph.invoke(state)

            debate_log = [
                f"{entry['role']}: {entry['content'][:100]}..."
                if len(entry["content"]) > 100
                else f"{entry['role']}: {entry['content']}"
                for entry in final_state.debate_history
            ]

            logger.info(
                f"LangGraph debate: {final_state.winner_type} wins "
                f"(confidence={final_state.confidence:.2f})"
            )

            return ReconciliationResult(
                winner_type=final_state.winner_type,
                rationale=final_state.moderator_decision,
                confidence=final_state.confidence,
                debate_log=debate_log,
            )

        except Exception as e:
            logger.warning(f"LangGraph debate failed: {e}; using heuristic")
            return self._heuristic_fallback(top1, top2)

    def _heuristic_fallback(self, top1: Proposal, top2: Proposal) -> ReconciliationResult:
        """Fallback to heuristic when API fails."""
        # Risk comparison
        if abs(top2.estimated_risk - top1.estimated_risk) > 0.05:
            safer = top1 if top1.estimated_risk < top2.estimated_risk else top2
            winner = safer
            reason = f"{safer.agent_type} is safer (risk {safer.estimated_risk:.2f})"
        # Confidence comparison
        elif top1.confidence > top2.confidence:
            winner = top1
            reason = f"{top1.agent_type} is more confident ({top1.confidence:.2f})"
        else:
            winner = top2
            reason = f"{top2.agent_type} is more confident ({top2.confidence:.2f})"

        return ReconciliationResult(
            winner_type=winner.agent_type,
            rationale=reason,
            confidence=winner.confidence,
            debate_log=[reason],
        )
