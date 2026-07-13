"""Remediation policy agent base class: state-based confidence + structured plans."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RemediationPlan:
    """Structured remediation plan from a remediation policy agent."""

    agent_type: str
    action: str
    confidence: float
    expected_effect: dict[str, Any]
    reasoning: str
    cost: str
    execution_time: str
    risk: float = 0.0


@dataclass(slots=True)
class ExecutionResult:
    """Result of remediation plan execution."""

    success: bool
    actual_improvement: dict[str, Any]
    duration: float
    error: str | None = None
    logs: list[str] = field(default_factory=list)


class RemediationPolicyAgent(ABC):
    """Base class for remediation policy agents.

    Each agent encapsulates a strategy for fixing ML incidents (threshold adjustment,
    retraining, rollback, fallback, data repair). Agents receive structured state,
    compute confidence from state features, and produce detailed remediation plans.
    """

    agent_type: str

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    @abstractmethod
    def can_handle(self, state: dict[str, Any]) -> bool:
        """Check if this agent can handle the incident state.

        Args:
            state: agent-specific state dict

        Returns:
            True if agent has a meaningful response
        """
        ...

    @abstractmethod
    async def analyze(self, state: dict[str, Any]) -> RemediationPlan:
        """Analyze state and produce remediation plan.

        Args:
            state: agent-specific state dict with all relevant metrics

        Returns:
            RemediationPlan with action, confidence, expected effect, reasoning
        """
        ...

    @abstractmethod
    async def execute(self, plan: RemediationPlan) -> ExecutionResult:
        """Execute the remediation plan.

        Args:
            plan: the plan to execute

        Returns:
            ExecutionResult with actual improvement measured
        """
        ...

    @staticmethod
    def _compute_confidence_from_state(
        state_features: dict[str, float], weights: dict[str, float]
    ) -> float:
        """Compute confidence as weighted sum of state features.

        Args:
            state_features: dict of metric -> value (0-1 normalized)
            weights: dict of metric -> weight (sum should = 1.0)

        Returns:
            Confidence score 0-1
        """
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.5

        confidence = 0.0
        for feature, weight in weights.items():
            if feature in state_features:
                confidence += state_features[feature] * (weight / total_weight)

        return min(max(confidence, 0.0), 1.0)
