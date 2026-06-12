from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from self_healing_pipeline.gateway.events import Incident


@dataclass(slots=True)
class Proposal:
    agent_id: str
    agent_type: str
    confidence: float
    estimated_business_savings: float
    estimated_risk: float
    estimated_compute_cost: float
    estimated_time: float
    rationale: str
    memory_context: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not 0.0 <= self.estimated_risk <= 1.0:
            raise ValueError(f"estimated_risk must be in [0, 1], got {self.estimated_risk}")
        if self.estimated_business_savings < 0:
            raise ValueError(
                f"estimated_business_savings must be >= 0, got {self.estimated_business_savings}"
            )
        if self.estimated_compute_cost < 0:
            raise ValueError(
                f"estimated_compute_cost must be >= 0, got {self.estimated_compute_cost}"
            )
        if self.estimated_time < 0:
            raise ValueError(f"estimated_time must be >= 0, got {self.estimated_time}")


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    actual_business_savings: float
    duration: float
    error: str | None = None
    logs: list[str] = field(default_factory=list)


class Agent(ABC):
    agent_type: str

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    @abstractmethod
    def can_handle(self, incident: Incident) -> bool: ...

    @abstractmethod
    async def analyze(
        self, incident: Incident, memory_context: dict[str, Any]
    ) -> Proposal: ...

    @abstractmethod
    async def execute(
        self, proposal: Proposal, incident: Incident
    ) -> ExecutionResult: ...
