"""Commander V3: 3-layer architecture with observation, remediation, verification."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Any

from self_healing_pipeline.agents.remediation_policy import RemediationPolicyAgent
from self_healing_pipeline.gateway.events import Incident, IncidentType
from self_healing_pipeline.observability import (
    SeverityCalculator,
    StateConstructor,
    TelemetryCollector,
)
from self_healing_pipeline.verification import RewardCalculator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CommanderResultV3:
    """Commander execution result with verification."""

    incident_id: str
    incident_type: str
    severity: float
    winning_agent_type: str
    winning_plan: dict[str, Any]
    execution_result: dict[str, Any]
    reward: float
    incident_resolved: bool
    verification_breakdown: dict[str, Any]


class CommanderV3:
    """3-Layer Commander: Observe → Decide → Verify → Learn.

    1. Observation: Collect telemetry → calculate severity → construct state
    2. Remediation: Agent selects plan based on state-based confidence
    3. Verification: Execute → measure outcome → calculate reward
    """

    def __init__(self, agents: list[RemediationPolicyAgent], sonnet_model: str | None = None) -> None:
        """Initialize Commander V3.

        Args:
            agents: list of RemediationPolicyAgent instances
            sonnet_model: optional Sonnet model for LLM reconciliation
        """
        self.agents = agents
        self.sonnet_model = sonnet_model
        self.telemetry_collector = TelemetryCollector(use_mock=True)
        self.severity_calculator = SeverityCalculator()
        self.state_constructor = StateConstructor()
        self.reward_calculator = RewardCalculator()

    async def handle_incident(self, incident: Incident) -> CommanderResultV3:
        """Handle incident end-to-end: observe → decide → verify → reward.

        Args:
            incident: the incident to resolve

        Returns:
            CommanderResultV3 with execution + verification
        """
        # Layer 1: OBSERVATION
        logger.info(f"[Observe] Incident {incident.id} type={incident.type.value}")

        telemetry_before = self.telemetry_collector.collect()
        severity, severity_breakdown = self.severity_calculator.calculate(
            incident.type, telemetry_before
        )

        logger.info(
            f"[Observe] Severity={severity:.3f} "
            f"(impact={severity_breakdown.impact:.2f}, "
            f"deviation={severity_breakdown.deviation:.2f})"
        )

        # Layer 2: REMEDIATION (Decide)
        logger.info("[Decide] Getting agent recommendations...")

        eligible_agents = [a for a in self.agents if a.can_handle(self._get_agent_state(incident.type, telemetry_before))]

        if not eligible_agents:
            logger.error(f"No eligible agents for {incident.type.value}")
            return CommanderResultV3(
                incident_id=incident.id,
                incident_type=incident.type.value,
                severity=severity,
                winning_agent_type="none",
                winning_plan={},
                execution_result={},
                reward=-1.0,
                incident_resolved=False,
                verification_breakdown={"error": "no_eligible_agents"},
            )

        # Get plans from all eligible agents
        plans = []
        for agent in eligible_agents:
            state = self._get_agent_state(incident.type, telemetry_before)
            plan = await agent.analyze(state)
            plans.append((agent, plan))

        # Sort by confidence
        plans.sort(key=lambda x: x[1].confidence, reverse=True)
        winning_agent, winning_plan = plans[0]

        logger.info(
            f"[Decide] Winner: {winning_agent.agent_type} "
            f"(confidence={winning_plan.confidence:.3f})"
        )

        # Layer 2b: EXECUTION (Run plan)
        logger.info(f"[Execute] Running {winning_plan.action}...")

        execution_result = await winning_agent.execute(winning_plan)

        logger.info(
            f"[Execute] {winning_plan.action} completed "
            f"(success={execution_result.success}, duration={execution_result.duration:.1f}s)"
        )

        # Layer 3: VERIFICATION (Measure & Reward)
        logger.info("[Verify] Collecting post-execution telemetry...")

        telemetry_after = self.telemetry_collector.collect()

        # Calculate reward based on incident type
        incident_resolved = self._check_incident_resolved(incident.type, telemetry_before, telemetry_after)

        reward_func = {
            IncidentType.DRIFT: RewardCalculator.calculate_drift_reward,
            IncidentType.DATA_QUALITY: RewardCalculator.calculate_data_quality_reward,
            IncidentType.LATENCY_BREACH: RewardCalculator.calculate_latency_reward,
            IncidentType.COST_THRESHOLD: RewardCalculator.calculate_cost_reward,
        }.get(incident.type, RewardCalculator.calculate_drift_reward)

        reward, reward_breakdown = reward_func(
            telemetry_before, telemetry_after, winning_agent.agent_type, incident_resolved
        )

        logger.info(
            f"[Verify] Reward={reward:.3f} "
            f"(resolved={incident_resolved}, "
            f"metric_improvement={reward_breakdown.metric_improvement:.2f}, "
            f"cost_efficiency={reward_breakdown.cost_efficiency:.2f})"
        )

        return CommanderResultV3(
            incident_id=incident.id,
            incident_type=incident.type.value,
            severity=severity,
            winning_agent_type=winning_agent.agent_type,
            winning_plan=asdict(winning_plan) if hasattr(winning_plan, '__dataclass_fields__') else vars(winning_plan),
            execution_result=asdict(execution_result) if hasattr(execution_result, '__dataclass_fields__') else vars(execution_result),
            reward=reward,
            incident_resolved=incident_resolved,
            verification_breakdown=asdict(reward_breakdown),
        )

    def _get_agent_state(self, incident_type: IncidentType, telemetry: Any) -> dict[str, Any]:
        """Construct appropriate state dict for agents based on incident type.

        Args:
            incident_type: type of incident
            telemetry: current telemetry snapshot

        Returns:
            Agent-specific state dict
        """
        if incident_type == IncidentType.DRIFT:
            # For drift, provide retrain/threshold/rollback state
            state = self.state_constructor.retrain_state(telemetry).to_dict()
            state.update(self.state_constructor.threshold_state(telemetry).to_dict())
            return state
        elif incident_type == IncidentType.DATA_QUALITY:
            # For data quality, provide datarepair/fallback state
            state = self.state_constructor.datarepair_state(telemetry).to_dict()
            state.update(self.state_constructor.fallback_state(telemetry).to_dict())
            return state
        elif incident_type == IncidentType.LATENCY_BREACH:
            # For latency, provide threshold/fallback/rollback state
            state = self.state_constructor.threshold_state(telemetry).to_dict()
            state.update(self.state_constructor.fallback_state(telemetry).to_dict())
            return state
        elif incident_type == IncidentType.COST_THRESHOLD:
            # For cost, provide threshold/fallback state
            state = self.state_constructor.threshold_state(telemetry).to_dict()
            state.update(self.state_constructor.fallback_state(telemetry).to_dict())
            return state
        else:
            return {}

    def _check_incident_resolved(
        self, incident_type: IncidentType, telemetry_before: Any, telemetry_after: Any
    ) -> bool:
        """Check if incident was actually resolved.

        Args:
            incident_type: type of incident
            telemetry_before: telemetry before remediation
            telemetry_after: telemetry after remediation

        Returns:
            True if incident resolved (metrics improved significantly)
        """
        if incident_type == IncidentType.DRIFT:
            auc_recovery = telemetry_after.model.auc - telemetry_before.model.auc
            drift_reduction = (
                max(telemetry_before.data.feature_drift_scores.values() or [0])
                - max(telemetry_after.data.feature_drift_scores.values() or [0])
            )
            return auc_recovery > 0.03 or drift_reduction > 0.5

        elif incident_type == IncidentType.DATA_QUALITY:
            missing_improvement = telemetry_before.data.missing_rate - telemetry_after.data.missing_rate
            schema_improvement = telemetry_before.data.schema_violations - telemetry_after.data.schema_violations
            return missing_improvement > 0.10 or schema_improvement > 10

        elif incident_type == IncidentType.LATENCY_BREACH:
            latency_improvement = telemetry_before.system.latency_p95 - telemetry_after.system.latency_p95
            return latency_improvement > 20

        elif incident_type == IncidentType.COST_THRESHOLD:
            cost_reduction = telemetry_before.system.cost_per_prediction - telemetry_after.system.cost_per_prediction
            return cost_reduction > 0.0005

        return False
