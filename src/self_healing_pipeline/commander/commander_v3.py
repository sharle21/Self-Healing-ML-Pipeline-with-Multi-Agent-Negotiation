"""Commander V3: 3-layer architecture with observation, remediation, verification."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.orm import Session

from self_healing_pipeline.agents.remediation_policy import RemediationPolicyAgent
from self_healing_pipeline.commander.reconciliation_langgraph import LangGraphReconciliation
from self_healing_pipeline.db.models import (
    IncidentHistory,
    RemediationAction,
    TenantConfig,
)
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
    reconciliation_triggered: bool = False
    reconciliation_log: dict[str, Any] | None = None
    escalation_triggered: bool = False
    escalation_log: dict[str, Any] | None = None


class CommanderV3:
    """3-Layer Commander: Observe → Decide → Verify → Learn.

    1. Observation: Collect telemetry → calculate severity → construct state
    2. Remediation: Agent selects plan based on state-based confidence
    3. Verification: Execute → measure outcome → calculate reward
    """

    def __init__(
        self,
        agents: list[RemediationPolicyAgent],
        sonnet_model: str | None = None,
        db_session: Session | None = None,
    ) -> None:
        """Initialize Commander V3.

        Args:
            agents: list of RemediationPolicyAgent instances
            sonnet_model: optional Sonnet model for LLM reconciliation
            db_session: optional SQLAlchemy session for DB persistence
        """
        self.agents = agents
        self.sonnet_model = sonnet_model
        self.db_session = db_session
        self.telemetry_collector = TelemetryCollector(use_mock=True)
        self.severity_calculator = SeverityCalculator()
        self.state_constructor = StateConstructor()
        self.reward_calculator = RewardCalculator()
        self.reconciliation = LangGraphReconciliation(model_name=sonnet_model)
        self.tenant_configs: dict[str, dict] = {}

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

        # Store incident history
        self._store_incident_history(incident, severity)

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

        # Check if reconciliation needed (close call)
        reconciliation_triggered = False
        reconciliation_log: dict[str, Any] | None = None
        if len(plans) > 1 and self._needs_reconciliation(plans[0][1].confidence, plans[1][1].confidence):
            logger.info(
                f"[Decide] Reconciliation triggered "
                f"(top 2 confidences: {plans[0][1].confidence:.3f}, {plans[1][1].confidence:.3f})"
            )
            reconciliation_triggered = True
            # Reuse LangGraph for close calls
            try:
                result = await self.reconciliation.debate(
                    plans[0][1], plans[1][1], incident  # type: ignore
                )
                reconciliation_log = {
                    "winner_type": result.winner_type,
                    "rationale": result.rationale,
                    "confidence": result.confidence,
                    "debate_log": result.debate_log,
                }
                winning_agent, winning_plan = next(
                    (p for p in plans if p[0].agent_type == result.winner_type),
                    plans[0],
                )
            except Exception as e:
                logger.warning(f"Reconciliation failed: {e}, using top candidate")

        logger.info(
            f"[Decide] Winner: {winning_agent.agent_type} "
            f"(confidence={winning_plan.confidence:.3f})"
        )

        # Layer 2b: EXECUTION (Run plan with fallback)
        logger.info(f"[Execute] Running {winning_plan.action}...")

        execution_result = await winning_agent.execute(winning_plan)
        execution_agent = winning_agent
        fallback_attempts = []

        # Fallback: try next best if winner fails
        if not execution_result.success and len(plans) > 1:
            logger.warning(
                f"Winner {winning_agent.agent_type} failed: {execution_result}. Trying next best."
            )
            for agent, plan in plans[1:]:
                execution_result = await agent.execute(plan)
                fallback_attempts.append((agent.agent_type, plan.action, execution_result.success))
                if execution_result.success:
                    execution_agent = agent
                    winning_plan = plan
                    logger.info(f"Fallback succeeded with {agent.agent_type}")
                    break

        # Escalation: all agents failed
        escalation_triggered = False
        escalation_log: dict[str, Any] | None = None
        if not execution_result.success:
            escalation_triggered = True
            escalation_log = {
                "reason": f"All {len(plans)} agents failed",
                "failed_attempts": fallback_attempts or [(winning_agent.agent_type, winning_plan.action)],
            }
            logger.error(f"Escalation: {escalation_log['reason']}")

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
            telemetry_before, telemetry_after, execution_agent.agent_type, incident_resolved
        )

        logger.info(
            f"[Verify] Reward={reward:.3f} "
            f"(resolved={incident_resolved}, "
            f"metric_improvement={reward_breakdown.metric_improvement:.2f}, "
            f"cost_efficiency={reward_breakdown.cost_efficiency:.2f})"
        )

        # Store remediation action with reward
        self._store_remediation_action(incident, execution_agent, winning_plan, execution_result, reward)

        return CommanderResultV3(
            incident_id=incident.id,
            incident_type=incident.type.value,
            severity=severity,
            winning_agent_type=execution_agent.agent_type,
            winning_plan=asdict(winning_plan) if hasattr(winning_plan, '__dataclass_fields__') else vars(winning_plan),
            execution_result=asdict(execution_result) if hasattr(execution_result, '__dataclass_fields__') else vars(execution_result),
            reward=reward,
            incident_resolved=incident_resolved,
            verification_breakdown=asdict(reward_breakdown),
            reconciliation_triggered=reconciliation_triggered,
            reconciliation_log=reconciliation_log,
            escalation_triggered=escalation_triggered,
            escalation_log=escalation_log,
        )

    def _needs_reconciliation(self, top_confidence: float, second_confidence: float) -> bool:
        """Check if top 2 agents are close enough to warrant reconciliation.

        Args:
            top_confidence: winning agent confidence
            second_confidence: runner-up confidence

        Returns:
            True if margin < 10%
        """
        if top_confidence == 0:
            return False
        margin = abs(top_confidence - second_confidence) / top_confidence
        return margin < 0.10

    def _store_incident_history(self, incident: Incident, severity: float) -> None:
        """Store incident in DB for meta-harness learning.

        Args:
            incident: the incident
            severity: calculated severity
        """
        if not self.db_session:
            return

        history = IncidentHistory(
            incident_id=incident.id,
            tenant_id=incident.tenant_id,
            type=incident.type.value,
            severity=severity,
        )
        self.db_session.add(history)
        self.db_session.commit()

    def _store_remediation_action(
        self,
        incident: Incident,
        agent: RemediationPolicyAgent,
        plan: Any,
        execution_result: Any,
        reward: float = 0.0,
    ) -> None:
        """Store remediation action for audit trail.

        Args:
            incident: the incident
            agent: agent that executed
            plan: remediation plan
            execution_result: execution result
            reward: calculated reward
        """
        if not self.db_session:
            return

        proposal_dict = asdict(plan) if hasattr(plan, '__dataclass_fields__') else vars(plan)

        action = RemediationAction(
            incident_id=incident.id,
            agent=agent.agent_type,
            proposal=proposal_dict,
            chosen=True,
            reward=reward,
            success=execution_result.success,
        )
        self.db_session.add(action)
        self.db_session.commit()

    def _load_tenant_config(self, tenant_id: str) -> dict[str, Any]:
        """Load tenant-specific config from DB.

        Args:
            tenant_id: tenant identifier

        Returns:
            config dict with tenant settings
        """
        if tenant_id in self.tenant_configs:
            return self.tenant_configs[tenant_id]

        if not self.db_session:
            return self._default_tenant_config()

        config_row = self.db_session.query(TenantConfig).filter_by(tenant_id=tenant_id).first()
        if config_row:
            config = {
                "decision_threshold": config_row.decision_threshold,
                "model_version": config_row.model_version,
                "latency_sla": config_row.latency_sla,
                "accuracy_target": config_row.accuracy_target,
                "cost_budget": config_row.cost_budget,
                "last_training_time": config_row.last_training_time,
            }
        else:
            config = self._default_tenant_config()
            # Store default for future
            config_row = TenantConfig(tenant_id=tenant_id, **config)
            self.db_session.add(config_row)
            self.db_session.commit()

        self.tenant_configs[tenant_id] = config
        return config

    def _default_tenant_config(self) -> dict[str, Any]:
        """Return default tenant config."""
        return {
            "decision_threshold": 0.5,
            "model_version": "v1",
            "latency_sla": 100.0,
            "accuracy_target": 0.75,
            "cost_budget": 0.10,
            "last_training_time": None,
        }

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
