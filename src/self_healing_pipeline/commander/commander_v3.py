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
    ModelValidationReport,
    RemediationAction,
    RuntimeDeploymentProfile,
    TenantPolicy,
    TenantTierConfig,
)
from self_healing_pipeline.observability.metrics import (
    agent_proposal_count,
    agent_win_count,
    incident_count,
    prediction_latency,
)
from self_healing_pipeline.gateway.events import Incident, IncidentType
from self_healing_pipeline.observability import (
    IncidentStateBuilder,
    SeverityCalculator,
    StateConstructor,
    TelemetryCollector,
)
from self_healing_pipeline.commander.utility import UtilityScorer, UtilityWeights
from self_healing_pipeline.verification import RewardCalculator
from self_healing_pipeline.verification.reward import OutcomeReward

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
    utility_score: float = 0.0             # Phase 11: pre-execution utility estimate
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
        session_factory: Any | None = None,
        use_mock_telemetry: bool = True,
        stabilization_seconds: float = 0.0,
    ) -> None:
        """Initialize Commander V3.

        Args:
            agents: list of RemediationPolicyAgent instances
            sonnet_model: optional Sonnet model for LLM reconciliation
            db_session: optional SQLAlchemy session for DB persistence
            session_factory: optional SQLAlchemy session factory for IncidentStateBuilder
            use_mock_telemetry: if True, TelemetryCollector uses mock data (for testing)
            stabilization_seconds: seconds to wait after execution before sampling post-action
                metrics. 0 in tests; 10-30 in production to let Prometheus gauges update.
        """
        self.agents = agents
        self.sonnet_model = sonnet_model
        self.db_session = db_session
        self.telemetry_collector = TelemetryCollector(use_mock=use_mock_telemetry)
        self.severity_calculator = SeverityCalculator()
        self.state_constructor = StateConstructor()
        self.reward_calculator = RewardCalculator()
        self.reconciliation = LangGraphReconciliation(model_name=sonnet_model)
        self.tenant_configs: dict[str, dict] = {}
        self.stabilization_seconds = stabilization_seconds
        self.incident_state_builder = IncidentStateBuilder(
            self.telemetry_collector, session_factory=session_factory
        )

    async def handle_incident(self, incident: Incident) -> CommanderResultV3:
        """Handle incident end-to-end: observe → decide → verify → reward.

        Args:
            incident: the incident to resolve

        Returns:
            CommanderResultV3 with execution + verification
        """
        # Layer 1: OBSERVATION
        logger.info(f"[Observe] Incident {incident.id} type={incident.type.value}")

        # Record incident
        incident_count.labels(tenant_id=incident.tenant_id, type=incident.type.value).inc()

        # Load policy + validation + runtime profile
        tenant_policy = self._load_tenant_policy(incident.tenant_id)
        validation_report = self._get_latest_validation_report(incident.tenant_id)
        runtime_profile = self._get_latest_runtime_profile(incident.tenant_id)

        # Compose into severity config
        severity_config = None
        if tenant_policy and validation_report and runtime_profile:
            severity_config = {
                "baseline_auc": validation_report.auc,
                "min_auc": tenant_policy.min_acceptable_auc,
                "baseline_latency_ms": runtime_profile.latency_p95_ms,
                "max_latency_ms": tenant_policy.max_acceptable_latency_ms,
                "max_missing_rate": tenant_policy.max_acceptable_missing_rate,
                "latency_sla_ms": tenant_policy.latency_sla_ms,
                "daily_cost_budget": tenant_policy.daily_cost_budget,
            }

        telemetry_before = self.telemetry_collector.collect()
        severity, severity_breakdown = self.severity_calculator.calculate(
            incident.type, telemetry_before, tenant_config=severity_config
        )

        comp_str = " ".join(f"{k}={v:.2f}" for k, v in severity_breakdown.components.items())
        logger.info("[Observe] Severity=%.3f (%s)", severity, comp_str)

        # Store incident history
        self._store_incident_history(incident, severity)

        # Build IncidentState: real values from Prometheus + DB
        incident_state = self.incident_state_builder.build(
            incident.tenant_id, incident.type.value
        )

        # Layer 2: REMEDIATION (Decide)
        logger.info("[Decide] Getting agent recommendations...")

        eligible_agents = [
            a for a in self.agents
            if a.can_handle(self._get_agent_state_v2(incident.type, incident_state, incident.payload))
        ]

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
            state = self._get_agent_state_v2(incident.type, incident_state, incident.payload)
            plan = await agent.analyze(state)
            plans.append((agent, plan))
            # Record every proposal (not just the winner)
            agent_proposal_count.labels(agent_type=agent.agent_type).inc()

        # Phase 11: Sort by utility score (not raw confidence)
        tier_config = self._load_tenant_tier_config(incident.tenant_id)
        utility_weights = UtilityScorer.weights_from_tier_config(tier_config, incident_state.incident_type)
        ranked = UtilityScorer.rank(plans, incident_state, utility_weights)

        for agent, plan, u in ranked:
            logger.info(
                "[Decide] Agent=%s utility=%.3f confidence=%.3f",
                agent.agent_type, u, plan.confidence,
            )

        winning_agent, winning_plan, winning_utility = ranked[0]
        plans = [(a, p) for a, p, _ in ranked]  # keep sorted order for fallback loop

        # Check if reconciliation needed (close utility scores → close call)
        reconciliation_triggered = False
        reconciliation_log: dict[str, Any] | None = None
        if len(ranked) > 1 and self._needs_reconciliation(ranked[0][2], ranked[1][2]):
            logger.info(
                "[Decide] Reconciliation triggered "
                "(top 2 utilities: %.3f, %.3f)",
                ranked[0][2], ranked[1][2],
            )
            reconciliation_triggered = True
            # Reuse LangGraph for close calls
            try:
                result = await self.reconciliation.debate(
                    ranked[0][1], ranked[1][1], incident  # type: ignore
                )
                reconciliation_log = {
                    "winner_type": result.winner_type,
                    "rationale": result.rationale,
                    "confidence": result.confidence,
                    "debate_log": result.debate_log,
                }
                winning_agent, winning_plan = next(
                    ((a, p) for a, p in plans if a.agent_type == result.winner_type),
                    (winning_agent, winning_plan),
                )
            except Exception as e:
                logger.warning(f"Reconciliation failed: {e}, using top candidate")

        logger.info(
            "[Decide] Winner: %s (utility=%.3f confidence=%.3f)",
            winning_agent.agent_type, winning_utility, winning_plan.confidence,
        )

        # Record agent win
        agent_win_count.labels(agent_type=winning_agent.agent_type).inc()

        # Layer 2b: EXECUTION (Run plan with fallback)
        logger.info(f"[Execute] Running {winning_plan.action}...")

        import time
        exec_start = time.time()
        execution_result = await winning_agent.execute(winning_plan)
        exec_duration = time.time() - exec_start
        execution_agent = winning_agent
        fallback_attempts = []

        # Record execution latency (in seconds)
        prediction_latency.labels(tenant_id=incident.tenant_id).observe(exec_duration)

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
        logger.info("[Verify] Waiting %.1fs for metrics to stabilize...", self.stabilization_seconds)

        if self.stabilization_seconds > 0:
            import asyncio
            await asyncio.sleep(self.stabilization_seconds)

        # Rebuild IncidentState from Prometheus to get post-action metrics
        incident_state_after = self.incident_state_builder.build(
            incident.tenant_id, incident.type.value
        )

        # Outcome-based reward: real before/after deltas, not agent estimates
        reward, outcome_breakdown = RewardCalculator.calculate_from_incident_states(
            incident_state, incident_state_after, execution_agent.agent_type, execution_result
        )
        incident_resolved = outcome_breakdown.resolution_score > 0.5

        logger.info(
            "[Verify] Reward=%.3f resolved=%s "
            "(quality_gain=%.2f cost_gain=%.2f reliability_gain=%.2f latency_gain=%.2f)",
            reward, incident_resolved,
            outcome_breakdown.quality_gain, outcome_breakdown.cost_gain,
            outcome_breakdown.reliability_gain, outcome_breakdown.latency_gain,
        )

        # Store remediation action with reward
        self._store_remediation_action(incident, execution_agent, winning_plan, execution_result, reward)

        verification_breakdown = {
            "quality_gain": outcome_breakdown.quality_gain,
            "cost_gain": outcome_breakdown.cost_gain,
            "reliability_gain": outcome_breakdown.reliability_gain,
            "latency_gain": outcome_breakdown.latency_gain,
            "resolution_score": outcome_breakdown.resolution_score,
            "exec_cost_penalty": outcome_breakdown.exec_cost_penalty,
            "time_penalty": outcome_breakdown.time_penalty,
            "regression_penalty": outcome_breakdown.regression_penalty,
            "auc_before": outcome_breakdown.auc_before,
            "auc_after": outcome_breakdown.auc_after,
            "reward": reward,
        }

        return CommanderResultV3(
            incident_id=incident.id,
            incident_type=incident.type.value,
            severity=severity,
            winning_agent_type=execution_agent.agent_type,
            winning_plan=asdict(winning_plan) if hasattr(winning_plan, '__dataclass_fields__') else vars(winning_plan),
            execution_result=asdict(execution_result) if hasattr(execution_result, '__dataclass_fields__') else vars(execution_result),
            reward=reward,
            incident_resolved=incident_resolved,
            verification_breakdown=verification_breakdown,
            utility_score=winning_utility,
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

    def _load_tenant_policy(self, tenant_id: str) -> TenantPolicy | None:
        """Load tenant policy (governance decisions) from DB.

        Args:
            tenant_id: tenant identifier

        Returns:
            TenantPolicy row or None if not found
        """
        if not self.db_session:
            return None
        return self.db_session.query(TenantPolicy).filter_by(tenant_id=tenant_id).first()

    def _get_latest_validation_report(self, tenant_id: str) -> ModelValidationReport | None:
        """Get latest model validation report for tenant.

        Args:
            tenant_id: tenant identifier

        Returns:
            Latest ModelValidationReport or None
        """
        if not self.db_session:
            return None
        return (
            self.db_session.query(ModelValidationReport)
            .filter_by(tenant_id=tenant_id)
            .order_by(ModelValidationReport.validated_at.desc())
            .first()
        )

    def _get_latest_runtime_profile(self, tenant_id: str) -> RuntimeDeploymentProfile | None:
        """Get latest runtime deployment profile for tenant.

        Args:
            tenant_id: tenant identifier

        Returns:
            Latest RuntimeDeploymentProfile or None
        """
        if not self.db_session:
            return None
        return (
            self.db_session.query(RuntimeDeploymentProfile)
            .filter_by(tenant_id=tenant_id)
            .order_by(RuntimeDeploymentProfile.updated_at.desc())
            .first()
        )

    def _load_tenant_tier_config(self, tenant_id: str) -> TenantTierConfig | None:
        """Load tenant tier config (agent eligibility + weights) from DB.

        Args:
            tenant_id: tenant identifier

        Returns:
            TenantTierConfig row or None if not found
        """
        if not self.db_session:
            return None
        return self.db_session.query(TenantTierConfig).filter_by(tenant_id=tenant_id).first()

    def _get_agent_state(
        self, incident_type: IncidentType, telemetry: Any, payload: dict | None = None
    ) -> dict[str, Any]:
        """Construct agent state from telemetry, overriding with live incident payload values."""
        if incident_type == IncidentType.DRIFT:
            state = self.state_constructor.retrain_state(telemetry).to_dict()
            state.update(self.state_constructor.threshold_state(telemetry).to_dict())
        elif incident_type == IncidentType.DATA_QUALITY:
            state = self.state_constructor.datarepair_state(telemetry).to_dict()
            state.update(self.state_constructor.fallback_state(telemetry).to_dict())
        elif incident_type == IncidentType.LATENCY_BREACH:
            state = self.state_constructor.threshold_state(telemetry).to_dict()
            state.update(self.state_constructor.fallback_state(telemetry).to_dict())
        elif incident_type == IncidentType.COST_THRESHOLD:
            state = self.state_constructor.threshold_state(telemetry).to_dict()
            state.update(self.state_constructor.fallback_state(telemetry).to_dict())
        else:
            state = {}

        # Override with real values from incident payload (beats mock telemetry)
        if payload:
            state.update({k: v for k, v in payload.items() if v is not None})

        return state

    def _get_agent_state_v2(
        self, incident_type: IncidentType, incident_state: Any, payload: dict | None = None
    ) -> dict[str, Any]:
        """Construct agent state from IncidentState (data-driven, no hardcoded constants).

        Merges all relevant per-agent state fields into a single dict so every
        agent sees the full picture and can decide whether it can_handle().
        """
        sc = self.state_constructor

        if incident_type == IncidentType.DRIFT:
            state = sc.retrain_state_from_incident(incident_state).to_dict()
            state.update(sc.threshold_state_from_incident(incident_state).to_dict())
            state.update(sc.rollback_state_from_incident(incident_state).to_dict())
        elif incident_type == IncidentType.DATA_QUALITY:
            state = sc.datarepair_state_from_incident(incident_state).to_dict()
            state.update(sc.fallback_state_from_incident(incident_state).to_dict())
        elif incident_type == IncidentType.LATENCY_BREACH:
            state = sc.threshold_state_from_incident(incident_state).to_dict()
            state.update(sc.fallback_state_from_incident(incident_state).to_dict())
        elif incident_type == IncidentType.COST_THRESHOLD:
            state = sc.threshold_state_from_incident(incident_state).to_dict()
            state.update(sc.fallback_state_from_incident(incident_state).to_dict())
        else:
            state = {}

        # Incident payload values always win (most specific signal)
        if payload:
            state.update({k: v for k, v in payload.items() if v is not None})

        # Always inject tenant_id so agents can write to the correct DB row
        state["tenant_id"] = incident_state.tenant_id

        return state

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
