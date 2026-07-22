"""Integration: simulator → CommanderV3 → meta-harness learning loop."""

import asyncio
from datetime import datetime

import pytest

from self_healing_pipeline.agents.datarepair_v2 import DataRepairAgent
from self_healing_pipeline.agents.fallback_v2 import FallbackAgent
from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
from self_healing_pipeline.agents.rollback_v2 import RollbackAgent
from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
from self_healing_pipeline.commander.commander_v3 import CommanderV3
from self_healing_pipeline.config.tenant_config import (
    DeploymentProfile,
    ValidationMetrics,
    initialize_tenant_config,
)
from self_healing_pipeline.db.models import (
    TenantPolicy,
    ModelValidationReport,
    RuntimeDeploymentProfile,
    DecisionOutcome,
    Base,
)
from self_healing_pipeline.db.session import get_engine, session_scope
from self_healing_pipeline.gateway.events import Incident, IncidentType


class TestFullIntegrationLoop:
    """Test full loop: simulate → decide → verify → learn."""

    @pytest.mark.asyncio
    async def test_simulator_to_commander_to_learning(self):
        """Inject incidents → CommanderV3 → verify → collect outcomes."""
        # Setup agents
        agents = [
            ThresholdAdjustmentAgent("threshold-1"),
            RetrainAgent("retrain-1"),
            RollbackAgent("rollback-1"),
            FallbackAgent("fallback-1"),
            DataRepairAgent("datarepair-1"),
        ]

        # Create DB session (drop first to ensure clean state between runs)
        engine = get_engine()
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        for session in session_scope():
            # Initialize tenant with realistic metrics
            validation_metrics = ValidationMetrics(
                auc=0.78,
                optimal_threshold=0.55,
                precision=0.80,
                recall=0.76,
                f1=0.78,
                validation_timestamp=datetime.now(),
            )

            deployment_profile = DeploymentProfile(
                model_version="v5",
                latency_p95_ms=65.0,
                latency_p99_ms=120.0,
                throughput_rps=1000,
                deployment_timestamp=datetime.now(),
            )

            # Create TenantPolicy (governance)
            policy = TenantPolicy(
                tenant_id="test-tenant",
                min_acceptable_auc=validation_metrics.auc - 0.03,
                max_acceptable_latency_ms=deployment_profile.latency_p95_ms * 1.2,
                daily_cost_budget=100.0,
                latency_sla_ms=100.0,
            )
            session.add(policy)

            # Create ModelValidationReport (immutable)
            val_report = ModelValidationReport(
                model_version=deployment_profile.model_version,
                tenant_id="test-tenant",
                auc=validation_metrics.auc,
                precision=validation_metrics.precision,
                recall=validation_metrics.recall,
                f1_score=validation_metrics.f1,
                optimal_threshold=validation_metrics.optimal_threshold,
                calibration_error=0.05,
                validated_at=validation_metrics.validation_timestamp,
            )
            session.add(val_report)

            # Create RuntimeDeploymentProfile (continuous)
            runtime = RuntimeDeploymentProfile(
                tenant_id="test-tenant",
                model_version=deployment_profile.model_version,
                latency_p95_ms=deployment_profile.latency_p95_ms,
                latency_p99_ms=deployment_profile.latency_p99_ms,
                throughput_rps=float(deployment_profile.throughput_rps),
                measured_at=deployment_profile.deployment_timestamp,
            )
            session.add(runtime)
            session.commit()

            # Initialize commander with DB session
            commander = CommanderV3(agents, db_session=session)

            # Run 5 incidents of different types
            outcomes = []
            incident_types = [
                IncidentType.DRIFT,
                IncidentType.DATA_QUALITY,
                IncidentType.LATENCY_BREACH,
                IncidentType.COST_THRESHOLD,
                IncidentType.DRIFT,
            ]
            for i, inc_type in enumerate(incident_types):
                incident = Incident(
                    tenant_id="test-tenant",
                    type=inc_type,
                    payload={"index": i},
                )

                # Handle incident
                result = await commander.handle_incident(incident)

                # Store outcome
                outcome = DecisionOutcome(
                    tenant_id=incident.tenant_id,
                    incident_type=incident.type.value,
                    agent_type=result.winning_agent_type,
                    success=result.execution_result.get("success", False),
                    business_savings=0.0,
                    duration=result.execution_result.get("duration", 0.0),
                    reward=result.reward,
                    incident_resolved=result.incident_resolved,
                )
                session.add(outcome)
                outcomes.append(result)

            session.commit()

            # Verify outcomes
            assert len(outcomes) == 5
            for result in outcomes:
                # Agent might be "none" if no agents eligible for this incident type/state
                assert result.winning_agent_type in [
                    "threshold",
                    "retrain",
                    "rollback",
                    "fallback",
                    "data_repair",
                    "none",
                ]
                assert -1 <= result.reward <= 1
                assert isinstance(result.incident_resolved, bool)

            # Check DB stored outcomes
            stored = session.query(DecisionOutcome).filter_by(tenant_id="test-tenant").all()
            assert len(stored) == 5

    @pytest.mark.skip(reason="DB isolation needed between tests")
    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self):
        """Test that different tenants get different severity scores."""
        Base.metadata.create_all(get_engine())
        for session in session_scope():
            # Strict tenant (high expectations)
            strict_config = initialize_tenant_config(
                tenant_id="strict",
                validation_metrics=ValidationMetrics(
                    auc=0.85,
                    optimal_threshold=0.60,
                    precision=0.87,
                    recall=0.83,
                    f1=0.85,
                    validation_timestamp=datetime.now(),
                ),
                deployment_profile=DeploymentProfile(
                    model_version="v5",
                    latency_p95_ms=50.0,
                    latency_p99_ms=100.0,
                    throughput_rps=5000,
                    deployment_timestamp=datetime.now(),
                ),
                daily_cost_budget=50.0,
                latency_sla_ms=75.0,
            )

            # Lenient tenant (low expectations)
            lenient_config = initialize_tenant_config(
                tenant_id="lenient",
                validation_metrics=ValidationMetrics(
                    auc=0.70,
                    optimal_threshold=0.50,
                    precision=0.72,
                    recall=0.68,
                    f1=0.70,
                    validation_timestamp=datetime.now(),
                ),
                deployment_profile=DeploymentProfile(
                    model_version="v3",
                    latency_p95_ms=200.0,
                    latency_p99_ms=400.0,
                    throughput_rps=100,
                    deployment_timestamp=datetime.now(),
                ),
                daily_cost_budget=500.0,
                latency_sla_ms=500.0,
            )

            # Store policies
            strict_policy = TenantPolicy(
                tenant_id="strict",
                min_acceptable_auc=strict_config["min_auc"],
                max_acceptable_latency_ms=strict_config["max_latency_ms"],
                latency_sla_ms=strict_config["latency_sla_ms"],
                daily_cost_budget=strict_config["daily_cost_budget"],
            )
            lenient_policy = TenantPolicy(
                tenant_id="lenient",
                min_acceptable_auc=lenient_config["min_auc"],
                max_acceptable_latency_ms=lenient_config["max_latency_ms"],
                latency_sla_ms=lenient_config["latency_sla_ms"],
                daily_cost_budget=lenient_config["daily_cost_budget"],
            )
            session.add(strict_policy)
            session.add(lenient_policy)

            # Store validation reports
            strict_val = ModelValidationReport(
                model_version="v5",
                tenant_id="strict",
                auc=0.85,
                precision=0.87,
                recall=0.83,
                f1_score=0.85,
                optimal_threshold=0.60,
                calibration_error=0.02,
                validated_at=datetime.now(),
            )
            lenient_val = ModelValidationReport(
                model_version="v3",
                tenant_id="lenient",
                auc=0.70,
                precision=0.72,
                recall=0.68,
                f1_score=0.70,
                optimal_threshold=0.50,
                calibration_error=0.05,
                validated_at=datetime.now(),
            )
            session.add(strict_val)
            session.add(lenient_val)
            session.commit()

            # Verify they have different policies
            strict = session.query(TenantPolicy).filter_by(tenant_id="strict").first()
            lenient = session.query(TenantPolicy).filter_by(tenant_id="lenient").first()

            assert strict.min_acceptable_auc > lenient.min_acceptable_auc
            assert strict.latency_sla_ms < lenient.latency_sla_ms
            assert strict.daily_cost_budget < lenient.daily_cost_budget
