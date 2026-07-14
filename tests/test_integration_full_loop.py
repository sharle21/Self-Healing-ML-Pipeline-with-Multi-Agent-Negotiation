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
from self_healing_pipeline.db.models import TenantConfig, DecisionOutcome, Base
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

        # Create DB session
        Base.metadata.create_all(get_engine())
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

            config_dict = initialize_tenant_config(
                tenant_id="test-tenant",
                validation_metrics=validation_metrics,
                deployment_profile=deployment_profile,
                daily_cost_budget=100.0,
                latency_sla_ms=100.0,
            )

            tenant_config = TenantConfig(**config_dict)
            session.add(tenant_config)
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

            # Store both
            session.add(TenantConfig(**strict_config))
            session.add(TenantConfig(**lenient_config))
            session.commit()

            # Verify they have different thresholds
            strict = session.query(TenantConfig).filter_by(tenant_id="strict").first()
            lenient = session.query(TenantConfig).filter_by(tenant_id="lenient").first()

            assert strict.baseline_auc > lenient.baseline_auc
            assert strict.latency_sla_ms < lenient.latency_sla_ms
            assert strict.daily_cost_budget < lenient.daily_cost_budget
