"""End-to-end demo: 3-layer pipeline (observe → decide → verify)."""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.orm import Session

from self_healing_pipeline.agents.datarepair_v2 import DataRepairAgent
from self_healing_pipeline.agents.fallback_v2 import FallbackAgent
from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
from self_healing_pipeline.agents.rollback_v2 import RollbackAgent
from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
from self_healing_pipeline.commander.commander_v3 import CommanderV3
from self_healing_pipeline.config import get_settings
from self_healing_pipeline.db.session import create_all, get_engine
from self_healing_pipeline.gateway.events import Incident, IncidentType
from self_healing_pipeline.memory.tier3_traces import BundleWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    # Initialize 5 remediation policy agents
    agents = [
        ThresholdAdjustmentAgent("threshold-1"),
        RetrainAgent("retrain-1"),
        RollbackAgent("rollback-1"),
        FallbackAgent("fallback-1"),
        DataRepairAgent("datarepair-1"),
    ]

    create_all()
    with Session(get_engine(), future=True) as session:
        commander = CommanderV3(
            agents,
            db_session=session,
            bundle_writer=BundleWriter(get_settings().traces_dir),
        )

        incident = Incident(
            tenant_id="enterprise",
            type=IncidentType.DRIFT,
            payload={"feature": "income", "threshold": 50000.0, "flip_prob": 0.8},
            severity=0.85,
            affected_features=("income", "age"),
        )

        logger.info(f"🚨 Incident created: {incident.id} (type={incident.type.value})")
        logger.info(f"   Tenant: {incident.tenant_id}")

        result = await commander.handle_incident(incident)

    logger.info(f"\n✅ Winner: {result.winning_agent_type}")
    logger.info(
        f"   Severity: {result.severity:.3f}, "
        f"Confidence: {result.winning_plan.get('confidence', 0.0):.3f}"
    )
    logger.info(f"   Reward: {result.reward:.3f}")
    logger.info(f"   Execution: {'SUCCESS' if result.execution_result.get('success') else 'FAILED'}")
    logger.info(f"   Incident Resolved: {result.incident_resolved}")

    if result.reconciliation_triggered:
        logger.warning("   ⚠️  Reconciliation was triggered for close call")

    if result.escalation_triggered:
        logger.error("   ⛔ Escalation: all agents failed")

    logger.info(f"\n📊 Verification breakdown:")
    for key, value in result.verification_breakdown.items():
        logger.info(f"   {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
