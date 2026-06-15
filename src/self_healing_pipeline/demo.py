"""End-to-end demo: inject incident → agents bid → commander picks → evidence bundle."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from self_healing_pipeline.agents.retrain import RetrainAgent
from self_healing_pipeline.agents.threshold import ThresholdAgent
from self_healing_pipeline.commander.commander import Commander
from self_healing_pipeline.gateway.events import Incident, IncidentType
from self_healing_pipeline.memory import BundleWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    threshold_agent = ThresholdAgent(agent_id="threshold-001")
    retrain_agent = RetrainAgent(agent_id="retrain-001")

    commander = Commander([threshold_agent, retrain_agent])
    bundle_writer = BundleWriter(traces_dir=Path("traces"))

    incident = Incident(
        tenant_id="enterprise",
        type=IncidentType.DRIFT,
        payload={"feature": "income", "threshold": 50000.0, "flip_prob": 0.8},
        severity=0.85,
        affected_features=("income", "age"),
    )

    logger.info(f"🚨 Incident created: {incident.id} (type={incident.type.value})")
    logger.info(f"   Tenant: {incident.tenant_id}, Severity: {incident.severity}")

    result = await commander.handle_incident(incident)

    logger.info(f"\n✅ Winner: {result.winning_agent_type}")
    logger.info(
        f"   Score: {result.winning_proposal.get('score', 0.0):.3f}, "
        f"Savings: ${result.winning_proposal.get('estimated_business_savings', 0.0):.2f}"
    )
    logger.info(f"   Execution: {'SUCCESS' if result.execution_result.get('success') else 'FAILED'}")

    if result.fallback_used:
        logger.warning("   ⚠️  Fallback to next-best agent was used")

    logger.info(f"\n📊 All proposals ({len(result.all_proposals)}):")
    for i, proposal in enumerate(result.all_proposals, 1):
        logger.info(
            f"   {i}. {proposal['agent_type']}: "
            f"confidence={proposal['confidence']:.2f}, "
            f"savings=${proposal['estimated_business_savings']:.2f}"
        )

    bundle_path = bundle_writer.write_commander_result(incident, result)
    logger.info(f"\n📦 Evidence bundle written: {bundle_path}")

    bundle = bundle_writer.read(incident.id)
    if bundle:
        logger.info(f"\n📄 Bundle contents (excerpt):")
        logger.info(f"   Incident: {bundle['incident']['id']}")
        logger.info(f"   Winner: {bundle['winner']['agent_type']} (score: {bundle['winner']['score']:.3f})")
        logger.info(f"   Execution: {bundle['execution_result']['success']}")


if __name__ == "__main__":
    asyncio.run(main())
