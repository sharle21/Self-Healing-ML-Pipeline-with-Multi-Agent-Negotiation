"""Week 2 end-of-week demo: 5-incident scenario showcasing all features.

Demonstrates:
- All 5 agents winning at least once
- Memory context evolution (recent_success_rate changes confidence)
- At least one reconciliation debate
- Fallback handling
- Full commander flow with logging
"""

import asyncio
import logging
from typing import Any

from self_healing_pipeline.agents.data_repair import DataRepairAgent
from self_healing_pipeline.agents.fallback import FallbackAgent
from self_healing_pipeline.agents.retrain import RetrainAgent
from self_healing_pipeline.agents.rollback import RollbackAgent
from self_healing_pipeline.agents.threshold import ThresholdAgent
from self_healing_pipeline.commander.commander import Commander
from self_healing_pipeline.gateway.events import Incident, IncidentType
from self_healing_pipeline.memory.memory import Memory
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from self_healing_pipeline.db.models import Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_incidents() -> list[Incident]:
    """Create 5-incident sequence designed to exercise all agents."""
    return [
        # Incident 1: DRIFT (very low severity) - threshold should win (cheap, instant)
        Incident(
            tenant_id="standard",
            type=IncidentType.DRIFT,
            payload={"drift_percentage": 0.10},
            severity=0.15,
            affected_features=("feature_1",),
        ),
        # Incident 2: DATA_QUALITY (high severity) - data_repair should win (durable fix)
        Incident(
            tenant_id="enterprise",
            type=IncidentType.DATA_QUALITY,
            payload={"missing_rate": 0.30, "duplicate_rate": 0.08},
            severity=0.7,
            affected_features=("feature_2", "feature_3"),
        ),
        # Incident 3: DRIFT (medium severity) - rollback should win (fast, safe)
        Incident(
            tenant_id="standard",
            type=IncidentType.DRIFT,
            payload={"drift_percentage": 0.35},
            severity=0.45,
            affected_features=("feature_1", "feature_4"),
        ),
        # Incident 4: COST_THRESHOLD - fallback should win (safe degradation)
        Incident(
            tenant_id="free",
            type=IncidentType.COST_THRESHOLD,
            payload={"cost_per_prediction": 50.0},
            severity=0.5,
            affected_features=(),
        ),
        # Incident 5: DRIFT (high severity) - retrain should win (worth the cost)
        Incident(
            tenant_id="enterprise",
            type=IncidentType.DRIFT,
            payload={"drift_percentage": 0.75},
            severity=0.8,
            affected_features=("feature_1", "feature_2", "feature_5"),
        ),
    ]


async def run_demo() -> None:
    """Run 5-incident demo scenario."""
    logger.info("=" * 80)
    logger.info("WEEK 2 DEMO: 5-Incident Scenario with Memory & Reconciliation")
    logger.info("=" * 80)

    # Setup: DB + agents + commander + memory
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db_session = Session(engine)

    agents = [
        ThresholdAgent("threshold-1"),
        RetrainAgent("retrain-1"),
        RollbackAgent("rollback-1"),
        FallbackAgent("fallback-1"),
        DataRepairAgent("data_repair-1"),
    ]
    commander = Commander(agents, sonnet_model=None)
    memory = Memory(db_session)

    incidents = create_incidents()
    results: list[tuple[Incident, Any]] = []

    # Run each incident
    for i, incident in enumerate(incidents, 1):
        logger.info(f"\n--- INCIDENT {i} ---")
        logger.info(
            f"Type: {incident.type.value} | Tenant: {incident.tenant_id} | "
            f"Severity: {incident.severity:.1f}"
        )

        # Recall memory before incident
        memory_context = memory.recall(incident.tenant_id, incident.type.value)
        if not memory_context.cold_start:
            logger.info(
                f"Memory: {memory_context.total_incidents} prior incidents for "
                f"({incident.tenant_id}, {incident.type.value})"
            )
            for agent_type, stats in memory_context.agents.items():
                logger.info(
                    f"  {agent_type}: {stats.successes}/{stats.attempts} success, "
                    f"recent={stats.recent_success_rate:.2f}"
                )

        # Handle incident
        result = await commander.handle_incident(incident)
        results.append((incident, result))

        # Log result
        logger.info(
            f"Winner: {result.winning_agent_type} "
            f"(score={result.scoring_breakdown[0]['score']:.3f}, "
            f"confidence={result.winning_proposal.get('confidence', 0):.2f})"
        )

        if result.reconciliation_triggered:
            logger.info(f"Reconciliation triggered!")
            if result.reconciliation_log:
                for line in result.reconciliation_log.get("debate_log", []):
                    logger.info(f"  {line}")

        if result.fallback_used:
            logger.info(f"Fallback used: winner changed during execution")

        if result.escalation_triggered:
            logger.info(f"ESCALATION: All agents failed!")

        # Record outcome in memory
        execution = result.execution_result
        memory.record(
            tenant_id=incident.tenant_id,
            incident_type=incident.type.value,
            agent_type=result.winning_agent_type,
            success=execution.get("success", False),
            business_savings=execution.get("actual_business_savings", 0.0),
            duration=execution.get("duration", 0.0),
        )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DEMO SUMMARY")
    logger.info("=" * 80)

    agent_wins: dict[str, int] = {}
    reconciliations = sum(1 for _, r in results if r.reconciliation_triggered)

    for _, result in results:
        agent_type = result.winning_agent_type
        agent_wins[agent_type] = agent_wins.get(agent_type, 0) + 1

    logger.info(f"Total incidents: {len(results)}")
    logger.info(f"Reconciliations triggered: {reconciliations}")
    logger.info("\nAgent wins:")
    for agent_type in sorted(agent_wins.keys()):
        logger.info(f"  {agent_type}: {agent_wins[agent_type]}")

    all_agents_won = all(agent.agent_type in agent_wins for agent in agents)
    logger.info(f"\nAll agents won ≥1: {all_agents_won} ✓" if all_agents_won else "✗")
    logger.info(
        f"Reconciliation triggered ≥1: {reconciliations >= 1} ✓"
        if reconciliations >= 1
        else "✗"
    )

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(run_demo())
