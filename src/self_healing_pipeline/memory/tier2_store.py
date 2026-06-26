"""Tier 2: Structured summaries from DB (agent_summary table)."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from self_healing_pipeline.db.models import AgentSummary, DecisionOutcome


@dataclass(slots=True)
class AgentStats:
    """Stats for a single agent across recent outcomes."""

    agent_type: str
    attempts: int
    successes: int
    success_rate: float
    recent_success_rate: float
    avg_business_savings: float
    avg_duration: float


class Tier2Store:
    """Structured summaries: agent_summary table with pre-aggregated stats."""

    def __init__(self, db_session: Session) -> None:
        self.db = db_session

    def record_outcome(
        self,
        tenant_id: str,
        incident_type: str,
        agent_type: str,
        success: bool,
        business_savings: float = 0.0,
        duration: float = 0.0,
    ) -> None:
        """Record a decision outcome (Tier 2 + Tier 3).

        Args:
            tenant_id: tenant identifier
            incident_type: type of incident
            agent_type: which agent was selected
            success: whether agent fixed the incident
            business_savings: estimated $ saved
            duration: execution time in seconds
        """
        # Record in decision_outcomes (Tier 3 source of truth)
        outcome = DecisionOutcome(
            tenant_id=tenant_id,
            incident_type=incident_type,
            agent_type=agent_type,
            success=success,
            business_savings=business_savings,
            duration=duration,
        )
        self.db.add(outcome)
        self.db.commit()

        # Update agent_summary via UPSERT
        self._upsert_agent_summary(
            tenant_id, incident_type, agent_type, success, business_savings, duration
        )

    def get_agent_stats(
        self, tenant_id: str, incident_type: str
    ) -> dict[str, AgentStats]:
        """Get all agent stats for (tenant_id, incident_type).

        Returns:
            Dict mapping agent_type → AgentStats
        """
        stmt = select(AgentSummary).where(
            (AgentSummary.tenant_id == tenant_id)
            & (AgentSummary.incident_type == incident_type)
        )
        rows = self.db.execute(stmt).scalars().all()

        result = {}
        for row in rows:
            recent_success_rate = (
                sum(row.recent_outcomes) / len(row.recent_outcomes)
                if row.recent_outcomes
                else 0.0
            )
            success_rate = (
                row.successes / row.attempts if row.attempts > 0 else 0.0
            )
            avg_savings = (
                row.sum_business_savings / row.attempts if row.attempts > 0 else 0.0
            )
            avg_duration = (
                row.sum_duration / row.attempts if row.attempts > 0 else 0.0
            )

            result[row.agent_type] = AgentStats(
                agent_type=row.agent_type,
                attempts=row.attempts,
                successes=row.successes,
                success_rate=success_rate,
                recent_success_rate=recent_success_rate,
                avg_business_savings=avg_savings,
                avg_duration=avg_duration,
            )

        return result

    def _upsert_agent_summary(
        self,
        tenant_id: str,
        incident_type: str,
        agent_type: str,
        success: bool,
        business_savings: float,
        duration: float,
    ) -> None:
        """Update agent_summary via insert-or-update."""
        # Get existing row
        stmt = select(AgentSummary).where(
            (AgentSummary.tenant_id == tenant_id)
            & (AgentSummary.incident_type == incident_type)
            & (AgentSummary.agent_type == agent_type)
        )
        existing = self.db.execute(stmt).scalars().first()

        if existing:
            # Update existing
            existing.attempts += 1
            existing.successes += int(success)
            existing.sum_business_savings += business_savings
            existing.sum_duration += duration
            # Keep last 5 outcomes - create new list to trigger SQLAlchemy update
            recent = existing.recent_outcomes.copy() if existing.recent_outcomes else []
            recent.append(success)
            if len(recent) > 5:
                recent.pop(0)
            existing.recent_outcomes = recent  # Reassign to trigger update
        else:
            # Create new
            new_row = AgentSummary(
                tenant_id=tenant_id,
                incident_type=incident_type,
                agent_type=agent_type,
                attempts=1,
                successes=int(success),
                sum_business_savings=business_savings,
                sum_duration=duration,
                recent_outcomes=[success],
            )
            self.db.add(new_row)

        self.db.commit()
