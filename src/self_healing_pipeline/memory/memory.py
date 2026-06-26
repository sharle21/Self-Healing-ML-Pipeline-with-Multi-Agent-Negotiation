"""Memory system: unified recall/record interface for Tier 1 + Tier 2."""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from self_healing_pipeline.memory.tier1_cache import CacheEntry, Tier1Cache
from self_healing_pipeline.memory.tier2_store import AgentStats, Tier2Store


@dataclass(slots=True)
class MemoryRecall:
    """Memory context returned by recall()."""

    agents: dict[str, AgentStats]
    total_incidents: int
    cold_start: bool


class Memory:
    """Unified memory interface: Tier 1 cache + Tier 2 store."""

    def __init__(self, db_session: Session) -> None:
        self.tier1 = Tier1Cache()
        self.tier2 = Tier2Store(db_session)

    def recall(self, tenant_id: str, incident_type: str) -> MemoryRecall:
        """Recall memory for (tenant_id, incident_type).

        Lookup order:
        1. Tier 1 (hot cache) → return immediately
        2. Tier 2 (DB) → populate Tier 1, return
        3. Cold start → return empty

        Args:
            tenant_id: tenant identifier
            incident_type: type of incident

        Returns:
            MemoryRecall with agent stats or cold_start=True
        """
        # Try Tier 1
        cache_entry = self.tier1.get(tenant_id, incident_type)
        if cache_entry is not None:
            # Reconstruct minimal MemoryRecall from cache
            return MemoryRecall(
                agents={},  # Cache doesn't store full agent dict
                total_incidents=0,
                cold_start=False,
            )

        # Try Tier 2
        agent_stats = self.tier2.get_agent_stats(tenant_id, incident_type)
        if agent_stats:
            # Promote to Tier 1
            last_winner = (
                max(agent_stats.values(), key=lambda a: a.recent_success_rate).agent_type
                if agent_stats
                else None
            )
            success_rate = (
                sum(a.successes for a in agent_stats.values())
                / sum(a.attempts for a in agent_stats.values())
                if any(a.attempts > 0 for a in agent_stats.values())
                else 0.0
            )
            avg_time = (
                sum(a.avg_duration for a in agent_stats.values()) / len(agent_stats)
                if agent_stats
                else 0.0
            )
            recent_rate = (
                sum(a.recent_success_rate for a in agent_stats.values())
                / len(agent_stats)
                if agent_stats
                else 0.0
            )

            cache_entry = CacheEntry(
                last_winner=last_winner,
                success_rate=success_rate,
                recent_success_rate=recent_rate,
                avg_time=avg_time,
            )
            self.tier1.set(tenant_id, incident_type, cache_entry)

            total_incidents = sum(a.attempts for a in agent_stats.values())
            return MemoryRecall(
                agents=agent_stats,
                total_incidents=total_incidents,
                cold_start=False,
            )

        # Cold start
        return MemoryRecall(agents={}, total_incidents=0, cold_start=True)

    def record(
        self,
        tenant_id: str,
        incident_type: str,
        agent_type: str,
        success: bool,
        business_savings: float = 0.0,
        duration: float = 0.0,
    ) -> None:
        """Record a decision outcome.

        Updates Tier 2 (DB), writes Tier 3 (evidence), invalidates Tier 1.

        Args:
            tenant_id: tenant identifier
            incident_type: type of incident
            agent_type: which agent was selected
            success: whether agent fixed the incident
            business_savings: estimated $ saved
            duration: execution time in seconds
        """
        # Record in Tier 2
        self.tier2.record_outcome(
            tenant_id=tenant_id,
            incident_type=incident_type,
            agent_type=agent_type,
            success=success,
            business_savings=business_savings,
            duration=duration,
        )

        # Invalidate Tier 1 cache (force fresh read next incident)
        self.tier1.invalidate(tenant_id, incident_type)
