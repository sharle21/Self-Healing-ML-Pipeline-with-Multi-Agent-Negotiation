"""Tests for tiered memory system (Tier 1 cache + Tier 2 store)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from self_healing_pipeline.db.models import Base, DecisionOutcome
from self_healing_pipeline.memory.memory import Memory
from self_healing_pipeline.memory.tier1_cache import CacheEntry, Tier1Cache
from self_healing_pipeline.memory.tier2_store import Tier2Store


@pytest.fixture
def memory_db():
    """In-memory SQLite database for tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


class TestTier1Cache:
    """Tier 1 hot cache tests."""

    def test_init_empty(self):
        """Test cache initializes empty."""
        cache = Tier1Cache()
        assert cache.size() == 0

    def test_set_and_get(self):
        """Test set/get operations."""
        cache = Tier1Cache()
        entry = CacheEntry(
            last_winner="threshold_agent",
            success_rate=0.8,
            recent_success_rate=0.9,
            avg_time=10.5,
        )
        cache.set("standard", "DRIFT", entry)
        retrieved = cache.get("standard", "DRIFT")
        assert retrieved is not None
        assert retrieved.last_winner == "threshold_agent"
        assert retrieved.success_rate == 0.8

    def test_get_nonexistent(self):
        """Test get on nonexistent key."""
        cache = Tier1Cache()
        assert cache.get("standard", "DRIFT") is None

    def test_invalidate(self):
        """Test cache invalidation."""
        cache = Tier1Cache()
        entry = CacheEntry("agent", 0.8, 0.9, 10.0)
        cache.set("standard", "DRIFT", entry)
        assert cache.size() == 1
        cache.invalidate("standard", "DRIFT")
        assert cache.size() == 0
        assert cache.get("standard", "DRIFT") is None

    def test_clear(self):
        """Test clear all cache."""
        cache = Tier1Cache()
        cache.set("standard", "DRIFT", CacheEntry("a", 0.8, 0.9, 10.0))
        cache.set("enterprise", "COST", CacheEntry("b", 0.7, 0.8, 20.0))
        assert cache.size() == 2
        cache.clear()
        assert cache.size() == 0


class TestTier2Store:
    """Tier 2 structured store tests."""

    def test_record_outcome_creates_entry(self, memory_db):
        """Test recording outcome creates decision_outcome row."""
        store = Tier2Store(memory_db)
        store.record_outcome(
            tenant_id="standard",
            incident_type="DRIFT",
            agent_type="retrain",
            success=True,
            business_savings=500.0,
            duration=45.5,
        )
        # Verify decision_outcome was created
        rows = memory_db.query(DecisionOutcome).all()
        assert len(rows) == 1
        assert rows[0].agent_type == "retrain"
        assert rows[0].success is True

    def test_get_agent_stats_empty(self, memory_db):
        """Test getting stats for unknown (tenant, type) pair."""
        store = Tier2Store(memory_db)
        stats = store.get_agent_stats("standard", "DRIFT")
        assert stats == {}

    def test_get_agent_stats_single(self, memory_db):
        """Test getting stats after one outcome."""
        store = Tier2Store(memory_db)
        store.record_outcome(
            "standard", "DRIFT", "retrain", True, 500.0, 45.5
        )
        stats = store.get_agent_stats("standard", "DRIFT")
        assert "retrain" in stats
        assert stats["retrain"].attempts == 1
        assert stats["retrain"].successes == 1
        assert stats["retrain"].success_rate == 1.0

    def test_get_agent_stats_multiple(self, memory_db):
        """Test getting stats with multiple outcomes."""
        store = Tier2Store(memory_db)
        # 3 successful, 1 failed
        store.record_outcome("standard", "DRIFT", "retrain", True, 500.0, 45.0)
        store.record_outcome("standard", "DRIFT", "retrain", True, 450.0, 50.0)
        store.record_outcome("standard", "DRIFT", "retrain", True, 400.0, 48.0)
        store.record_outcome("standard", "DRIFT", "retrain", False, 0.0, 120.0)

        stats = store.get_agent_stats("standard", "DRIFT")
        retrain_stats = stats["retrain"]
        assert retrain_stats.attempts == 4
        assert retrain_stats.successes == 3
        assert retrain_stats.success_rate == 0.75
        assert retrain_stats.avg_business_savings == pytest.approx(337.5)

    def test_recent_success_rate(self, memory_db):
        """Test recent_success_rate tracks last 5 outcomes."""
        store = Tier2Store(memory_db)
        # Record 7 outcomes (success/fail pattern)
        outcomes = [True, False, True, False, True, False, True]
        for outcome in outcomes:
            store.record_outcome(
                "standard", "DRIFT", "retrain", outcome, 100.0, 10.0
            )

        stats = store.get_agent_stats("standard", "DRIFT")
        retrain_stats = stats["retrain"]
        # Recent 5 (last 5 of 7): [True, False, True, False, True] (indices 2-6)
        # Successes: True (2), True (4), True (6) = 3 out of 5 = 0.6
        assert retrain_stats.recent_success_rate == pytest.approx(0.6)

    def test_multiple_agents_same_incident(self, memory_db):
        """Test multiple agents with different stats."""
        store = Tier2Store(memory_db)
        # Retrain: 3/4 successes
        for i in range(4):
            store.record_outcome(
                "standard", "DRIFT", "retrain", i < 3, 100.0, 40.0
            )
        # Threshold: 2/2 successes
        for i in range(2):
            store.record_outcome(
                "standard", "DRIFT", "threshold", True, 50.0, 5.0
            )

        stats = store.get_agent_stats("standard", "DRIFT")
        assert len(stats) == 2
        assert stats["retrain"].success_rate == 0.75
        assert stats["threshold"].success_rate == 1.0


class TestUnifiedMemory:
    """Unified Memory interface tests."""

    def test_recall_cold_start(self, memory_db):
        """Test recall on empty database."""
        mem = Memory(memory_db)
        result = mem.recall("standard", "DRIFT")
        assert result.cold_start is True
        assert result.agents == {}
        assert result.total_incidents == 0

    def test_record_and_recall(self, memory_db):
        """Test record followed by recall."""
        mem = Memory(memory_db)
        # Record an outcome
        mem.record(
            "standard", "DRIFT", "retrain", True, 500.0, 45.0
        )
        # Recall should not be cold start
        result = mem.recall("standard", "DRIFT")
        assert result.cold_start is False
        assert result.total_incidents == 1
        assert "retrain" in result.agents

    def test_tier1_cache_hit(self, memory_db):
        """Test Tier 1 cache is populated after recall."""
        mem = Memory(memory_db)
        mem.record("standard", "DRIFT", "retrain", True, 500.0, 45.0)
        # First recall populates cache
        mem.recall("standard", "DRIFT")
        # Cache should have entry
        assert mem.tier1.size() == 1
        # Second recall hits cache (no DB query)
        result = mem.recall("standard", "DRIFT")
        assert result.cold_start is False

    def test_invalidation_after_record(self, memory_db):
        """Test Tier 1 cache is invalidated after record."""
        mem = Memory(memory_db)
        mem.record("standard", "DRIFT", "retrain", True, 500.0, 45.0)
        mem.recall("standard", "DRIFT")
        # Cache has entry
        assert mem.tier1.size() == 1
        # Record another outcome
        mem.record("standard", "DRIFT", "threshold", True, 100.0, 10.0)
        # Cache should be invalidated
        assert mem.tier1.size() == 0

    def test_scoped_recall(self, memory_db):
        """Test recall is scoped by (tenant_id, incident_type)."""
        mem = Memory(memory_db)
        mem.record("standard", "DRIFT", "retrain", True, 500.0, 45.0)
        mem.record("enterprise", "COST", "threshold", True, 1000.0, 5.0)

        drift_result = mem.recall("standard", "DRIFT")
        cost_result = mem.recall("enterprise", "COST")

        assert drift_result.total_incidents == 1
        assert cost_result.total_incidents == 1
        assert drift_result.agents["retrain"].attempts == 1
        assert cost_result.agents["threshold"].attempts == 1
