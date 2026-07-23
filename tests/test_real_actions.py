"""Tests for Phase 12: real agent actions (ThresholdAgent, RetrainAgent, RollbackAgent)."""

from __future__ import annotations

import asyncio
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from self_healing_pipeline.db.models import Base, TenantThresholdOverride


# ---------------------------------------------------------------------------
# Shared DB fixture (in-memory SQLite, all tables created)
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session_factory(engine):
    Session = sessionmaker(bind=engine)

    @contextmanager
    def factory():
        s = Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return factory


# ---------------------------------------------------------------------------
# ThresholdAdjustmentAgent
# ---------------------------------------------------------------------------

class TestThresholdAgentReal:
    from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent

    @pytest.fixture()
    def agent(self, session_factory):
        from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
        return ThresholdAdjustmentAgent("test-threshold", session_factory=session_factory)

    @pytest.fixture()
    def agent_no_db(self):
        from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
        return ThresholdAdjustmentAgent("test-threshold-nodb")

    def test_can_handle_high_fn_cost(self, agent):
        assert agent.can_handle({"cost_false_negative": 500, "recall_drop": 0.0})

    def test_can_handle_recall_drop(self, agent):
        assert agent.can_handle({"recall_drop": 0.10, "cost_false_negative": 0})

    def test_cannot_handle_no_signal(self, agent):
        assert not agent.can_handle({"recall_drop": 0.01, "cost_false_negative": 50})

    def test_analyze_produces_new_threshold(self, agent):
        state = {
            "current_threshold": 0.5,
            "recall_drop": 0.15,
            "cost_false_negative": 500,
            "tenant_id": "standard",
        }
        plan = asyncio.run(agent.analyze(state))
        assert plan.action == "change_threshold"
        assert "new_threshold" in plan.expected_effect
        new_t = plan.expected_effect["new_threshold"]
        assert 0.1 <= new_t <= 0.9
        # high recall drop → threshold should decrease
        assert new_t < 0.5

    def test_analyze_includes_tenant_id(self, agent):
        state = {
            "current_threshold": 0.5,
            "recall_drop": 0.15,
            "cost_false_negative": 500,
            "tenant_id": "enterprise",
        }
        plan = asyncio.run(agent.analyze(state))
        assert plan.expected_effect["tenant_id"] == "enterprise"

    def test_execute_writes_to_db(self, agent, session_factory):
        state = {
            "current_threshold": 0.5,
            "recall_drop": 0.15,
            "cost_false_negative": 500,
            "tenant_id": "standard",
        }
        plan = asyncio.run(agent.analyze(state))
        result = asyncio.run(agent.execute(plan))

        assert result.success is True
        new_t = plan.expected_effect["new_threshold"]

        # Verify it's actually in the DB
        with session_factory() as session:
            row = session.get(TenantThresholdOverride, "standard")
            assert row is not None
            assert row.threshold == pytest.approx(new_t)
            assert row.updated_by == "test-threshold"

    def test_execute_updates_existing_row(self, agent, session_factory):
        """Second execute updates the existing row rather than creating a duplicate."""
        state = {
            "current_threshold": 0.5,
            "recall_drop": 0.15,
            "cost_false_negative": 500,
            "tenant_id": "free",
        }
        plan1 = asyncio.run(agent.analyze(state))
        asyncio.run(agent.execute(plan1))

        # Execute again with different threshold
        state2 = {**state, "current_threshold": 0.45}
        plan2 = asyncio.run(agent.analyze(state2))
        asyncio.run(agent.execute(plan2))

        with session_factory() as session:
            rows = session.query(TenantThresholdOverride).filter_by(tenant_id="free").all()
            assert len(rows) == 1  # upsert, not insert

    def test_execute_no_db_succeeds_silently(self, agent_no_db):
        """When no session_factory, execute succeeds without writing to DB."""
        state = {
            "current_threshold": 0.5,
            "recall_drop": 0.15,
            "cost_false_negative": 500,
            "tenant_id": "standard",
        }
        plan = asyncio.run(agent_no_db.analyze(state))
        result = asyncio.run(agent_no_db.execute(plan))
        assert result.success is True  # simulated success

    def test_execute_missing_threshold_fails(self, agent):
        from self_healing_pipeline.agents.remediation_policy import RemediationPlan
        bad_plan = RemediationPlan(
            agent_type="threshold",
            action="change_threshold",
            confidence=0.8,
            expected_effect={"tenant_id": "standard"},  # missing new_threshold
            reasoning="bad plan",
            cost="$0",
            execution_time="1s",
        )
        result = asyncio.run(agent.execute(bad_plan))
        assert result.success is False
        assert "new_threshold missing" in result.error


# ---------------------------------------------------------------------------
# ModelServer: live threshold read
# ---------------------------------------------------------------------------

class TestModelServerLiveThreshold:

    @pytest.fixture()
    def tmp_db(self, tmp_path, engine):
        """Write TenantThresholdOverride to a temp SQLite file for ModelServer to read."""
        db_file = tmp_path / "test.db"
        import sqlite3
        Base.metadata.create_all(engine)
        # Create the table in a real file-based DB
        file_engine = create_engine(f"sqlite:///{db_file}")
        Base.metadata.create_all(file_engine)
        return str(db_file)

    def test_reads_live_threshold(self, tmp_db):
        import sqlite3
        # Write override directly
        with sqlite3.connect(tmp_db) as conn:
            conn.execute(
                "INSERT INTO tenant_threshold_overrides(tenant_id, threshold, updated_by, updated_at)"
                " VALUES (?, ?, ?, ?)",
                ("standard", 0.35, "test", datetime.now(UTC).isoformat()),
            )

        from self_healing_pipeline.pipeline.serving import ModelServer
        # We can test _live_threshold without a real model
        class _Stub(ModelServer):
            pass

        server = object.__new__(_Stub)
        server._db_path = tmp_db

        result = server._live_threshold("standard", 0.5)
        assert result == pytest.approx(0.35)

    def test_falls_back_to_default_when_no_override(self, tmp_db):
        from self_healing_pipeline.pipeline.serving import ModelServer

        server = object.__new__(ModelServer)
        server._db_path = tmp_db

        result = server._live_threshold("enterprise", 0.8)
        assert result == pytest.approx(0.8)  # no override → default

    def test_falls_back_when_db_path_none(self):
        from self_healing_pipeline.pipeline.serving import ModelServer

        server = object.__new__(ModelServer)
        server._db_path = None

        result = server._live_threshold("standard", 0.6)
        assert result == pytest.approx(0.6)

    def test_falls_back_on_bad_db_path(self):
        from self_healing_pipeline.pipeline.serving import ModelServer

        server = object.__new__(ModelServer)
        server._db_path = "/nonexistent/path/db.sqlite"

        result = server._live_threshold("standard", 0.55)
        assert result == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# RetrainAgent: simulated path (real training not run in unit tests)
# ---------------------------------------------------------------------------

class TestRetrainAgentSimulated:

    @pytest.fixture()
    def agent_simulated(self):
        from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
        return RetrainAgent("retrain-sim")  # no model_path → simulated

    def test_can_handle_drift(self, agent_simulated):
        assert agent_simulated.can_handle({"drift_score": 2.0, "data_quality_score": 0.9})

    def test_can_handle_auc_drop(self, agent_simulated):
        assert agent_simulated.can_handle({"auc_drop": 0.08, "data_quality_score": 0.9})

    def test_cannot_handle_poor_data_quality(self, agent_simulated):
        """Don't retrain when data is too corrupted."""
        assert not agent_simulated.can_handle({
            "drift_score": 2.0, "auc_drop": 0.1, "data_quality_score": 0.3
        })

    def test_analyze_returns_plan(self, agent_simulated):
        state = {
            "drift_score": 1.8,
            "auc_drop": 0.06,
            "data_quality_score": 0.92,
            "model_age_days": 45,
        }
        plan = asyncio.run(agent_simulated.analyze(state))
        assert plan.action == "retrain_model"
        assert 0 <= plan.confidence <= 1
        assert "auc_delta" in plan.expected_effect  # Phase 10: renamed from auc_recovery

    def test_execute_simulated_succeeds(self, agent_simulated):
        state = {"drift_score": 1.8, "auc_drop": 0.06, "data_quality_score": 0.92}
        plan = asyncio.run(agent_simulated.analyze(state))
        result = asyncio.run(agent_simulated.execute(plan))
        assert result.success is True
        assert "[simulated]" in result.logs[0]


# ---------------------------------------------------------------------------
# RollbackAgent: real path (file-based)
# ---------------------------------------------------------------------------

class TestRollbackAgentReal:

    @pytest.fixture()
    def model_dir(self, tmp_path):
        """Create fake model + backup files."""
        main = tmp_path / "model.joblib"
        backup = tmp_path / "model.backup.joblib"
        main.write_text("main_model_v2")
        backup.write_text("main_model_v1")
        return tmp_path, main, backup

    @pytest.fixture()
    def agent_real(self, model_dir, session_factory):
        from self_healing_pipeline.agents.rollback_v2 import RollbackAgent
        _, main, _ = model_dir
        return RollbackAgent("rollback-real", model_path=main, session_factory=session_factory)

    @pytest.fixture()
    def agent_simulated(self):
        from self_healing_pipeline.agents.rollback_v2 import RollbackAgent
        return RollbackAgent("rollback-sim")

    @pytest.fixture()
    def rollback_plan(self, agent_simulated):
        state = {
            "current_auc": 0.68,
            "previous_auc": 0.77,
            "deployment_age_hours": 4,
            "current_error_rate": 0.18,
            "previous_error_rate": 0.09,
        }
        return asyncio.run(agent_simulated.analyze(state))

    def test_can_handle_recent_deployment_auc_regression(self, agent_simulated):
        assert agent_simulated.can_handle({
            "deployment_age_hours": 4,
            "current_auc": 0.68,
            "previous_auc": 0.77,
        })

    def test_cannot_handle_old_deployment(self, agent_simulated):
        assert not agent_simulated.can_handle({
            "deployment_age_hours": 48,
            "current_auc": 0.68,
            "previous_auc": 0.77,
        })

    def test_execute_simulated(self, agent_simulated, rollback_plan):
        result = asyncio.run(agent_simulated.execute(rollback_plan))
        assert result.success is True
        assert "[simulated]" in result.logs[0]

    def test_execute_real_restores_backup(self, agent_real, model_dir, rollback_plan):
        _, main, backup = model_dir
        assert main.read_text() == "main_model_v2"

        result = asyncio.run(agent_real.execute(rollback_plan))

        assert result.success is True
        # main file should now contain backup content
        assert main.read_text() == "main_model_v1"

    def test_execute_fails_when_no_backup(self, tmp_path, session_factory, rollback_plan):
        from self_healing_pipeline.agents.rollback_v2 import RollbackAgent
        main = tmp_path / "model.joblib"
        main.write_text("current_model")
        # No backup file

        agent = RollbackAgent("rb-nobackup", model_path=main, session_factory=session_factory)
        result = asyncio.run(agent.execute(rollback_plan))
        assert result.success is False
        assert "no backup found" in result.error
