"""Tests for IncidentState + IncidentStateBuilder (Phase 8)."""

from __future__ import annotations

import pytest

from self_healing_pipeline.observability import (
    IncidentState,
    IncidentStateBuilder,
    StateConstructor,
    TelemetryCollector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_collector():
    return TelemetryCollector(use_mock=True)


@pytest.fixture()
def builder_no_db(mock_collector):
    """Builder with no DB — uses defaults for all DB-sourced values."""
    return IncidentStateBuilder(mock_collector, session_factory=None)


# ---------------------------------------------------------------------------
# IncidentState construction (no DB)
# ---------------------------------------------------------------------------

class TestIncidentStateNoDb:
    def test_build_drift(self, builder_no_db):
        state = builder_no_db.build("standard", "drift")

        assert state.tenant_id == "standard"
        assert state.incident_type == "drift"
        # Telemetry values propagated
        assert state.missing_rate == 0.08
        assert state.latency_p95_ms == 85
        # Defaults applied when no DB
        assert state.baseline_auc == 0.80
        assert state.min_auc == 0.75
        assert state.current_model_version == "v1"
        assert state.previous_model_version is None

    def test_auc_drop_computed(self, builder_no_db):
        state = builder_no_db.build("standard", "drift")
        # mock telemetry auc=0.75, baseline_auc=0.80 → drop=0.05
        assert state.current_auc == pytest.approx(0.75)
        assert state.auc_drop == pytest.approx(0.05)

    def test_drifted_features_from_telemetry(self, builder_no_db):
        # mock telemetry has income=1.2, age=0.5 → only income > 1.5? no, 1.2 < 1.5
        state = builder_no_db.build("standard", "drift")
        assert isinstance(state.drifted_features, list)
        assert "age" not in state.drifted_features  # 0.5 < 1.5

    def test_max_feature_drift(self, builder_no_db):
        state = builder_no_db.build("standard", "drift")
        # mock has income=1.2, age=0.5 → max=1.2
        assert state.max_feature_drift == pytest.approx(1.2)

    def test_cost_per_1000(self, builder_no_db):
        state = builder_no_db.build("standard", "cost_threshold")
        # mock cost_per_prediction=0.002 → cost_per_1000=2.0
        assert state.cost_per_1000_predictions == pytest.approx(2.0)

    def test_cost_budget_per_1000(self, builder_no_db):
        state = builder_no_db.build("standard", "cost_threshold")
        # default daily_cost_budget=100, /50 batches = 2.0
        assert state.cost_budget_per_1000 == pytest.approx(2.0)

    def test_severity_in_range(self, builder_no_db):
        for inc_type in ("drift", "data_quality", "latency_breach", "cost_threshold"):
            state = builder_no_db.build("standard", inc_type)
            assert 0.0 <= state.severity <= 1.0, f"severity out of range for {inc_type}"

    def test_severity_components_present(self, builder_no_db):
        # Phase 9: components are per-type named keys (drift → auc_drop, drift, affected_volume)
        state = builder_no_db.build("standard", "drift")
        assert len(state.severity_components) > 0
        for v in state.severity_components.values():
            assert 0.0 <= v <= 1.0, f"component value out of [0,1]: {v}"
        assert "auc_drop" in state.severity_components
        assert "drift" in state.severity_components

    def test_historical_agent_success_empty_no_db(self, builder_no_db):
        state = builder_no_db.build("standard", "drift")
        assert state.historical_agent_success == {}

    def test_unknown_incident_type_falls_back(self, builder_no_db):
        # Should not raise; unknown type → drift severity
        state = builder_no_db.build("standard", "unknown_type")
        assert 0.0 <= state.severity <= 1.0

    def test_schema_violation_rate(self, builder_no_db):
        state = builder_no_db.build("standard", "data_quality")
        # mock schema_violations=0 → rate=0
        assert state.schema_violation_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# StateConstructor._from_incident methods
# ---------------------------------------------------------------------------

class TestStateConstructorFromIncident:

    @pytest.fixture()
    def sample_state(self, builder_no_db):
        return builder_no_db.build("standard", "drift")

    def test_threshold_from_incident(self, sample_state):
        agent_state = StateConstructor.threshold_state_from_incident(sample_state)
        assert agent_state.current_threshold == 0.50  # default (no DB)
        assert isinstance(agent_state.historical_threshold_success, float)
        assert isinstance(agent_state.to_dict(), dict)

    def test_threshold_recall_drop_computed(self, sample_state):
        agent_state = StateConstructor.threshold_state_from_incident(sample_state)
        # recall_drop = baseline_recall(0.71) - current_recall(0.71) = 0.0
        # (mock telemetry precision=0.82 = baseline, recall=0.71 = baseline)
        assert agent_state.recall_drop == pytest.approx(0.0, abs=0.01)

    def test_retrain_from_incident(self, sample_state):
        agent_state = StateConstructor.retrain_state_from_incident(sample_state)
        assert agent_state.drift_score == pytest.approx(1.2)  # max from mock
        assert agent_state.auc_drop == pytest.approx(0.05)
        assert isinstance(agent_state.affected_features, list)

    def test_retrain_model_age_from_incident(self, sample_state):
        agent_state = StateConstructor.retrain_state_from_incident(sample_state)
        # default last_training_age_days=30
        assert agent_state.model_age_days == 30

    def test_rollback_from_incident(self, sample_state):
        agent_state = StateConstructor.rollback_state_from_incident(sample_state)
        assert agent_state.current_model == "v1"  # default
        assert agent_state.previous_model == "unknown"  # no previous
        assert isinstance(agent_state.deployment_related_incident_probability, float)
        assert 0.0 <= agent_state.deployment_related_incident_probability <= 1.0

    def test_rollback_deployment_prob_recent(self, mock_collector):
        """Recent training → high deployment probability."""
        builder = IncidentStateBuilder(mock_collector)
        state = builder.build("t1", "drift")
        # Override last_training_age_days to simulate very recent training
        state.last_training_age_days = 0.5  # 12 hours ago
        agent_state = StateConstructor.rollback_state_from_incident(state)
        assert agent_state.deployment_related_incident_probability == pytest.approx(0.80)

    def test_fallback_from_incident(self, sample_state):
        agent_state = StateConstructor.fallback_state_from_incident(sample_state)
        assert agent_state.latency_p95 == pytest.approx(85.0)
        assert agent_state.missing_rate == pytest.approx(0.08)

    def test_datarepair_from_incident(self, sample_state):
        agent_state = StateConstructor.datarepair_state_from_incident(sample_state)
        assert agent_state.missing_rate == pytest.approx(0.08)
        assert agent_state.duplicate_rate == pytest.approx(0.02)
        assert agent_state.available_backup_data is True

    def test_all_to_dict(self, sample_state):
        """All from_incident methods return to_dict() without error."""
        for method in (
            StateConstructor.threshold_state_from_incident,
            StateConstructor.retrain_state_from_incident,
            StateConstructor.rollback_state_from_incident,
            StateConstructor.fallback_state_from_incident,
            StateConstructor.datarepair_state_from_incident,
        ):
            result = method(sample_state)
            d = result.to_dict()
            assert isinstance(d, dict)
            assert len(d) > 0


# ---------------------------------------------------------------------------
# DB-backed builder (SQLite in-memory)
# ---------------------------------------------------------------------------

class TestIncidentStateWithDb:

    @pytest.fixture()
    def db_factory(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from contextlib import contextmanager
        from self_healing_pipeline.db.models import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
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

    @pytest.fixture()
    def builder_with_db(self, mock_collector, db_factory):
        return IncidentStateBuilder(mock_collector, session_factory=db_factory)

    def test_build_empty_db_uses_defaults(self, builder_with_db):
        state = builder_with_db.build("tenant_x", "drift")
        # DB is empty → defaults
        assert state.baseline_auc == 0.80
        assert state.historical_agent_success == {}

    def test_policy_loaded_from_db(self, mock_collector, db_factory):
        from self_healing_pipeline.db.models import TenantPolicy
        with db_factory() as session:
            session.add(TenantPolicy(
                tenant_id="bank_a",
                min_acceptable_auc=0.82,
                max_acceptable_latency_ms=200.0,
                max_acceptable_missing_rate=0.03,
                latency_sla_ms=200.0,
                daily_cost_budget=200.0,
            ))
            session.commit()

        builder = IncidentStateBuilder(mock_collector, session_factory=db_factory)
        state = builder.build("bank_a", "drift")

        assert state.min_auc == pytest.approx(0.82)
        assert state.latency_sla_ms == pytest.approx(200.0)

    def test_model_version_from_validation_report(self, mock_collector, db_factory):
        from datetime import datetime, UTC
        from self_healing_pipeline.db.models import ModelValidationReport

        with db_factory() as session:
            session.add(ModelValidationReport(
                tenant_id="t1",
                model_version="v7",
                auc=0.84,
                precision=0.86,
                recall=0.79,
                f1_score=0.82,
                optimal_threshold=0.42,
                calibration_error=0.03,
                validated_at=datetime.now(UTC),
            ))
            session.commit()

        builder = IncidentStateBuilder(mock_collector, session_factory=db_factory)
        state = builder.build("t1", "drift")

        assert state.current_model_version == "v7"
        assert state.baseline_auc == pytest.approx(0.84)
        assert state.current_threshold == pytest.approx(0.42)
        # current_auc from mock telemetry = 0.75, baseline 0.84 → drop=0.09
        assert state.auc_drop == pytest.approx(0.09)

    def test_previous_model_version(self, mock_collector, db_factory):
        from datetime import datetime, UTC, timedelta
        from self_healing_pipeline.db.models import ModelValidationReport

        base = datetime.now(UTC)
        with db_factory() as session:
            for version, days_ago in [("v3", 10), ("v4", 2)]:
                session.add(ModelValidationReport(
                    tenant_id="t2",
                    model_version=version,
                    auc=0.80,
                    precision=0.82,
                    recall=0.71,
                    f1_score=0.76,
                    optimal_threshold=0.50,
                    calibration_error=0.05,
                    validated_at=base - timedelta(days=days_ago),
                ))
            session.commit()

        builder = IncidentStateBuilder(mock_collector, session_factory=db_factory)
        state = builder.build("t2", "drift")

        assert state.current_model_version == "v4"    # most recent
        assert state.previous_model_version == "v3"   # second most recent

    def test_historical_success_from_remediation_actions(self, mock_collector, db_factory):
        from self_healing_pipeline.db.models import RemediationAction

        with db_factory() as session:
            for success in [True, True, False, True]:
                session.add(RemediationAction(
                    incident_id="inc-1",
                    agent="retrain",
                    proposal={},
                    chosen=True,
                    reward=0.5,
                    success=success,
                ))
            session.commit()

        builder = IncidentStateBuilder(mock_collector, session_factory=db_factory)
        state = builder.build("t3", "drift")

        assert "retrain" in state.historical_agent_success
        assert state.historical_agent_success["retrain"] == pytest.approx(0.75)  # 3/4
