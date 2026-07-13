"""Tests for observation layer (telemetry, severity, state)."""

import pytest

from self_healing_pipeline.gateway.events import IncidentType
from self_healing_pipeline.observability import (
    SeverityCalculator,
    StateConstructor,
    TelemetryCollector,
)


class TestTelemetryCollector:
    """Test telemetry collection."""

    def test_collect_mock_telemetry(self):
        """Test collecting mock telemetry."""
        collector = TelemetryCollector(use_mock=True)
        telemetry = collector.collect()

        assert telemetry.model.auc == 0.75
        assert telemetry.data.missing_rate == 0.08
        assert telemetry.system.latency_p95 == 85


class TestSeverityCalculator:
    """Test severity calculation."""

    def test_drift_severity(self):
        """Test drift incident severity calculation."""
        collector = TelemetryCollector(use_mock=True)
        telemetry = collector.collect()
        calculator = SeverityCalculator()

        severity, breakdown = calculator.calculate(IncidentType.DRIFT, telemetry)

        assert 0 <= severity <= 1
        assert 0 <= breakdown.impact <= 1
        assert 0 <= breakdown.deviation <= 1
        assert 0 <= breakdown.urgency <= 1
        assert 0 <= breakdown.business_risk <= 1

    def test_data_quality_severity(self):
        """Test data quality incident severity."""
        collector = TelemetryCollector(use_mock=True)
        telemetry = collector.collect()
        calculator = SeverityCalculator()

        severity, breakdown = calculator.calculate(IncidentType.DATA_QUALITY, telemetry)

        assert 0 <= severity <= 1

    def test_cost_severity(self):
        """Test cost threshold incident severity."""
        collector = TelemetryCollector(use_mock=True)
        telemetry = collector.collect()
        calculator = SeverityCalculator()

        severity, breakdown = calculator.calculate(IncidentType.COST_THRESHOLD, telemetry)

        assert 0 <= severity <= 1

    def test_latency_severity(self):
        """Test latency breach incident severity."""
        collector = TelemetryCollector(use_mock=True)
        telemetry = collector.collect()
        calculator = SeverityCalculator()

        severity, breakdown = calculator.calculate(IncidentType.LATENCY_BREACH, telemetry)

        assert 0 <= severity <= 1


class TestStateConstructor:
    """Test state construction."""

    def test_threshold_agent_state(self):
        """Test threshold agent state construction."""
        collector = TelemetryCollector(use_mock=True)
        telemetry = collector.collect()

        state = StateConstructor.threshold_state(telemetry)

        assert state.current_threshold == 0.50
        assert state.historical_threshold_success == 0.75
        assert isinstance(state.to_dict(), dict)

    def test_retrain_agent_state(self):
        """Test retrain agent state construction."""
        collector = TelemetryCollector(use_mock=True)
        telemetry = collector.collect()

        state = StateConstructor.retrain_state(telemetry)

        assert state.model_age_days == 30
        assert state.historical_retrain_success == 0.72
        assert isinstance(state.affected_features, list)

    def test_rollback_agent_state(self):
        """Test rollback agent state construction."""
        collector = TelemetryCollector(use_mock=True)
        telemetry = collector.collect()

        state = StateConstructor.rollback_state(telemetry)

        assert state.current_model == "v13"
        assert state.previous_model == "v12"
        assert state.historical_rollback_success == 0.91

    def test_fallback_agent_state(self):
        """Test fallback agent state construction."""
        collector = TelemetryCollector(use_mock=True)
        telemetry = collector.collect()

        state = StateConstructor.fallback_state(telemetry)

        assert state.fallback_quality == 0.70
        assert state.historical_fallback_success == 0.85

    def test_datarepair_agent_state(self):
        """Test data repair agent state construction."""
        collector = TelemetryCollector(use_mock=True)
        telemetry = collector.collect()

        state = StateConstructor.datarepair_state(telemetry)

        assert state.available_backup_data is True
        assert state.historical_repair_success == 0.70
        assert isinstance(state.affected_features, list)
