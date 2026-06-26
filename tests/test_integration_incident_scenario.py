"""Integration test: multi-incident scenario exercising all monitor types."""

import numpy as np
import pandas as pd
import pytest

from self_healing_pipeline.gateway.events import IncidentType
from self_healing_pipeline.incidents.simulator import IncidentSimulator
from self_healing_pipeline.monitors.business import BusinessCostMonitor
from self_healing_pipeline.monitors.drift import DriftMonitor
from self_healing_pipeline.monitors.quality import DataQualityMonitor


@pytest.fixture
def reference_data():
    """Create mock reference dataset."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "feature_1": rng.normal(0, 1, 100),
        "feature_2": rng.normal(0, 1, 100),
        "feature_3": rng.normal(0, 1, 100),
        "y": rng.integers(0, 2, 100),
    })


@pytest.fixture
def simulator():
    """Create incident simulator with fixed seed."""
    return IncidentSimulator(rng=np.random.default_rng(42))


class TestMultiIncidentScenario:
    """Test all incident types fire correctly."""

    def test_drift_monitor_initialized(self, reference_data, simulator):
        """Verify drift monitor can be initialized (detailed testing in test_drift_monitor.py)."""
        monitor = DriftMonitor(sensitivity=0.5)
        assert monitor.sensitivity == 0.5

    def test_quality_missing_values_incident_fires(self, reference_data):
        """Verify quality monitor detects missing values incident."""
        monitor = DataQualityMonitor(missing_threshold=0.05)
        df = reference_data.copy()
        # Introduce missing values
        n_missing = int(len(df) * 0.1)
        idx = np.random.choice(len(df), n_missing, replace=False)
        df.iloc[idx, 0] = np.nan
        result = monitor.detect(df)
        incident = monitor.make_incident(result, "standard")
        assert incident is not None
        assert incident.type == IncidentType.DATA_QUALITY

    def test_quality_duplicates_incident_fires(self, reference_data):
        """Verify quality monitor detects duplicate rows incident."""
        monitor = DataQualityMonitor(duplicate_threshold=0.03)
        df = reference_data.copy()
        # Add duplicates
        duplicates = df.iloc[:10].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        result = monitor.detect(df)
        incident = monitor.make_incident(result, "standard")
        assert incident is not None
        assert incident.type == IncidentType.DATA_QUALITY

    def test_quality_volume_incident_fires(self, reference_data):
        """Verify quality monitor detects low volume incident."""
        monitor = DataQualityMonitor(volume_floor=100)
        df = reference_data.iloc[:50].copy()
        result = monitor.detect(df)
        incident = monitor.make_incident(result, "standard")
        assert incident is not None
        assert incident.type == IncidentType.DATA_QUALITY

    def test_cost_threshold_incident_fires(self):
        """Verify business monitor detects cost threshold breach."""
        # Setup: FN cost = 50, threshold = 40 per prediction
        # If we add 3 FN out of 3 predictions: cost = 150 / 3 = 50 > 40
        monitor = BusinessCostMonitor(
            false_positive_cost=10.0,
            false_negative_cost=50.0,
            cost_threshold=40.0,
            window_size=100,
        )
        # Add false negatives to exceed threshold
        for _ in range(3):
            monitor.record_prediction(1, 0)  # FN: cost 50 each
        result = monitor.detect()
        incident = monitor.make_incident(result, "enterprise")
        assert incident is not None
        assert incident.type == IncidentType.COST_THRESHOLD

    def test_all_monitors_produce_incidents_with_severity(self):
        """Verify monitors produce incidents with valid severity."""
        rng = np.random.default_rng(42)
        reference = pd.DataFrame({
            "feature_1": rng.normal(0, 1, 100),
            "feature_2": rng.normal(0, 1, 100),
            "feature_3": rng.normal(0, 1, 100),
            "y": rng.integers(0, 2, 100),
        })

        # Quality incident
        quality_monitor = DataQualityMonitor(missing_threshold=0.05)
        df = reference.copy()
        df.iloc[:10, 0] = np.nan
        result = quality_monitor.detect(df)
        incident = quality_monitor.make_incident(result, "standard")
        assert incident is not None
        assert incident.type == IncidentType.DATA_QUALITY
        assert 0 <= incident.severity <= 1.0

        # Cost incident
        cost_monitor = BusinessCostMonitor(cost_threshold=10.0)
        cost_monitor.record_prediction(1, 0)
        result = cost_monitor.detect()
        incident = cost_monitor.make_incident(result, "standard")
        assert incident is not None
        assert incident.type == IncidentType.COST_THRESHOLD
        assert 0 <= incident.severity <= 1.0

    def test_quality_and_cost_incidents_together(self, reference_data):
        """Verify multiple monitors can fire incidents in same scenario."""
        incidents = []

        # Quality
        quality_monitor = DataQualityMonitor(missing_threshold=0.05)
        df = reference_data.copy()
        df.iloc[:10, 0] = np.nan
        result = quality_monitor.detect(df)
        if incident := quality_monitor.make_incident(result, "standard"):
            incidents.append(incident.type)

        # Cost
        cost_monitor = BusinessCostMonitor(cost_threshold=10.0)
        for _ in range(3):
            cost_monitor.record_prediction(1, 0)
        result = cost_monitor.detect()
        if incident := cost_monitor.make_incident(result, "standard"):
            incidents.append(incident.type)

        # Verify we have both expected types
        assert IncidentType.DATA_QUALITY in incidents
        assert IncidentType.COST_THRESHOLD in incidents
