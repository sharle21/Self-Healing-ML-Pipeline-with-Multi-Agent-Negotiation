"""Tests for DriftMonitor."""

from __future__ import annotations

import pandas as pd
import pytest

from self_healing_pipeline.gateway.events import IncidentType
from self_healing_pipeline.monitors.drift import DriftMonitor


@pytest.fixture
def reference_df() -> pd.DataFrame:
    """Create reference dataset."""
    return pd.DataFrame(
        {
            "amount": [100, 200, 300, 400, 500] * 20,
            "age": [25, 35, 45, 55, 65] * 20,
            "income": [30000, 50000, 70000, 90000, 110000] * 20,
        }
    )


@pytest.fixture
def current_no_drift_df(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Create current dataset with no drift."""
    return reference_df.copy()


@pytest.fixture
def current_with_drift_df(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Create current dataset with significant drift."""
    df = reference_df.copy()
    df["income"] = df["income"] * 2.0
    return df


def test_drift_monitor_init() -> None:
    """Test monitor initialization."""
    monitor = DriftMonitor(sensitivity=0.5)
    assert monitor.sensitivity == 0.5


def test_drift_monitor_invalid_sensitivity() -> None:
    """Test invalid sensitivity raises error."""
    with pytest.raises(ValueError, match="sensitivity must be in"):
        DriftMonitor(sensitivity=1.5)


def test_no_drift_detected(reference_df: pd.DataFrame, current_no_drift_df: pd.DataFrame) -> None:
    """Test drift detection when no drift present."""
    monitor = DriftMonitor(sensitivity=0.5)
    result = monitor.detect(reference_df, current_no_drift_df)

    assert not result.drift_detected
    assert result.drift_percentage == 0.0
    assert result.drifted_features == []


def test_drift_detected(reference_df: pd.DataFrame, current_with_drift_df: pd.DataFrame) -> None:
    """Test drift detection when drift present."""
    monitor = DriftMonitor(sensitivity=0.3)
    result = monitor.detect(reference_df, current_with_drift_df)

    assert result.drift_detected
    assert result.drift_percentage > 0.0
    assert len(result.drifted_features) > 0


def test_empty_dataframes() -> None:
    """Test detection with empty dataframes."""
    monitor = DriftMonitor()
    df_empty = pd.DataFrame()

    result = monitor.detect(df_empty, df_empty)
    assert not result.drift_detected
    assert result.drift_percentage == 0.0


def test_make_incident_no_drift(reference_df: pd.DataFrame, current_no_drift_df: pd.DataFrame) -> None:
    """Test incident creation when no drift."""
    monitor = DriftMonitor(sensitivity=0.5)
    result = monitor.detect(reference_df, current_no_drift_df)

    incident = monitor.make_incident(result, tenant_id="test")
    assert incident is None


def test_make_incident_with_drift(
    reference_df: pd.DataFrame, current_with_drift_df: pd.DataFrame
) -> None:
    """Test incident creation when drift detected."""
    monitor = DriftMonitor(sensitivity=0.3)
    result = monitor.detect(reference_df, current_with_drift_df)

    incident = monitor.make_incident(result, tenant_id="enterprise")
    assert incident is not None
    assert incident.tenant_id == "enterprise"
    assert incident.type == IncidentType.DRIFT
    assert incident.severity > 0.0
    assert len(incident.affected_features) > 0
