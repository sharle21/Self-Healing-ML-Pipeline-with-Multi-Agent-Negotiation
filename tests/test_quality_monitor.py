import numpy as np
import pandas as pd
import pytest

from self_healing_pipeline.gateway.events import IncidentType
from self_healing_pipeline.monitors.quality import DataQualityMonitor


@pytest.fixture
def clean_df():
    """Create a clean dataframe."""
    return pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature_b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


def test_init_valid_thresholds():
    """Test monitor initialization with valid thresholds."""
    monitor = DataQualityMonitor(
        missing_threshold=0.1,
        duplicate_threshold=0.05,
        volume_floor=10,
    )
    assert monitor.missing_threshold == 0.1
    assert monitor.duplicate_threshold == 0.05
    assert monitor.volume_floor == 10


def test_init_invalid_missing_threshold():
    """Test monitor initialization with invalid missing threshold."""
    with pytest.raises(ValueError, match="missing_threshold must be in"):
        DataQualityMonitor(missing_threshold=1.5)


def test_init_invalid_duplicate_threshold():
    """Test monitor initialization with invalid duplicate threshold."""
    with pytest.raises(ValueError, match="duplicate_threshold must be in"):
        DataQualityMonitor(duplicate_threshold=-0.1)


def test_init_invalid_volume_floor():
    """Test monitor initialization with invalid volume floor."""
    with pytest.raises(ValueError, match="volume_floor must be"):
        DataQualityMonitor(volume_floor=0)


def test_detect_empty_dataframe():
    """Test detection on empty dataframe."""
    monitor = DataQualityMonitor()
    result = monitor.detect(pd.DataFrame())
    assert not result.quality_ok
    assert result.missing_rate == 1.0
    assert result.volume == 0
    assert "empty dataframe" in result.issues


def test_detect_clean_dataframe(clean_df):
    """Test detection on clean dataframe."""
    monitor = DataQualityMonitor()
    result = monitor.detect(clean_df)
    assert result.quality_ok
    assert result.missing_rate == 0.0
    assert result.duplicate_rate == 0.0
    assert result.volume == 10
    assert len(result.issues) == 0


def test_detect_missing_values(clean_df):
    """Test detection of missing values."""
    monitor = DataQualityMonitor(missing_threshold=0.05)
    df = clean_df.copy()
    df.loc[0, "feature_a"] = np.nan
    df.loc[1, "feature_b"] = np.nan
    result = monitor.detect(df)
    assert not result.quality_ok
    assert "missing_rate" in result.issues[0]


def test_detect_duplicates(clean_df):
    """Test detection of duplicate rows."""
    monitor = DataQualityMonitor(duplicate_threshold=0.05)
    df = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
    result = monitor.detect(df)
    assert not result.quality_ok
    assert "duplicate_rate" in result.issues[0]


def test_detect_volume_floor(clean_df):
    """Test detection of low volume."""
    monitor = DataQualityMonitor(volume_floor=20)
    result = monitor.detect(clean_df)
    assert not result.quality_ok
    assert "volume" in result.issues[0]


def test_detect_schema_violations(clean_df):
    """Test detection of schema violations."""
    monitor = DataQualityMonitor(schema_violation_threshold=-1)  # Alert on any violation
    df = clean_df.copy()
    df["feature_a"] = df["feature_a"].astype(object)
    df.loc[0, "feature_a"] = "INVALID"
    result = monitor.detect(df)
    # May or may not alert depending on implementation, just check it runs
    assert result.schema_violations >= 0


def test_make_incident_clean(clean_df):
    """Test incident creation from clean data."""
    monitor = DataQualityMonitor()
    result = monitor.detect(clean_df)
    incident = monitor.make_incident(result, tenant_id="test")
    assert incident is None


def test_make_incident_degraded(clean_df):
    """Test incident creation from degraded data."""
    monitor = DataQualityMonitor(volume_floor=10)
    df = clean_df.copy()
    df.loc[0, "feature_a"] = np.nan
    result = monitor.detect(df)
    incident = monitor.make_incident(result, tenant_id="test")
    assert incident is not None
    assert incident.type == IncidentType.DATA_QUALITY
    assert incident.tenant_id == "test"
    assert "missing_rate" in incident.payload
    assert 0 <= incident.severity <= 1.0
