import pytest

from self_healing_pipeline.gateway.events import IncidentType
from self_healing_pipeline.monitors.business import BusinessCostMonitor


@pytest.fixture
def monitor():
    """Create a cost monitor."""
    return BusinessCostMonitor(
        false_positive_cost=10.0,
        false_negative_cost=50.0,
        cost_threshold=5.0,
        window_size=100,
    )


def test_init_valid_params():
    """Test monitor initialization with valid parameters."""
    monitor = BusinessCostMonitor(
        false_positive_cost=10.0,
        false_negative_cost=50.0,
        cost_threshold=5.0,
        window_size=100,
    )
    assert monitor.false_positive_cost == 10.0
    assert monitor.false_negative_cost == 50.0
    assert monitor.cost_threshold == 5.0
    assert monitor.window_size == 100


def test_init_invalid_fp_cost():
    """Test monitor initialization with invalid FP cost."""
    with pytest.raises(ValueError, match="false_positive_cost"):
        BusinessCostMonitor(false_positive_cost=-1.0)


def test_init_invalid_fn_cost():
    """Test monitor initialization with invalid FN cost."""
    with pytest.raises(ValueError, match="false_negative_cost"):
        BusinessCostMonitor(false_negative_cost=-1.0)


def test_init_invalid_window_size():
    """Test monitor initialization with invalid window size."""
    with pytest.raises(ValueError, match="window_size"):
        BusinessCostMonitor(window_size=0)


def test_record_prediction_valid(monitor):
    """Test recording valid predictions."""
    monitor.record_prediction(0, 0)  # TN
    monitor.record_prediction(1, 1)  # TP
    assert len(monitor.outcomes) == 2


def test_record_prediction_invalid_y_true(monitor):
    """Test recording invalid y_true."""
    with pytest.raises(ValueError, match="y_true must be 0 or 1"):
        monitor.record_prediction(2, 0)


def test_record_prediction_invalid_y_pred(monitor):
    """Test recording invalid y_pred."""
    with pytest.raises(ValueError, match="y_pred must be 0 or 1"):
        monitor.record_prediction(1, 2)


def test_detect_empty(monitor):
    """Test detection on empty outcomes."""
    result = monitor.detect()
    assert result.cost_ok
    assert result.cost_per_prediction == 0.0
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.predictions_evaluated == 0


def test_detect_all_correct(monitor):
    """Test detection with all correct predictions."""
    monitor.record_prediction(0, 0)  # TN
    monitor.record_prediction(1, 1)  # TP
    monitor.record_prediction(0, 0)  # TN
    result = monitor.detect()
    assert result.cost_ok
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.cost_per_prediction == 0.0


def test_detect_false_positives(monitor):
    """Test detection of false positives."""
    monitor.record_prediction(0, 1)  # FP: cost 10
    monitor.record_prediction(0, 1)  # FP: cost 10
    result = monitor.detect()
    assert result.false_positives == 2
    assert result.false_negatives == 0
    assert result.total_cost == 20.0
    assert result.cost_per_prediction == 10.0


def test_detect_false_negatives(monitor):
    """Test detection of false negatives."""
    monitor.record_prediction(1, 0)  # FN: cost 50
    result = monitor.detect()
    assert result.false_positives == 0
    assert result.false_negatives == 1
    assert result.total_cost == 50.0
    assert result.cost_per_prediction == 50.0


def test_detect_mixed_errors(monitor):
    """Test detection of mixed FP/FN."""
    monitor.record_prediction(0, 1)  # FP: cost 10
    monitor.record_prediction(1, 0)  # FN: cost 50
    monitor.record_prediction(1, 1)  # TP: cost 0
    result = monitor.detect()
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.total_cost == 60.0
    assert result.cost_per_prediction == 20.0


def test_detect_exceeds_threshold(monitor):
    """Test detection when cost exceeds threshold."""
    monitor.record_prediction(1, 0)  # FN: cost 50
    result = monitor.detect()
    assert not result.cost_ok
    assert result.cost_per_prediction == 50.0
    assert result.cost_per_prediction > monitor.cost_threshold


def test_detect_rolling_window(monitor):
    """Test rolling window behavior."""
    # Fill window with good predictions
    for _ in range(100):
        monitor.record_prediction(1, 1)  # All correct
    result = monitor.detect()
    assert result.cost_ok
    assert result.cost_per_prediction == 0.0

    # Add bad predictions (old ones drop out)
    for _ in range(50):
        monitor.record_prediction(1, 0)  # FN: cost 50
    result = monitor.detect()
    assert not result.cost_ok
    assert result.predictions_evaluated == 100


def test_make_incident_ok(monitor):
    """Test incident creation when cost is ok."""
    monitor.record_prediction(1, 1)
    result = monitor.detect()
    incident = monitor.make_incident(result, tenant_id="test")
    assert incident is None


def test_make_incident_threshold_exceeded(monitor):
    """Test incident creation when cost exceeds threshold."""
    monitor.record_prediction(1, 0)  # FN: cost 50
    result = monitor.detect()
    incident = monitor.make_incident(result, tenant_id="test")
    assert incident is not None
    assert incident.type == IncidentType.COST_THRESHOLD
    assert incident.tenant_id == "test"
    assert "cost_per_prediction" in incident.payload
    assert incident.payload["false_negatives"] == 1
    assert incident.payload["false_positives"] == 0
    assert 0 <= incident.severity <= 1.0
