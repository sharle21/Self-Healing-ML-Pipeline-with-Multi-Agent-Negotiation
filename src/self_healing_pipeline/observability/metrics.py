"""Prometheus metrics for the self-healing pipeline."""

from prometheus_client import Counter, Gauge, Histogram

# Prediction metrics
prediction_count = Counter(
    "predictions_total",
    "Total predictions made",
    labelnames=["tenant_id"],
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    labelnames=["tenant_id"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# Incident metrics
incident_count = Counter(
    "incidents_total",
    "Total incidents detected",
    labelnames=["tenant_id", "type"],
)

# Business metrics
cost_per_prediction = Gauge(
    "cost_per_prediction",
    "Cost per prediction (rolling average)",
    labelnames=["tenant_id"],
)

false_positives_total = Counter(
    "false_positives_total",
    "Total false positive predictions",
    labelnames=["tenant_id"],
)

false_negatives_total = Counter(
    "false_negatives_total",
    "Total false negative predictions",
    labelnames=["tenant_id"],
)

# FP/FN rates (rolling window, updated by replay script)
false_positive_rate = Gauge("false_positive_rate", "FP rate over rolling window", labelnames=["tenant_id"])
false_negative_rate = Gauge("false_negative_rate", "FN rate over rolling window", labelnames=["tenant_id"])

# Model quality metrics (updated by replay script via /internal/metrics/update)
model_auc = Gauge("model_auc", "Rolling AUC over last N predictions", labelnames=["tenant_id"])
model_precision = Gauge("model_precision", "Rolling precision", labelnames=["tenant_id"])
model_recall = Gauge("model_recall", "Rolling recall", labelnames=["tenant_id"])
model_error_rate = Gauge("model_error_rate", "Fraction of failed/incorrect predictions", labelnames=["tenant_id"])
model_calibration_error = Gauge("model_calibration_error", "Expected calibration error", labelnames=["tenant_id"])

# Data quality metrics (updated by replay script)
data_missing_rate = Gauge("data_missing_rate", "Fraction of requests with missing features", labelnames=["tenant_id"])
data_duplicate_rate = Gauge("data_duplicate_rate", "Fraction of duplicate rows in window", labelnames=["tenant_id"])
data_schema_violations = Gauge("data_schema_violations_total", "Schema violations in window", labelnames=["tenant_id"])
feature_drift_score = Gauge("feature_drift_score", "KS drift score vs training distribution", labelnames=["tenant_id", "feature"])
model_drift_percentage = Gauge("model_drift_percentage", "Fraction of features with drift > 1σ (DriftMonitor)", labelnames=["tenant_id"])

# System metrics (from histogram summaries)
system_latency_p95 = Gauge("system_latency_p95_ms", "p95 latency in ms (rolling)", labelnames=["tenant_id"])
system_latency_p99 = Gauge("system_latency_p99_ms", "p99 latency in ms (rolling)", labelnames=["tenant_id"])

# Agent metrics
agent_win_count = Counter(
    "agent_wins_total",
    "Total wins by agent (commander selection)",
    labelnames=["agent_type"],
)

agent_proposal_count = Counter(
    "agent_proposals_total",
    "Total proposals made by agent",
    labelnames=["agent_type"],
)
