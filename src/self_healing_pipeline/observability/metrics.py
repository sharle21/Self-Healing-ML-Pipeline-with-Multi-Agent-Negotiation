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
