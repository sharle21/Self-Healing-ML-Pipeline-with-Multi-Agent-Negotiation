"""Telemetry collector: gather system metrics from Prometheus-like sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ModelMetrics:
    """Current model performance metrics."""

    auc: float
    precision: float
    recall: float
    calibration_error: float
    error_rate: float


@dataclass(slots=True)
class DataMetrics:
    """Current data quality metrics."""

    missing_rate: float
    duplicate_rate: float
    schema_violations: int
    feature_drift_scores: dict[str, float]  # feature -> drift magnitude


@dataclass(slots=True)
class SystemMetrics:
    """System performance metrics."""

    latency_p95: float  # milliseconds
    latency_p99: float
    cpu_usage: float  # percentage
    memory_usage: float  # percentage
    cost_per_prediction: float  # dollars


@dataclass(slots=True)
class Telemetry:
    """Complete system telemetry snapshot."""

    model: ModelMetrics
    data: DataMetrics
    system: SystemMetrics
    timestamp: str


class TelemetryCollector:
    """Collect metrics from Prometheus or mock sources."""

    def __init__(self, use_mock: bool = True) -> None:
        """Initialize telemetry collector.

        Args:
            use_mock: if True, return mock metrics for testing
        """
        self.use_mock = use_mock
        self.baseline_model: ModelMetrics | None = None

    def set_baseline(self, baseline: ModelMetrics) -> None:
        """Set baseline (healthy) model metrics for comparison.

        Args:
            baseline: reference metrics to compare against
        """
        self.baseline_model = baseline

    def collect(self) -> Telemetry:
        """Collect current system telemetry from Prometheus or mock.

        Returns:
            Telemetry snapshot
        """
        if self.use_mock:
            return self._mock_telemetry()
        return self._prometheus_telemetry()

    def _prometheus_telemetry(self) -> Telemetry:
        """Query metrics from Prometheus server.

        Returns:
            Telemetry snapshot from Prometheus queries
        """
        import os

        from datetime import UTC, datetime

        prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

        try:
            import requests

            # Query model metrics
            model_metrics = self._query_prometheus(
                prometheus_url,
                {
                    "auc": "model_auc",
                    "precision": "model_precision",
                    "recall": "model_recall",
                    "calibration_error": "model_calibration_error",
                    "error_rate": "model_error_rate",
                },
            )

            # Query data metrics
            data_metrics = self._query_prometheus(
                prometheus_url,
                {
                    "missing_rate": "data_missing_rate",
                    "duplicate_rate": "data_duplicate_rate",
                    "schema_violations": "data_schema_violations",
                },
            )

            # Query system metrics
            system_metrics = self._query_prometheus(
                prometheus_url,
                {
                    "latency_p95": "system_latency_p95_ms",
                    "latency_p99": "system_latency_p99_ms",
                    "cpu_usage": "system_cpu_percent",
                    "memory_usage": "system_memory_percent",
                    "cost_per_prediction": "system_cost_per_prediction",
                },
            )

            # Query feature drift scores (dynamic dict)
            feature_drift = self._query_prometheus_dict(prometheus_url, "feature_drift_score")

            return Telemetry(
                model=ModelMetrics(
                    auc=model_metrics.get("auc", 0.75),
                    precision=model_metrics.get("precision", 0.82),
                    recall=model_metrics.get("recall", 0.71),
                    calibration_error=model_metrics.get("calibration_error", 0.08),
                    error_rate=model_metrics.get("error_rate", 0.12),
                ),
                data=DataMetrics(
                    missing_rate=data_metrics.get("missing_rate", 0.08),
                    duplicate_rate=data_metrics.get("duplicate_rate", 0.02),
                    schema_violations=int(data_metrics.get("schema_violations", 0)),
                    feature_drift_scores=feature_drift,
                ),
                system=SystemMetrics(
                    latency_p95=system_metrics.get("latency_p95", 85),
                    latency_p99=system_metrics.get("latency_p99", 150),
                    cpu_usage=system_metrics.get("cpu_usage", 45),
                    memory_usage=system_metrics.get("memory_usage", 60),
                    cost_per_prediction=system_metrics.get("cost_per_prediction", 0.005),
                ),
                timestamp=datetime.now(UTC).isoformat(),
            )
        except Exception:
            # Fallback to mock if Prometheus unavailable
            return self._mock_telemetry()

    def _query_prometheus(self, prometheus_url: str, queries: dict[str, str]) -> dict[str, float]:
        """Query Prometheus for multiple metrics.

        Args:
            prometheus_url: Prometheus server URL
            queries: dict of {result_key: prometheus_metric_name}

        Returns:
            dict of {result_key: value}
        """
        import requests

        results = {}
        for key, metric in queries.items():
            try:
                response = requests.get(
                    f"{prometheus_url}/api/v1/query",
                    params={"query": metric},
                    timeout=5,
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("data", {}).get("result"):
                        value = float(data["data"]["result"][0]["value"][1])
                        results[key] = value
            except Exception:
                pass
        return results

    def _query_prometheus_dict(self, prometheus_url: str, metric_prefix: str) -> dict[str, float]:
        """Query Prometheus for metrics with labels (e.g., feature_drift_score{feature="x"}).

        Args:
            prometheus_url: Prometheus server URL
            metric_prefix: metric name prefix

        Returns:
            dict of {label_value: metric_value}
        """
        import requests

        results = {}
        try:
            response = requests.get(
                f"{prometheus_url}/api/v1/query",
                params={"query": metric_prefix},
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                for result in data.get("data", {}).get("result", []):
                    labels = result.get("metric", {})
                    value = float(result.get("value", [0, 0])[1])
                    # Use feature label if available
                    key = labels.get("feature", labels.get("name", "unknown"))
                    results[key] = value
        except Exception:
            pass
        return results

    def _mock_telemetry(self) -> Telemetry:
        """Return mock telemetry for testing."""
        from datetime import UTC, datetime

        return Telemetry(
            model=ModelMetrics(
                auc=0.75,
                precision=0.82,
                recall=0.71,
                calibration_error=0.08,
                error_rate=0.12,
            ),
            data=DataMetrics(
                missing_rate=0.08,
                duplicate_rate=0.02,
                schema_violations=0,
                feature_drift_scores={"income": 1.2, "age": 0.5},
            ),
            system=SystemMetrics(
                latency_p95=85,
                latency_p99=150,
                cpu_usage=45,
                memory_usage=62,
                cost_per_prediction=0.002,
            ),
            timestamp=datetime.now(UTC).isoformat(),
        )
