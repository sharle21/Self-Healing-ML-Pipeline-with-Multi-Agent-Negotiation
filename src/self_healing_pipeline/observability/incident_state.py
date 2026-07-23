"""Unified IncidentState: data-driven snapshot built from Prometheus + SQLite.

Replaces per-agent hardcoded fallbacks in StateConstructor.
Agents receive real values instead of assumed constants.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# σ threshold above which a feature is counted as "drifted"
_DRIFT_THRESHOLD = 1.5

# Assumed daily prediction volume (for cost_budget_per_1000 calculation)
_ASSUMED_DAILY_PREDICTIONS = 50_000.0


@dataclass(slots=True)
class IncidentState:
    """Unified state snapshot for a single tenant incident.

    Built from Prometheus (current live metrics) + SQLite (baselines, policy, history).
    All agents consume this object; per-agent state dicts are derived from it.
    """

    tenant_id: str
    incident_type: str  # IncidentType.value

    # --- Model quality: current (Prometheus) vs baseline (ModelValidationReport) ---
    current_auc: float | None              # model_auc{tenant_id=...} — None if not yet published
    baseline_auc: float                    # ModelValidationReport.auc (latest)
    current_precision: float | None        # model_precision{tenant_id=...}
    baseline_precision: float              # ModelValidationReport.precision
    current_recall: float | None           # model_recall{tenant_id=...}
    baseline_recall: float                 # ModelValidationReport.recall
    auc_drop: float | None                 # baseline_auc - current_auc (positive = degraded)
    false_positive_rate: float             # false_positive_rate{tenant_id=...}
    false_negative_rate: float             # false_negative_rate{tenant_id=...}

    # --- Drift (from feature_drift_score{tenant_id=...}) ---
    max_feature_drift: float               # max drift score across all features
    drifted_features: list[str]            # features with drift > _DRIFT_THRESHOLD

    # --- Data quality ---
    missing_rate: float                    # data_missing_rate{tenant_id=...}
    duplicate_rate: float                  # data_duplicate_rate{tenant_id=...}
    schema_violation_rate: float           # data_schema_violations_total{...} / window_size

    # --- Latency ---
    latency_p95_ms: float                  # system_latency_p95_ms{tenant_id=...}
    latency_p99_ms: float                  # system_latency_p99_ms{tenant_id=...}
    latency_sla_ms: float                  # TenantPolicy.latency_sla_ms

    # --- Cost ---
    cost_per_1000_predictions: float       # cost_per_prediction{...} * 1000
    cost_budget_per_1000: float            # daily_cost_budget / (daily_predictions / 1000)

    # --- Model identity (ModelValidationReport) ---
    current_threshold: float               # optimal_threshold from latest report
    current_model_version: str             # model_version from latest report
    previous_model_version: str | None     # model_version from second-latest report
    last_training_age_days: float          # days since latest report.validated_at

    # --- Policy limits (TenantPolicy) ---
    min_auc: float                         # min_acceptable_auc
    max_latency_ms: float                  # max_acceptable_latency_ms
    max_missing_rate: float                # max_acceptable_missing_rate

    # --- Historical agent performance (RemediationAction table) ---
    historical_agent_success: dict[str, float]  # agent_type -> success_rate [0, 1]

    # --- Computed severity ---
    severity: float
    severity_components: dict[str, float]  # breakdown keys: impact, deviation, urgency, business_risk


class IncidentStateBuilder:
    """Build IncidentState from live Prometheus metrics + SQLite baselines.

    Falls back gracefully: missing DB → use validation defaults; Prometheus
    unavailable → TelemetryCollector returns mock values.
    """

    def __init__(
        self,
        collector: Any,  # TelemetryCollector
        session_factory: Any | None = None,
    ) -> None:
        self._collector = collector
        self._session_factory = session_factory

    def build(self, tenant_id: str, incident_type: str) -> IncidentState:
        """Build IncidentState for a tenant + incident type.

        Args:
            tenant_id: tenant identifier
            incident_type: IncidentType.value string ("drift", "data_quality", etc.)

        Returns:
            IncidentState populated with real values where available.
        """
        telemetry = self._collector.collect()

        # --- Defaults (overridden from DB when available) ---
        baseline_auc = 0.80
        baseline_precision = 0.82
        baseline_recall = 0.71
        min_auc = 0.75
        max_latency_ms = 120.0
        max_missing_rate = 0.05
        latency_sla_ms = 100.0
        daily_cost_budget = 100.0
        current_threshold = 0.50
        current_model_version = "v1"
        previous_model_version: str | None = None
        last_training_age_days = 30.0
        historical_agent_success: dict[str, float] = {}

        if self._session_factory is not None:
            historical_agent_success, *rest = self._load_from_db(
                tenant_id,
                baseline_auc,
                baseline_precision,
                baseline_recall,
                min_auc,
                max_latency_ms,
                max_missing_rate,
                latency_sla_ms,
                daily_cost_budget,
                current_threshold,
                current_model_version,
                previous_model_version,
                last_training_age_days,
            )
            (
                baseline_auc,
                baseline_precision,
                baseline_recall,
                min_auc,
                max_latency_ms,
                max_missing_rate,
                latency_sla_ms,
                daily_cost_budget,
                current_threshold,
                current_model_version,
                previous_model_version,
                last_training_age_days,
            ) = rest

        # --- Current metrics from Prometheus (or mock) ---
        current_auc = telemetry.model.auc if telemetry.model.auc > 0 else None
        current_precision = telemetry.model.precision if telemetry.model.precision > 0 else None
        current_recall = telemetry.model.recall if telemetry.model.recall > 0 else None
        auc_drop = (baseline_auc - current_auc) if current_auc is not None else None

        # FP/FN rate: Prometheus has dedicated gauges; approximate from error_rate if not set
        fpr = getattr(telemetry.model, "false_positive_rate", None)
        fnr = getattr(telemetry.model, "false_negative_rate", None)
        false_positive_rate = fpr if fpr is not None else telemetry.model.error_rate * 0.40
        false_negative_rate = fnr if fnr is not None else telemetry.model.error_rate * 0.60

        drift_scores = telemetry.data.feature_drift_scores or {}
        max_drift = max(drift_scores.values()) if drift_scores else 0.0
        drifted = [f for f, s in drift_scores.items() if s > _DRIFT_THRESHOLD]

        # schema_violations is a count; normalize to rate per assumed 500-row window
        schema_violation_rate = telemetry.data.schema_violations / 500.0

        cost_per_1000 = telemetry.system.cost_per_prediction * 1000.0
        cost_budget_per_1000 = (
            daily_cost_budget / (_ASSUMED_DAILY_PREDICTIONS / 1000.0)
        )

        # --- Severity (reuse existing SeverityCalculator) ---
        severity, breakdown = self._compute_severity(
            incident_type, telemetry,
            baseline_auc=baseline_auc,
            min_auc=min_auc,
            max_latency_ms=max_latency_ms,
            max_missing_rate=max_missing_rate,
            latency_sla_ms=latency_sla_ms,
            daily_cost_budget=daily_cost_budget,
        )

        return IncidentState(
            tenant_id=tenant_id,
            incident_type=incident_type,
            current_auc=current_auc,
            baseline_auc=baseline_auc,
            current_precision=current_precision,
            baseline_precision=baseline_precision,
            current_recall=current_recall,
            baseline_recall=baseline_recall,
            auc_drop=auc_drop,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            max_feature_drift=max_drift,
            drifted_features=drifted,
            missing_rate=telemetry.data.missing_rate,
            duplicate_rate=telemetry.data.duplicate_rate,
            schema_violation_rate=schema_violation_rate,
            latency_p95_ms=telemetry.system.latency_p95,
            latency_p99_ms=telemetry.system.latency_p99,
            latency_sla_ms=latency_sla_ms,
            cost_per_1000_predictions=cost_per_1000,
            cost_budget_per_1000=cost_budget_per_1000,
            current_threshold=current_threshold,
            current_model_version=current_model_version,
            previous_model_version=previous_model_version,
            last_training_age_days=last_training_age_days,
            min_auc=min_auc,
            max_latency_ms=max_latency_ms,
            max_missing_rate=max_missing_rate,
            historical_agent_success=historical_agent_success,
            severity=severity,
            severity_components=dict(breakdown.components),
        )

    def _load_from_db(
        self,
        tenant_id: str,
        *defaults: Any,
    ) -> tuple[Any, ...]:
        """Load tenant-specific values from DB, falling back to provided defaults.

        Returns tuple: (historical_agent_success, baseline_auc, baseline_precision,
                        baseline_recall, min_auc, max_latency_ms, max_missing_rate,
                        latency_sla_ms, daily_cost_budget, current_threshold,
                        current_model_version, previous_model_version, last_training_age_days)
        """
        (
            baseline_auc, baseline_precision, baseline_recall,
            min_auc, max_latency_ms, max_missing_rate, latency_sla_ms,
            daily_cost_budget, current_threshold, current_model_version,
            previous_model_version, last_training_age_days,
        ) = defaults

        historical_agent_success: dict[str, float] = {}

        try:
            from self_healing_pipeline.db.models import (
                ModelValidationReport,
                RemediationAction,
                TenantPolicy,
            )

            with self._session_factory() as session:
                # Tenant policy → alert thresholds + cost budget
                policy = session.query(TenantPolicy).filter_by(tenant_id=tenant_id).first()
                if policy:
                    min_auc = policy.min_acceptable_auc
                    max_latency_ms = policy.max_acceptable_latency_ms
                    max_missing_rate = policy.max_acceptable_missing_rate
                    latency_sla_ms = policy.latency_sla_ms
                    daily_cost_budget = policy.daily_cost_budget

                # Model validation reports → baselines + model identity
                reports = (
                    session.query(ModelValidationReport)
                    .filter_by(tenant_id=tenant_id)
                    .order_by(ModelValidationReport.validated_at.desc())
                    .limit(2)
                    .all()
                )
                if reports:
                    cur = reports[0]
                    baseline_auc = cur.auc
                    baseline_precision = cur.precision
                    baseline_recall = cur.recall
                    current_threshold = cur.optimal_threshold
                    current_model_version = cur.model_version
                    validated_at = cur.validated_at
                    if validated_at.tzinfo is None:
                        validated_at = validated_at.replace(tzinfo=UTC)
                    last_training_age_days = (
                        datetime.now(UTC) - validated_at
                    ).total_seconds() / 86400.0
                    if len(reports) > 1:
                        previous_model_version = reports[1].model_version

                # Historical agent success from RemediationAction
                actions = session.query(RemediationAction).all()
                counts: dict[str, list[bool]] = {}
                for action in actions:
                    counts.setdefault(action.agent, []).append(action.success)
                historical_agent_success = {
                    agent: sum(outcomes) / len(outcomes)
                    for agent, outcomes in counts.items()
                    if outcomes
                }

        except Exception as exc:
            logger.debug("DB load failed, using defaults: %s", exc)

        return (
            historical_agent_success,
            baseline_auc, baseline_precision, baseline_recall,
            min_auc, max_latency_ms, max_missing_rate, latency_sla_ms,
            daily_cost_budget, current_threshold, current_model_version,
            previous_model_version, last_training_age_days,
        )

    @staticmethod
    def _compute_severity(
        incident_type: str,
        telemetry: Any,
        *,
        baseline_auc: float,
        min_auc: float,
        max_latency_ms: float,
        max_missing_rate: float,
        latency_sla_ms: float,
        daily_cost_budget: float,
    ) -> tuple[float, Any]:
        from self_healing_pipeline.gateway.events import IncidentType
        from self_healing_pipeline.observability.severity import SeverityCalculator

        try:
            inc_type = IncidentType(incident_type)
        except ValueError:
            inc_type = IncidentType.DRIFT

        tenant_cfg = {
            "baseline_auc": baseline_auc,
            "min_auc": min_auc,
            "max_latency_ms": max_latency_ms,
            "max_missing_rate": max_missing_rate,
            "latency_sla_ms": latency_sla_ms,
            "daily_cost_budget": daily_cost_budget,
            "baseline_latency_ms": telemetry.system.latency_p95,
        }
        return SeverityCalculator().calculate(inc_type, telemetry, tenant_config=tenant_cfg)
