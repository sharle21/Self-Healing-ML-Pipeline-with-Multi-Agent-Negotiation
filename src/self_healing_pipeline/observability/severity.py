"""Severity calculation: derive incident severity from telemetry, not hardcoded."""

from __future__ import annotations

from dataclasses import dataclass

from self_healing_pipeline.gateway.events import IncidentType
from self_healing_pipeline.observability.telemetry import Telemetry


@dataclass(slots=True)
class SeverityBreakdown:
    """Component breakdown of severity calculation."""

    impact: float
    deviation: float
    urgency: float
    business_risk: float
    severity: float


class SeverityCalculator:
    """Calculate incident severity from telemetry using incident-specific formulas."""

    def calculate(
        self,
        incident_type: IncidentType,
        telemetry: Telemetry,
        baseline: Telemetry | None = None,
        tenant_config: dict | None = None,
    ) -> tuple[float, SeverityBreakdown]:
        """Calculate severity for an incident type.

        Args:
            incident_type: type of incident detected
            telemetry: current system telemetry
            baseline: baseline/healthy telemetry for comparison
            tenant_config: tenant-specific thresholds (min_auc, max_latency_ms, max_missing_rate, etc.)

        Returns:
            (severity: 0-1, breakdown: component breakdown)
        """
        if incident_type == IncidentType.DRIFT:
            return self._drift_severity(telemetry, baseline, tenant_config)
        elif incident_type == IncidentType.DATA_QUALITY:
            return self._data_quality_severity(telemetry, baseline, tenant_config)
        elif incident_type == IncidentType.COST_THRESHOLD:
            return self._cost_severity(telemetry, baseline, tenant_config)
        elif incident_type == IncidentType.LATENCY_BREACH:
            return self._latency_severity(telemetry, baseline, tenant_config)
        else:
            return 0.5, SeverityBreakdown(0.5, 0.5, 0.5, 0.5, 0.5)

    def _drift_severity(
        self, telemetry: Telemetry, baseline: Telemetry | None, tenant_config: dict | None = None
    ) -> tuple[float, SeverityBreakdown]:
        """Severity for DRIFT incidents.

        Components:
        - Impact: AUC drop (normalized to baseline_auc - min_auc = 1.0)
        - Deviation: Max feature drift in sigma units (normalized to 3 sigma = 1.0)
        - Urgency: Predictions per second affected
        - Business risk: False negative cost
        """
        # Extract tenant-specific thresholds
        baseline_auc = tenant_config.get("baseline_auc", 0.80) if tenant_config else 0.80
        min_auc = tenant_config.get("min_auc", baseline_auc - 0.03) if tenant_config else baseline_auc - 0.03
        auc_threshold = baseline_auc - min_auc  # How much drop triggers catastrophic

        # Impact: AUC degradation
        auc_drop = 0.05  # Default: assume 5% drop
        if baseline:
            auc_drop = max(0, baseline.model.auc - telemetry.model.auc)
        impact = min(auc_drop / max(auc_threshold, 0.01), 1.0)  # Tenant's threshold = catastrophic

        # Deviation: Feature drift magnitude
        max_drift = max(telemetry.data.feature_drift_scores.values()) if telemetry.data.feature_drift_scores else 0.0
        deviation = min(max_drift / 3.0, 1.0)  # 3 sigma = catastrophic

        # Urgency: Assume 50k predictions/day affected if drift detected
        predictions_affected = 50000 if max_drift > 0 else 0
        urgency = min(predictions_affected / 10000, 1.0)  # 10k affected = high urgency

        # Business risk: False negative cost for default prediction
        # Assume $500 false negative cost
        business_risk = 0.8

        severity = 0.35 * impact + 0.30 * deviation + 0.20 * urgency + 0.15 * business_risk

        return severity, SeverityBreakdown(impact, deviation, urgency, business_risk, severity)

    def _data_quality_severity(
        self, telemetry: Telemetry, baseline: Telemetry | None, tenant_config: dict | None = None
    ) -> tuple[float, SeverityBreakdown]:
        """Severity for DATA_QUALITY incidents.

        Components:
        - Impact: Missing rate (normalized to tenant's max_missing_rate = 1.0)
        - Deviation: Schema violations (normalized to 100 = 1.0)
        - Urgency: Volume affected
        - Business risk: Data corruption impact
        """
        # Extract tenant-specific threshold
        max_missing_rate = tenant_config.get("max_missing_rate", 0.05) if tenant_config else 0.05

        # Impact: Missing data
        impact = min(telemetry.data.missing_rate / max(max_missing_rate, 0.01), 1.0)  # Tenant's threshold = catastrophic

        # Deviation: Schema violations
        schema_norm = min(telemetry.data.schema_violations / 100, 1.0)

        # Urgency: Duplicate rate (data corruption indicator)
        urgency = telemetry.data.duplicate_rate * 2  # Double weight on duplicates
        urgency = min(urgency, 1.0)

        # Business risk: Data quality issues affect all downstream
        business_risk = 0.75

        severity = 0.5 * impact + 0.3 * schema_norm + 0.2 * urgency - 0.1 * business_risk

        return severity, SeverityBreakdown(impact, schema_norm, urgency, business_risk, severity)

    def _cost_severity(
        self, telemetry: Telemetry, baseline: Telemetry | None, tenant_config: dict | None = None
    ) -> tuple[float, SeverityBreakdown]:
        """Severity for COST_THRESHOLD incidents.

        Components:
        - Impact: Cost per prediction spike
        - Deviation: Latency increase (slower = more compute)
        - Urgency: Daily cost projection
        - Business risk: Budget overrun (tenant's daily_cost_budget)
        """
        # Extract tenant thresholds
        daily_cost_budget = tenant_config.get("daily_cost_budget", 100.0) if tenant_config else 100.0

        # Impact: Cost increase
        baseline_cost = baseline.system.cost_per_prediction if baseline else 0.001
        cost_increase = max(0, telemetry.system.cost_per_prediction - baseline_cost)
        impact = min(cost_increase / 0.005, 1.0)  # $0.005 increase = catastrophic

        # Deviation: Latency increase (proxy for compute)
        baseline_latency = baseline.system.latency_p95 if baseline else 80
        latency_increase = max(0, telemetry.system.latency_p95 - baseline_latency)
        deviation = min(latency_increase / 100, 1.0)  # 100ms increase = high

        # Urgency: Daily cost at scale
        daily_predictions = 1_000_000
        daily_cost = telemetry.system.cost_per_prediction * daily_predictions
        urgency = min(daily_cost / 10000, 1.0)  # $10k/day = high

        # Business risk: Budget constraint
        business_risk = 0.6

        severity = 0.35 * impact + 0.30 * deviation + 0.20 * urgency + 0.15 * business_risk

        return severity, SeverityBreakdown(impact, deviation, urgency, business_risk, severity)

    def _latency_severity(
        self, telemetry: Telemetry, baseline: Telemetry | None, tenant_config: dict | None = None
    ) -> tuple[float, SeverityBreakdown]:
        """Severity for LATENCY_BREACH incidents.

        Components:
        - Impact: P95 latency degradation vs tenant SLA
        - Deviation: P99 latency (tail latency)
        - Urgency: User experience impact
        - Business risk: SLA breach cost
        """
        # Extract tenant SLA
        latency_sla_ms = tenant_config.get("latency_sla_ms", 100.0) if tenant_config else 100.0
        max_latency = tenant_config.get("max_latency_ms", latency_sla_ms * 1.2) if tenant_config else latency_sla_ms * 1.2
        baseline_latency = tenant_config.get("baseline_latency_ms", 85.0) if tenant_config else 85.0

        # Impact: P95 increase from baseline
        baseline_p95 = baseline.system.latency_p95 if baseline else baseline_latency
        p95_increase = max(0, telemetry.system.latency_p95 - baseline_p95)
        impact = min(p95_increase / max(max_latency - baseline_latency, 50), 1.0)  # Tenant's threshold = catastrophic

        # Deviation: P99 (tail latency is critical)
        p99_increase = max(0, telemetry.system.latency_p99 - 150)  # 150ms baseline
        deviation = min(p99_increase / 300, 1.0)

        # Urgency: SLA breach based on tenant SLA
        sla_breach = 1.0 if telemetry.system.latency_p95 > latency_sla_ms else 0.0
        urgency = sla_breach

        # Business risk: SLA penalties
        business_risk = 0.7

        severity = 0.35 * impact + 0.30 * deviation + 0.20 * urgency + 0.15 * business_risk

        return severity, SeverityBreakdown(impact, deviation, urgency, business_risk, severity)
