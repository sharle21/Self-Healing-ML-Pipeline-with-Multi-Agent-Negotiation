"""Severity calculation: per-incident-type formulas with named component breakdown."""

from __future__ import annotations

from dataclasses import dataclass, field

from self_healing_pipeline.gateway.events import IncidentType
from self_healing_pipeline.observability.telemetry import Telemetry

_ASSUMED_PREDICTIONS_PER_DAY = 50_000.0
_ASSUMED_HIGH_VOLUME = 100_000.0   # predictions/day that saturate the urgency signal
_ASSUMED_HIGH_RPS = 100.0           # requests/sec that saturate traffic_volume


@dataclass
class SeverityBreakdown:
    """Per-incident-type named severity components + rolled-up severity.

    Phase 9: components dict stores type-specific names so callers can surface
    them in dashboards and logs for explainability.

    Backward-compat: .impact/.deviation/.urgency/.business_risk properties
    return the first four component values (by insertion order) so old code
    continues to work unchanged.
    """

    components: dict[str, float]
    severity: float

    # Backward-compat shims (old code accesses breakdown.impact etc.)
    @property
    def impact(self) -> float:
        vals = list(self.components.values())
        return vals[0] if vals else 0.0

    @property
    def deviation(self) -> float:
        vals = list(self.components.values())
        return vals[1] if len(vals) > 1 else 0.0

    @property
    def urgency(self) -> float:
        vals = list(self.components.values())
        return vals[2] if len(vals) > 2 else 0.0

    @property
    def business_risk(self) -> float:
        vals = list(self.components.values())
        return vals[3] if len(vals) > 3 else 0.0


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


class SeverityCalculator:
    """Calculate incident severity using per-type formulas and named components."""

    def calculate(
        self,
        incident_type: IncidentType,
        telemetry: Telemetry,
        baseline: Telemetry | None = None,
        tenant_config: dict | None = None,
    ) -> tuple[float, SeverityBreakdown]:
        """Return (severity ∈ [0,1], breakdown with named components)."""
        if incident_type == IncidentType.DRIFT:
            return self._drift_severity(telemetry, baseline, tenant_config)
        elif incident_type == IncidentType.DATA_QUALITY:
            return self._data_quality_severity(telemetry, baseline, tenant_config)
        elif incident_type == IncidentType.COST_THRESHOLD:
            return self._cost_severity(telemetry, baseline, tenant_config)
        elif incident_type == IncidentType.LATENCY_BREACH:
            return self._latency_severity(telemetry, baseline, tenant_config)
        return 0.5, SeverityBreakdown(components={"unknown": 0.5}, severity=0.5)

    # ------------------------------------------------------------------
    # Phase 9: per-type formulas
    # ------------------------------------------------------------------

    def _drift_severity(
        self,
        telemetry: Telemetry,
        baseline: Telemetry | None,
        tenant_config: dict | None,
    ) -> tuple[float, SeverityBreakdown]:
        """severity = 0.45*auc_drop + 0.35*drift + 0.20*affected_volume"""
        baseline_auc = (tenant_config or {}).get("baseline_auc", 0.80)
        min_auc = (tenant_config or {}).get("min_auc", baseline_auc - 0.03)
        auc_threshold = max(baseline_auc - min_auc, 0.01)

        # auc_drop: how far into the danger zone
        auc_drop_abs = 0.0
        if baseline is not None:
            auc_drop_abs = max(0.0, baseline.model.auc - telemetry.model.auc)
        auc_drop_norm = _clip01(auc_drop_abs / auc_threshold)

        # drift: max feature drift in σ (3σ = catastrophic)
        max_drift = max(telemetry.data.feature_drift_scores.values(), default=0.0)
        drift_norm = _clip01(max_drift / 3.0)

        # affected_volume: predictions at risk
        affected = _ASSUMED_PREDICTIONS_PER_DAY if max_drift > 0 else 0.0
        volume_norm = _clip01(affected / _ASSUMED_HIGH_VOLUME)

        severity = 0.45 * auc_drop_norm + 0.35 * drift_norm + 0.20 * volume_norm
        return severity, SeverityBreakdown(
            components={"auc_drop": auc_drop_norm, "drift": drift_norm, "affected_volume": volume_norm},
            severity=severity,
        )

    def _data_quality_severity(
        self,
        telemetry: Telemetry,
        baseline: Telemetry | None,
        tenant_config: dict | None,
    ) -> tuple[float, SeverityBreakdown]:
        """severity = 0.40*missing_rate + 0.25*schema_rate + 0.15*duplicate_rate + 0.20*affected_volume"""
        max_missing = (tenant_config or {}).get("max_missing_rate", 0.05)

        missing_norm = _clip01(telemetry.data.missing_rate / max(max_missing, 0.01))
        # schema violations: 10% of rows with violations = catastrophic
        total_rows = max(telemetry.data.schema_violations / 0.10, 1.0) if telemetry.data.schema_violations > 0 else 1.0
        schema_rate = telemetry.data.schema_violations / total_rows
        schema_norm = _clip01(schema_rate / 0.10)
        # duplicate rate: 20% = catastrophic
        dup_norm = _clip01(telemetry.data.duplicate_rate / 0.20)
        # affected volume: rows corrupted (proxy: missing + dup rate * assumed volume)
        affected_frac = min(telemetry.data.missing_rate + telemetry.data.duplicate_rate, 1.0)
        volume_norm = _clip01(affected_frac * _ASSUMED_PREDICTIONS_PER_DAY / _ASSUMED_HIGH_VOLUME)

        severity = 0.40 * missing_norm + 0.25 * schema_norm + 0.15 * dup_norm + 0.20 * volume_norm
        return severity, SeverityBreakdown(
            components={
                "missing_rate": missing_norm,
                "schema_rate": schema_norm,
                "duplicate_rate": dup_norm,
                "affected_volume": volume_norm,
            },
            severity=severity,
        )

    def _cost_severity(
        self,
        telemetry: Telemetry,
        baseline: Telemetry | None,
        tenant_config: dict | None,
    ) -> tuple[float, SeverityBreakdown]:
        """severity = 0.70*budget_overrun + 0.30*cost_growth"""
        daily_cost_budget = (tenant_config or {}).get("daily_cost_budget", 100.0)

        # budget_overrun: how much over budget (50% over = catastrophic)
        daily_cost = telemetry.system.cost_per_prediction * _ASSUMED_PREDICTIONS_PER_DAY
        budget_ratio = daily_cost / max(daily_cost_budget, 1.0)
        overrun_norm = _clip01((budget_ratio - 1.0) / 0.50)

        # cost_growth: proportional increase from baseline (2x growth = catastrophic)
        baseline_cost = baseline.system.cost_per_prediction if baseline else telemetry.system.cost_per_prediction
        growth_ratio = (telemetry.system.cost_per_prediction - baseline_cost) / max(baseline_cost, 1e-6)
        growth_norm = _clip01(growth_ratio / 2.0)

        severity = 0.70 * overrun_norm + 0.30 * growth_norm
        return severity, SeverityBreakdown(
            components={"budget_overrun": overrun_norm, "cost_growth": growth_norm},
            severity=severity,
        )

    def _latency_severity(
        self,
        telemetry: Telemetry,
        baseline: Telemetry | None,
        tenant_config: dict | None,
    ) -> tuple[float, SeverityBreakdown]:
        """severity = 0.60*latency_ratio + 0.25*error_rate + 0.15*traffic_volume"""
        latency_sla_ms = (tenant_config or {}).get("latency_sla_ms", 100.0)

        # latency_ratio: how far over SLA (2x SLA = catastrophic)
        ratio = telemetry.system.latency_p95 / max(latency_sla_ms, 1.0)
        latency_norm = _clip01((ratio - 1.0) / 1.0)

        # error_rate: prediction errors (10% = catastrophic)
        error_norm = _clip01(telemetry.model.error_rate / 0.10)

        # traffic_volume: higher traffic → more impactful breach
        # Approximate RPS from latency p95 (assume constant; no rps in Telemetry)
        rps_estimate = _ASSUMED_PREDICTIONS_PER_DAY / 86_400.0
        traffic_norm = _clip01(rps_estimate / _ASSUMED_HIGH_RPS)

        severity = 0.60 * latency_norm + 0.25 * error_norm + 0.15 * traffic_norm
        return severity, SeverityBreakdown(
            components={"latency_ratio": latency_norm, "error_rate": error_norm, "traffic_volume": traffic_norm},
            severity=severity,
        )
