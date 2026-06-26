from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from self_healing_pipeline.gateway.events import Incident, IncidentType


@dataclass(slots=True)
class DriftResult:
    """Result of drift detection."""

    drift_detected: bool
    drift_percentage: float
    drifted_features: list[str]
    report: dict[str, Any]


class DriftMonitor:
    """Detect feature drift using Evidently AI."""

    def __init__(self, sensitivity: float = 0.5) -> None:
        """Initialize drift monitor.

        Args:
            sensitivity: threshold for drift detection (0-1). Higher = more sensitive.
        """
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError(f"sensitivity must be in [0, 1], got {sensitivity}")
        self.sensitivity = sensitivity

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> DriftResult:
        """Detect drift between reference and current datasets.

        Uses statistical comparison: if mean/std differ significantly, column has drifted.

        Args:
            reference: baseline/training dataset
            current: new dataset to check for drift

        Returns:
            DriftResult with drift metrics and drifted features
        """
        if len(reference) == 0 or len(current) == 0:
            return DriftResult(
                drift_detected=False,
                drift_percentage=0.0,
                drifted_features=[],
                report={},
            )

        drifted_features = []

        # Check each numeric column for drift (simple: compare means)
        for col in reference.columns:
            if col not in current.columns:
                continue

            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(reference[col]):
                continue

            ref_mean = reference[col].mean()
            curr_mean = current[col].mean()
            ref_std = reference[col].std()

            if ref_std > 0:
                # Normalized difference: (curr - ref) / ref_std
                normalized_diff = abs(curr_mean - ref_mean) / ref_std
                # Drift if difference > 1 std dev
                if normalized_diff > 1.0:
                    drifted_features.append(col)

        total_features = len(reference.columns)
        drifted_count = len(drifted_features)
        drift_percentage = drifted_count / max(total_features, 1) if total_features > 0 else 0.0

        # Threshold: drift detected if > sensitivity
        drift_detected = drift_percentage > self.sensitivity

        return DriftResult(
            drift_detected=drift_detected,
            drift_percentage=drift_percentage,
            drifted_features=drifted_features,
            report={
                "method": "statistical_comparison",
                "total_columns": total_features,
                "drifted_count": drifted_count,
                "drift_percentage": drift_percentage,
            },
        )

    def make_incident(
        self,
        drift_result: DriftResult,
        tenant_id: str,
    ) -> Incident | None:
        """Convert drift result to incident if drift detected.

        Args:
            drift_result: result from detect()
            tenant_id: tenant identifier

        Returns:
            Incident if drift detected, None otherwise
        """
        if not drift_result.drift_detected:
            return None

        return Incident(
            tenant_id=tenant_id,
            type=IncidentType.DRIFT,
            payload={
                "drift_percentage": drift_result.drift_percentage,
                "drifted_features": drift_result.drifted_features,
            },
            severity=min(drift_result.drift_percentage, 1.0),
            affected_features=tuple(drift_result.drifted_features),
        )
