from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from self_healing_pipeline.gateway.events import Incident, IncidentType


@dataclass(slots=True)
class DataQualityResult:
    """Result of data quality check."""

    quality_ok: bool
    missing_rate: float
    duplicate_rate: float
    schema_violations: int
    volume: int
    issues: list[str]
    report: dict[str, Any]


class DataQualityMonitor:
    """Monitor data quality: missing values, duplicates, schema, volume."""

    def __init__(
        self,
        *,
        missing_threshold: float = 0.1,
        duplicate_threshold: float = 0.05,
        schema_violation_threshold: int = 0,
        volume_floor: int = 10,
    ) -> None:
        """Initialize data quality monitor.

        Args:
            missing_threshold: alert if missing rate > this (default 10%)
            duplicate_threshold: alert if duplicate rate > this (default 5%)
            schema_violation_threshold: alert if violations > this (default 0)
            volume_floor: alert if rows < this (default 10)
        """
        if not 0.0 <= missing_threshold <= 1.0:
            raise ValueError(f"missing_threshold must be in [0, 1], got {missing_threshold}")
        if not 0.0 <= duplicate_threshold <= 1.0:
            raise ValueError(f"duplicate_threshold must be in [0, 1], got {duplicate_threshold}")
        if volume_floor < 1:
            raise ValueError(f"volume_floor must be >= 1, got {volume_floor}")

        self.missing_threshold = missing_threshold
        self.duplicate_threshold = duplicate_threshold
        self.schema_violation_threshold = schema_violation_threshold
        self.volume_floor = volume_floor

    def detect(self, df: pd.DataFrame) -> DataQualityResult:
        """Detect data quality issues in dataframe.

        Args:
            df: dataframe to check

        Returns:
            DataQualityResult with metrics and issues
        """
        if len(df) == 0:
            return DataQualityResult(
                quality_ok=False,
                missing_rate=1.0,
                duplicate_rate=0.0,
                schema_violations=0,
                volume=0,
                issues=["empty dataframe"],
                report={},
            )

        issues = []

        # Check missing values
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        missing_rate = missing_cells / total_cells if total_cells > 0 else 0.0
        if missing_rate > self.missing_threshold:
            issues.append(f"missing_rate={missing_rate:.1%} > {self.missing_threshold:.1%}")

        # Check duplicates
        n_duplicates = df.duplicated().sum()
        duplicate_rate = n_duplicates / len(df) if len(df) > 0 else 0.0
        if duplicate_rate > self.duplicate_threshold:
            issues.append(f"duplicate_rate={duplicate_rate:.1%} > {self.duplicate_threshold:.1%}")

        # Check schema violations (non-numeric in numeric columns)
        schema_violations = 0
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Count non-convertible values
                numeric_df = pd.to_numeric(df[col], errors="coerce")
                schema_violations += numeric_df.isna().sum()

        if schema_violations > self.schema_violation_threshold:
            issues.append(
                f"schema_violations={schema_violations} > "
                f"{self.schema_violation_threshold}"
            )

        # Check volume floor
        if len(df) < self.volume_floor:
            issues.append(f"volume={len(df)} < {self.volume_floor}")

        quality_ok = len(issues) == 0

        return DataQualityResult(
            quality_ok=quality_ok,
            missing_rate=missing_rate,
            duplicate_rate=duplicate_rate,
            schema_violations=schema_violations,
            volume=len(df),
            issues=issues,
            report={
                "missing_rate": missing_rate,
                "duplicate_rate": duplicate_rate,
                "schema_violations": schema_violations,
                "volume": len(df),
            },
        )

    def make_incident(
        self,
        result: DataQualityResult,
        tenant_id: str,
    ) -> Incident | None:
        """Convert quality result to incident if quality degraded.

        Args:
            result: result from detect()
            tenant_id: tenant identifier

        Returns:
            Incident if quality issues detected, None otherwise
        """
        if result.quality_ok:
            return None

        # Severity: how bad is it? (0-1)
        severity = (
            result.missing_rate * 0.3
            + result.duplicate_rate * 0.3
            + (min(result.schema_violations, 100) / 100) * 0.2
            + (0.2 if len(result.issues) > 0 else 0.0)
        )

        return Incident(
            tenant_id=tenant_id,
            type=IncidentType.DATA_QUALITY,
            payload={
                "missing_rate": result.missing_rate,
                "duplicate_rate": result.duplicate_rate,
                "schema_violations": result.schema_violations,
                "volume": result.volume,
                "issues": result.issues,
            },
            severity=min(severity, 1.0),
            affected_features=(),
        )
