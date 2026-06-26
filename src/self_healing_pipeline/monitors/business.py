from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from self_healing_pipeline.gateway.events import Incident, IncidentType


@dataclass(slots=True)
class PredictionOutcome:
    """Single prediction outcome."""

    y_true: int  # 0 or 1
    y_pred: int  # 0 or 1


@dataclass(slots=True)
class BusinessCostResult:
    """Result of business cost check."""

    cost_ok: bool
    cost_per_prediction: float
    total_cost: float
    false_positives: int
    false_negatives: int
    predictions_evaluated: int
    report: dict[str, Any]


class BusinessCostMonitor:
    """Monitor business cost: FP/FN × cost matrix, rolling window."""

    def __init__(
        self,
        *,
        false_positive_cost: float = 5.0,
        false_negative_cost: float = 50.0,
        cost_threshold: float = 100.0,
        window_size: int = 100,
    ) -> None:
        """Initialize business cost monitor.

        Args:
            false_positive_cost: cost per FP prediction
            false_negative_cost: cost per FN prediction
            cost_threshold: alert if cost_per_prediction > this
            window_size: rolling window for cost calculation (predictions)
        """
        if false_positive_cost < 0:
            raise ValueError(f"false_positive_cost must be >= 0, got {false_positive_cost}")
        if false_negative_cost < 0:
            raise ValueError(f"false_negative_cost must be >= 0, got {false_negative_cost}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        self.false_positive_cost = false_positive_cost
        self.false_negative_cost = false_negative_cost
        self.cost_threshold = cost_threshold
        self.window_size = window_size
        self.outcomes: deque[PredictionOutcome] = deque(maxlen=window_size)

    def record_prediction(self, y_true: int, y_pred: int) -> None:
        """Record a prediction outcome.

        Args:
            y_true: ground truth label (0 or 1)
            y_pred: predicted label (0 or 1)
        """
        if y_true not in (0, 1):
            raise ValueError(f"y_true must be 0 or 1, got {y_true}")
        if y_pred not in (0, 1):
            raise ValueError(f"y_pred must be 0 or 1, got {y_pred}")

        self.outcomes.append(PredictionOutcome(y_true=y_true, y_pred=y_pred))

    def detect(self) -> BusinessCostResult:
        """Detect if business cost exceeds threshold.

        Returns:
            BusinessCostResult with cost metrics
        """
        if len(self.outcomes) == 0:
            return BusinessCostResult(
                cost_ok=True,
                cost_per_prediction=0.0,
                total_cost=0.0,
                false_positives=0,
                false_negatives=0,
                predictions_evaluated=0,
                report={},
            )

        false_positives = sum(
            1 for o in self.outcomes if o.y_true == 0 and o.y_pred == 1
        )
        false_negatives = sum(
            1 for o in self.outcomes if o.y_true == 1 and o.y_pred == 0
        )

        total_cost = (
            false_positives * self.false_positive_cost
            + false_negatives * self.false_negative_cost
        )
        cost_per_prediction = total_cost / len(self.outcomes) if len(self.outcomes) > 0 else 0.0

        cost_ok = cost_per_prediction <= self.cost_threshold

        return BusinessCostResult(
            cost_ok=cost_ok,
            cost_per_prediction=cost_per_prediction,
            total_cost=total_cost,
            false_positives=false_positives,
            false_negatives=false_negatives,
            predictions_evaluated=len(self.outcomes),
            report={
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "total_cost": total_cost,
                "cost_per_prediction": cost_per_prediction,
                "window_size": len(self.outcomes),
            },
        )

    def make_incident(
        self,
        result: BusinessCostResult,
        tenant_id: str,
    ) -> Incident | None:
        """Convert cost result to incident if threshold exceeded.

        Args:
            result: result from detect()
            tenant_id: tenant identifier

        Returns:
            Incident if cost exceeds threshold, None otherwise
        """
        if result.cost_ok:
            return None

        # Severity: ratio of current to threshold
        severity = min(result.cost_per_prediction / max(self.cost_threshold, 1.0), 1.0)

        return Incident(
            tenant_id=tenant_id,
            type=IncidentType.COST_THRESHOLD,
            payload={
                "cost_per_prediction": result.cost_per_prediction,
                "total_cost": result.total_cost,
                "false_positives": result.false_positives,
                "false_negatives": result.false_negatives,
                "threshold": self.cost_threshold,
            },
            severity=severity,
            affected_features=(),
        )
