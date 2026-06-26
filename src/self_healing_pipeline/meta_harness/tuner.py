"""Weight tuner: optimize Commander scoring weights from evidence bundles.

Adjusts weights based on agent performance:
- High performers: increase their confidence weight
- Low performers: decrease their weight
- Reconciliation: boost agents that win debates
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from self_healing_pipeline.meta_harness.analyzer import AnalysisResult


@dataclass(slots=True)
class ScoringWeights:
    """Commander scoring weights."""

    business_value: float = 0.30
    confidence: float = 0.20
    risk_inverse: float = 0.20
    cost_efficiency: float = 0.10
    time_inverse: float = 0.05
    historical_success: float = 0.15

    def total(self) -> float:
        """Sum of all weights (should always be 1.0)."""
        return (
            self.business_value
            + self.confidence
            + self.risk_inverse
            + self.cost_efficiency
            + self.time_inverse
            + self.historical_success
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dict for JSON serialization."""
        return {
            "business_value": self.business_value,
            "confidence": self.confidence,
            "risk_inverse": self.risk_inverse,
            "cost_efficiency": self.cost_efficiency,
            "time_inverse": self.time_inverse,
            "historical_success": self.historical_success,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> ScoringWeights:
        """Create from dict."""
        return cls(
            business_value=data.get("business_value", 0.30),
            confidence=data.get("confidence", 0.20),
            risk_inverse=data.get("risk_inverse", 0.20),
            cost_efficiency=data.get("cost_efficiency", 0.10),
            time_inverse=data.get("time_inverse", 0.05),
            historical_success=data.get("historical_success", 0.15),
        )


class WeightTuner:
    """Optimize Commander weights based on agent performance."""

    @staticmethod
    def tune(
        analysis: AnalysisResult,
        current_weights: ScoringWeights | None = None,
        aggressiveness: float = 0.1,
    ) -> ScoringWeights:
        """Compute optimal weights from analysis results.

        Args:
            analysis: AnalysisResult from EvidenceBundleAnalyzer
            current_weights: current weights (uses defaults if None)
            aggressiveness: how much to adjust weights (0-1, higher = bigger changes)

        Returns:
            New ScoringWeights optimized for agent performance
        """
        if current_weights is None:
            current_weights = ScoringWeights()

        if analysis.total_incidents == 0:
            return current_weights

        # Start with current weights
        new_weights = replace(current_weights)

        # Adjust based on high/low performers
        # High performers likely have high confidence/success rates
        # → boost confidence weight to reward well-estimated agents
        if analysis.high_performers:
            adj = aggressiveness * 0.05  # Max +5% to confidence
            new_weights = replace(
                new_weights, confidence=new_weights.confidence + adj
            )

        # Low performers likely overestimate or fail frequently
        # → boost business_value weight to favor high-savings estimates
        # (which low performers tend to not achieve, so they'll score lower)
        if analysis.low_performers:
            adj = aggressiveness * 0.03  # Max +3% to business_value
            new_weights = replace(
                new_weights, business_value=new_weights.business_value + adj
            )

        # Reconciliation insights: boost historical_success
        # (it captures which agents actually win close calls)
        if analysis.reconciliations_triggered > 0:
            adj = aggressiveness * 0.02  # Max +2% to historical_success
            new_weights = replace(
                new_weights, historical_success=new_weights.historical_success + adj
            )

        # Normalize to sum to 1.0
        total = new_weights.total()
        if total > 0:
            scale = 1.0 / total
            new_weights = replace(
                new_weights,
                business_value=new_weights.business_value * scale,
                confidence=new_weights.confidence * scale,
                risk_inverse=new_weights.risk_inverse * scale,
                cost_efficiency=new_weights.cost_efficiency * scale,
                time_inverse=new_weights.time_inverse * scale,
                historical_success=new_weights.historical_success * scale,
            )

        return new_weights

    @staticmethod
    def compute_adjustment_reason(
        analysis: AnalysisResult,
    ) -> str:
        """Generate human-readable explanation of weight adjustments."""
        reasons = []

        if analysis.high_performers:
            reasons.append(
                f"High performers: {', '.join(analysis.high_performers)}. "
                "Boosted confidence weight."
            )

        if analysis.low_performers:
            reasons.append(
                f"Low performers: {', '.join(analysis.low_performers)}. "
                "Boosted business_value weight to penalize overestimation."
            )

        if analysis.reconciliations_triggered > 0:
            reasons.append(
                f"Reconciliations triggered {analysis.reconciliations_triggered}x. "
                "Boosted historical_success weight."
            )

        return " | ".join(reasons) if reasons else "No significant performance gaps."
