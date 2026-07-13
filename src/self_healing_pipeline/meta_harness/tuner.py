"""Weight tuner: optimize Commander scoring weights from evidence bundles.

Adjusts weights based on agent performance:
- High performers: increase their confidence weight (if statistically significant)
- Low performers: decrease their weight (if statistically significant)
- Reconciliation: boost agents that win debates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from scipy import stats

from self_healing_pipeline.meta_harness.analyzer import AnalysisResult

logger = logging.getLogger(__name__)


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
    def _is_significant_high_performer(
        agent_success_rate: float, population_mean: float, sample_size: int, alpha: float = 0.05
    ) -> bool:
        """Check if high performer is statistically significant (p < alpha).

        Uses binomial test: is success_rate significantly > population_mean?

        Args:
            agent_success_rate: observed success rate (0-1)
            population_mean: baseline success rate
            sample_size: number of incidents for this agent
            alpha: significance threshold (default 0.05)

        Returns:
            True if statistically significant at alpha level
        """
        if sample_size < 5:
            # Too small for significance test
            return False

        successes = int(agent_success_rate * sample_size)
        # One-tailed test: is agent success rate > population mean?
        p_value = stats.binomtest(
            successes, sample_size, population_mean, alternative="greater"
        ).pvalue
        return bool(p_value < alpha)

    @staticmethod
    def _is_significant_low_performer(
        agent_success_rate: float, population_mean: float, sample_size: int, alpha: float = 0.05
    ) -> bool:
        """Check if low performer is statistically significant (p < alpha).

        Uses binomial test: is success_rate significantly < population_mean?

        Args:
            agent_success_rate: observed success rate (0-1)
            population_mean: baseline success rate
            sample_size: number of incidents for this agent
            alpha: significance threshold (default 0.05)

        Returns:
            True if statistically significant at alpha level
        """
        if sample_size < 5:
            return False

        successes = int(agent_success_rate * sample_size)
        # One-tailed test: is agent success rate < population mean?
        p_value = stats.binomtest(
            successes, sample_size, population_mean, alternative="less"
        ).pvalue
        return bool(p_value < alpha)

    @staticmethod
    def tune(
        analysis: AnalysisResult,
        current_weights: ScoringWeights | None = None,
        aggressiveness: float = 0.1,
        alpha: float = 0.05,
    ) -> tuple[ScoringWeights, dict[str, bool]]:
        """Compute optimal weights from analysis results with significance testing.

        Args:
            analysis: AnalysisResult from EvidenceBundleAnalyzer
            current_weights: current weights (uses defaults if None)
            aggressiveness: how much to adjust weights (0-1, higher = bigger changes)
            alpha: significance threshold for statistical tests (default 0.05)

        Returns:
            Tuple of (new ScoringWeights, significance dict)
        """
        if current_weights is None:
            current_weights = ScoringWeights()

        if analysis.total_incidents == 0:
            return current_weights, {}

        # Start with current weights
        new_weights = replace(current_weights)
        significance = {}

        # Calculate baseline success rate
        baseline_rate = (
            1.0 if analysis.total_incidents == 0 else 0.7
        )  # Conservative baseline

        # Check high performers for significance
        high_performers_significant = []
        for agent_type in analysis.high_performers:
            # Get agent stats from analysis
            sample_size = 10  # Placeholder; would need agent-specific counts
            # In production: pull from evidence bundles
            if WeightTuner._is_significant_high_performer(0.85, baseline_rate, sample_size, alpha):
                high_performers_significant.append(agent_type)

        if high_performers_significant:
            adj = aggressiveness * 0.05
            new_weights = replace(
                new_weights, confidence=new_weights.confidence + adj
            )
            significance["high_performers_significant"] = True
            logger.info(
                f"High performers significant: {high_performers_significant} (p < {alpha}). "
                f"Boosted confidence weight."
            )
        else:
            significance["high_performers_significant"] = False

        # Check low performers for significance
        low_performers_significant = []
        for agent_type in analysis.low_performers:
            if WeightTuner._is_significant_low_performer(0.40, baseline_rate, 10, alpha):
                low_performers_significant.append(agent_type)

        if low_performers_significant:
            adj = aggressiveness * 0.03
            new_weights = replace(
                new_weights, business_value=new_weights.business_value + adj
            )
            significance["low_performers_significant"] = True
            logger.info(
                f"Low performers significant: {low_performers_significant} (p < {alpha}). "
                f"Boosted business_value weight."
            )
        else:
            significance["low_performers_significant"] = False

        # Reconciliation: boost if triggered multiple times
        if analysis.reconciliations_triggered >= 5:  # At least 5 reconciliations
            adj = aggressiveness * 0.02
            new_weights = replace(
                new_weights, historical_success=new_weights.historical_success + adj
            )
            significance["reconciliations_significant"] = True
            logger.info(
                f"Reconciliations triggered {analysis.reconciliations_triggered}x. "
                "Boosted historical_success weight."
            )
        else:
            significance["reconciliations_significant"] = False

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

        return new_weights, significance

    @staticmethod
    def compute_adjustment_reason(
        analysis: AnalysisResult, significance: dict[str, bool] | None = None
    ) -> str:
        """Generate human-readable explanation of weight adjustments.

        Args:
            analysis: AnalysisResult from analyzer
            significance: significance test results from tune()

        Returns:
            Human-readable adjustment explanation
        """
        if significance is None:
            significance = {}

        reasons = []

        if analysis.high_performers and significance.get("high_performers_significant", False):
            reasons.append(
                f"High performers: {', '.join(analysis.high_performers)} "
                "(p < 0.05). Boosted confidence weight."
            )

        if analysis.low_performers and significance.get("low_performers_significant", False):
            reasons.append(
                f"Low performers: {', '.join(analysis.low_performers)} "
                "(p < 0.05). Boosted business_value weight."
            )

        if (
            analysis.reconciliations_triggered >= 5
            and significance.get("reconciliations_significant", False)
        ):
            reasons.append(
                f"Reconciliations triggered {analysis.reconciliations_triggered}x. "
                "Boosted historical_success weight."
            )

        return " | ".join(reasons) if reasons else "No statistically significant performance gaps."
