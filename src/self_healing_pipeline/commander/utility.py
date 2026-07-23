"""Phase 11: Utility scoring — convert expected_effect vectors to tenant-weighted utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from self_healing_pipeline.agents.remediation_policy import RemediationPlan
    from self_healing_pipeline.observability.incident_state import IncidentState


@dataclass(slots=True)
class UtilityWeights:
    """Per-tenant, per-incident-type utility weights.

    All weights are positive; risk is subtracted in the scoring formula.
    Components (quality + cost + reliability + speed + confidence) need not sum
    to 1.0 — the result is clipped to [-1, +1].
    """

    quality: float      # weight on auc_delta improvement
    cost: float         # weight on cost_delta reduction
    reliability: float  # weight on false_negative_rate reduction
    speed: float        # weight on latency_p95 reduction
    confidence: float   # weight on agent's own confidence
    risk: float         # multiplied by plan.risk and subtracted


# Per-incident-type defaults mirror Phase 14 reward weights + confidence term.
_DEFAULT_UTILITY_WEIGHTS: dict[str, UtilityWeights] = {
    "drift": UtilityWeights(
        quality=0.40, cost=0.05, reliability=0.15, speed=0.05, confidence=0.20, risk=0.15
    ),
    "data_quality": UtilityWeights(
        quality=0.10, cost=0.05, reliability=0.35, speed=0.05, confidence=0.25, risk=0.20
    ),
    "latency_breach": UtilityWeights(
        quality=0.05, cost=0.05, reliability=0.10, speed=0.40, confidence=0.20, risk=0.20
    ),
    "cost_threshold": UtilityWeights(
        quality=0.10, cost=0.40, reliability=0.10, speed=0.05, confidence=0.20, risk=0.15
    ),
}
_FALLBACK_WEIGHTS = UtilityWeights(
    quality=0.25, cost=0.10, reliability=0.20, speed=0.10, confidence=0.20, risk=0.15
)


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class UtilityScorer:
    """Convert a RemediationPlan's expected_effect into a utility score.

    Replaces raw-confidence ranking with a multi-dimensional assessment that
    accounts for what each agent actually promises to change (quality, cost,
    reliability, latency) weighted by incident type and tenant preferences.
    """

    @staticmethod
    def score(
        plan: RemediationPlan,
        incident_state: IncidentState,
        weights: UtilityWeights | None = None,
    ) -> float:
        """Return utility ∈ [-1, +1] for a remediation plan.

        Normalisation denominators come from the live IncidentState so the
        scale is calibrated to the tenant's current operating point, not
        hardcoded constants.

        Args:
            plan:           plan to score
            incident_state: pre-action IncidentState (provides normalisers)
            weights:        UtilityWeights; defaults to per-incident-type preset

        Returns:
            utility in [-1, +1]; higher = better
        """
        if weights is None:
            weights = _DEFAULT_UTILITY_WEIGHTS.get(
                incident_state.incident_type, _FALLBACK_WEIGHTS
            )

        eff: dict[str, Any] = plan.expected_effect or {}

        # --- Quality: AUC improvement ---
        auc_delta = eff.get("auc_delta", 0.0)
        quality_val = _clip(auc_delta / 0.10)  # 10% AUC gain = +1

        # --- Cost: operational cost reduction (negative delta = cheaper) ---
        cost_delta = eff.get("cost_delta_usd", 0.0)
        cost_ref = max(incident_state.cost_per_1000_predictions * 0.20, 0.10)
        cost_val = _clip(-cost_delta / cost_ref)  # negative delta → positive utility

        # --- Reliability: false-negative rate improvement ---
        fnr_delta = eff.get("false_negative_rate_delta", 0.0)
        fnr_ref = max(incident_state.false_negative_rate, 0.01)
        reliability_val = _clip(-fnr_delta / fnr_ref)  # negative delta = fewer FNs = good

        # --- Speed: latency improvement (negative delta = faster) ---
        lat_delta = eff.get("latency_p95_delta_ms", 0.0)
        lat_ref = max(incident_state.latency_sla_ms, 50.0)
        speed_val = _clip(-lat_delta / lat_ref)  # negative delta → positive utility

        # --- Availability bonus (FallbackAgent) ---
        avail_delta = eff.get("availability_delta", 0.0)
        reliability_val = _clip(reliability_val + avail_delta)

        # --- Confidence (agent's own self-assessment, already in [0,1]) ---
        confidence_val = float(plan.confidence)

        # --- Risk (subtract) ---
        risk_val = float(plan.risk)

        raw = (
            weights.quality * quality_val
            + weights.cost * cost_val
            + weights.reliability * reliability_val
            + weights.speed * speed_val
            + weights.confidence * confidence_val
            - weights.risk * risk_val
        )
        return _clip(raw)

    @staticmethod
    def weights_from_tier_config(tier_config: Any, incident_type: str) -> UtilityWeights:
        """Build UtilityWeights from a TenantTierConfig DB row.

        Maps existing TenantTierConfig columns to utility dimensions:
          business_value_weight  → quality
          cost_efficiency_weight → cost
          risk_inverse_weight    → risk
          time_inverse_weight    → speed
          confidence_weight      → confidence

        Reliability inherits from the per-incident-type default since there is
        no dedicated column in TenantTierConfig yet.

        Args:
            tier_config: TenantTierConfig ORM row (or None → returns defaults)
            incident_type: for per-type reliability default
        """
        if tier_config is None:
            return _DEFAULT_UTILITY_WEIGHTS.get(incident_type, _FALLBACK_WEIGHTS)

        default = _DEFAULT_UTILITY_WEIGHTS.get(incident_type, _FALLBACK_WEIGHTS)
        return UtilityWeights(
            quality=float(getattr(tier_config, "business_value_weight", default.quality)),
            cost=float(getattr(tier_config, "cost_efficiency_weight", default.cost)),
            reliability=default.reliability,  # no column yet; keep per-type default
            speed=float(getattr(tier_config, "time_inverse_weight", default.speed)),
            confidence=float(getattr(tier_config, "confidence_weight", default.confidence)),
            risk=float(getattr(tier_config, "risk_inverse_weight", default.risk)),
        )

    @staticmethod
    def rank(
        plans: list[tuple[Any, RemediationPlan]],
        incident_state: IncidentState,
        weights: UtilityWeights | None = None,
    ) -> list[tuple[Any, RemediationPlan, float]]:
        """Sort (agent, plan) pairs by utility descending.

        Returns:
            List of (agent, plan, utility_score) sorted highest-first.
        """
        scored = [
            (agent, plan, UtilityScorer.score(plan, incident_state, weights))
            for agent, plan in plans
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored
