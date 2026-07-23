"""Phase 13: Explicit multi-dimensional guardrail checking and rollback decision.

Two-part verdict per action:
  resolved       -- post-action metrics satisfy all policy limits
  no_regression  -- no metric worsened significantly vs before the action
  should_rollback -- True only when BOTH conditions fail (unsafe + degraded)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from self_healing_pipeline.observability.incident_state import IncidentState


@dataclass
class GuardrailResult:
    """Verdict of the post-action guardrail check."""

    resolved: bool
    no_regression: bool
    should_rollback: bool
    violations: list[str] = field(default_factory=list)

    # Per-dimension outcomes (for dashboard / logs)
    auc_ok: bool = True
    latency_ok: bool = True
    missing_ok: bool = True
    cost_ok: bool = True
    fnr_ok: bool = True


class GuardrailChecker:
    """Check explicit guardrails after action execution.

    Resolution guardrails (vs policy limits carried in state_after):
      AUC     >= min_auc
      latency <= latency_sla_ms
      missing <= max_missing_rate

    Regression guardrails (vs state_before baseline):
      cost_per_1000 <= before * 1.10
      false_negative_rate <= before + 0.05

    Rollback triggered when BOTH sets fail — the action didn't help AND made
    things measurably worse.
    """

    ALLOWED_COST_MULTIPLIER: float = 1.10
    ALLOWED_FNR_INCREASE: float = 0.05

    @staticmethod
    def check(
        state_before: IncidentState,
        state_after: IncidentState,
    ) -> GuardrailResult:
        """Evaluate all guardrails.

        Args:
            state_before: IncidentState snapshot before action execution
            state_after:  IncidentState snapshot after stabilization window

        Returns:
            GuardrailResult with resolved / no_regression / should_rollback verdicts
        """
        violations: list[str] = []

        # --- Resolution guardrails: must be within policy limits after action ---

        auc_ok = (
            state_after.current_auc is None
            or state_after.current_auc >= state_after.min_auc
        )
        if not auc_ok:
            violations.append(
                f"auc={state_after.current_auc:.3f} < min_auc={state_after.min_auc:.3f}"
            )

        latency_ok = state_after.latency_p95_ms <= state_after.latency_sla_ms
        if not latency_ok:
            violations.append(
                f"latency_p95={state_after.latency_p95_ms:.1f}ms"
                f" > sla={state_after.latency_sla_ms:.1f}ms"
            )

        missing_ok = state_after.missing_rate <= state_after.max_missing_rate
        if not missing_ok:
            violations.append(
                f"missing_rate={state_after.missing_rate:.3f}"
                f" > max={state_after.max_missing_rate:.3f}"
            )

        resolved = auc_ok and latency_ok and missing_ok

        # --- Regression guardrails: nothing should get significantly worse ---

        cost_ok = (
            state_after.cost_per_1000_predictions
            <= state_before.cost_per_1000_predictions
            * GuardrailChecker.ALLOWED_COST_MULTIPLIER
        )
        if not cost_ok:
            ceiling = (
                state_before.cost_per_1000_predictions
                * GuardrailChecker.ALLOWED_COST_MULTIPLIER
            )
            violations.append(
                f"cost={state_after.cost_per_1000_predictions:.4f}"
                f" > {GuardrailChecker.ALLOWED_COST_MULTIPLIER}x before"
                f" ({ceiling:.4f})"
            )

        fnr_ceiling = (
            state_before.false_negative_rate + GuardrailChecker.ALLOWED_FNR_INCREASE
        )
        fnr_ok = state_after.false_negative_rate <= fnr_ceiling
        if not fnr_ok:
            violations.append(
                f"fnr={state_after.false_negative_rate:.3f}"
                f" > before+{GuardrailChecker.ALLOWED_FNR_INCREASE}"
                f" ({fnr_ceiling:.3f})"
            )

        no_regression = cost_ok and fnr_ok

        # Rollback only when incident not resolved AND regression detected
        should_rollback = not resolved and not no_regression

        return GuardrailResult(
            resolved=resolved,
            no_regression=no_regression,
            should_rollback=should_rollback,
            violations=violations,
            auc_ok=auc_ok,
            latency_ok=latency_ok,
            missing_ok=missing_ok,
            cost_ok=cost_ok,
            fnr_ok=fnr_ok,
        )
