"""Tune Commander scoring weights from real evidence bundles and apply them live.

Closes the meta-harness loop end to end: analyze evidence bundles written by
CommanderV3 -> tune ScoringWeights -> version them -> apply to each tenant's
TenantTierConfig row, which UtilityScorer.weights_from_tier_config() reads on
every incident. Without this, tuned weights only ever reached versioned JSON
files and never affected a live decision.

Usage:
    uv run python scripts/tune_weights.py
    uv run python scripts/tune_weights.py --dry-run
    uv run python scripts/tune_weights.py --aggressiveness 0.2 --alpha 0.1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger("tune_weights")


def run_tune_cycle(
    traces_dir: Path,
    session: Session,
    *,
    versions_dir: Path,
    aggressiveness: float = 0.1,
    alpha: float = 0.05,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Analyze evidence bundles, tune weights, and apply them to live tenants.

    Args:
        traces_dir: directory of per-incident evidence_bundle.json files
        session: active DB session (must see the same DB CommanderV3 reads)
        versions_dir: where tuned weight versions are persisted
        aggressiveness: WeightTuner adjustment strength (0-1)
        alpha: significance threshold for WeightTuner's tests
        dry_run: if True, analyze/tune but skip saving a version and applying to the DB

    Returns:
        Summary dict: total_incidents, adjustment_reason, old_weights, new_weights,
        tenants_updated.
    """
    from self_healing_pipeline.db.models import IncidentHistory
    from self_healing_pipeline.meta_harness.analyzer import EvidenceBundleAnalyzer
    from self_healing_pipeline.meta_harness.apply import sync_tuned_weights
    from self_healing_pipeline.meta_harness.tuner import ScoringWeights, WeightTuner
    from self_healing_pipeline.meta_harness.version_control import WeightVersionControl

    analysis = EvidenceBundleAnalyzer(traces_dir).analyze()
    if analysis.total_incidents == 0:
        logger.info("No evidence bundles found in %s — nothing to tune.", traces_dir)
        return {"total_incidents": 0, "tenants_updated": []}

    vc = WeightVersionControl(versions_dir)
    latest = vc.load_latest()
    current_weights = ScoringWeights.from_dict(latest.weights) if latest else ScoringWeights()

    new_weights, significance = WeightTuner.tune(
        analysis, current_weights, aggressiveness=aggressiveness, alpha=alpha
    )
    adjustment_reason = WeightTuner.compute_adjustment_reason(analysis, significance)

    logger.info("Analyzed %d incidents across %d agent(s)", analysis.total_incidents, len(analysis.agent_metrics))
    logger.info("Adjustment reason: %s", adjustment_reason)
    for field in ("business_value", "confidence", "risk_inverse", "cost_efficiency", "time_inverse", "historical_success"):
        old_v, new_v = getattr(current_weights, field), getattr(new_weights, field)
        logger.info("  %-20s %.3f -> %.3f", field, old_v, new_v)

    tenants_updated: list[str] = []
    if dry_run:
        logger.info("[dry-run] Skipping version save and DB apply.")
    else:
        version = vc.save_version(
            new_weights.to_dict(),
            adjustment_reason,
            validation_stats={
                "high_performers": len(analysis.high_performers),
                "low_performers": len(analysis.low_performers),
                "reconciliations": analysis.reconciliations_triggered,
            },
        )
        logger.info("Saved weight version v%d", version.version)

        tenant_ids = [
            row[0] for row in session.query(IncidentHistory.tenant_id).distinct().all()
        ]
        for tenant_id in tenant_ids:
            sync_tuned_weights(session, tenant_id, new_weights)
            tenants_updated.append(tenant_id)
        logger.info("Applied tuned weights to TenantTierConfig for tenants: %s", tenants_updated)

    return {
        "total_incidents": analysis.total_incidents,
        "adjustment_reason": adjustment_reason,
        "old_weights": current_weights.to_dict(),
        "new_weights": new_weights.to_dict(),
        "tenants_updated": tenants_updated,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggressiveness", type=float, default=0.1, help="weight adjustment strength (0-1)")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance threshold")
    parser.add_argument("--dry-run", action="store_true", help="analyze and tune but don't persist or apply")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from self_healing_pipeline.config import get_settings
    from self_healing_pipeline.db.session import create_all, get_engine

    settings = get_settings()
    create_all()

    with Session(get_engine(), future=True) as session:
        summary = run_tune_cycle(
            settings.traces_dir,
            session,
            versions_dir=settings.weight_versions_dir,
            aggressiveness=args.aggressiveness,
            alpha=args.alpha,
            dry_run=args.dry_run,
        )

    if summary["total_incidents"] == 0:
        sys.exit(0)


if __name__ == "__main__":
    main()
