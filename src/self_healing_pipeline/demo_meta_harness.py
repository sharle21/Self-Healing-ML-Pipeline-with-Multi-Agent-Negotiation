"""Day 21 Demo: Meta-Harness end-to-end (analyze → tune → version)."""

import json
import logging
import tempfile
from pathlib import Path

from self_healing_pipeline.meta_harness.analyzer import EvidenceBundleAnalyzer
from self_healing_pipeline.meta_harness.tuner import ScoringWeights, WeightTuner
from self_healing_pipeline.meta_harness.version_control import WeightVersionControl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_sample_bundles(traces_dir: Path) -> None:
    """Create sample evidence bundles for demo."""
    bundles = [
        # Threshold wins, succeeds
        {
            "incident": {"type": "DRIFT", "severity": 0.2},
            "all_proposals": [
                {"agent_type": "threshold", "estimated_business_savings": 500},
                {"agent_type": "retrain", "estimated_business_savings": 5000},
            ],
            "winner": {"agent_type": "threshold"},
            "execution_result": {"success": True, "actual_business_savings": 480},
            "reconciliation": None,
        },
        # Retrain wins, succeeds
        {
            "incident": {"type": "DRIFT", "severity": 0.8},
            "all_proposals": [
                {"agent_type": "threshold", "estimated_business_savings": 500},
                {"agent_type": "retrain", "estimated_business_savings": 5000},
            ],
            "winner": {"agent_type": "retrain"},
            "execution_result": {"success": True, "actual_business_savings": 4800},
            "reconciliation": {"winner_type": "retrain"},
        },
        # Rollback wins, succeeds
        {
            "incident": {"type": "DRIFT", "severity": 0.45},
            "all_proposals": [
                {"agent_type": "rollback", "estimated_business_savings": 1500},
                {"agent_type": "retrain", "estimated_business_savings": 5000},
            ],
            "winner": {"agent_type": "rollback"},
            "execution_result": {"success": True, "actual_business_savings": 1400},
            "reconciliation": None,
        },
        # Threshold wins again
        {
            "incident": {"type": "COST_THRESHOLD", "severity": 0.3},
            "all_proposals": [
                {"agent_type": "threshold", "estimated_business_savings": 700},
            ],
            "winner": {"agent_type": "threshold"},
            "execution_result": {"success": True, "actual_business_savings": 680},
            "reconciliation": None,
        },
        # Fallback wins, succeeds
        {
            "incident": {"type": "DATA_QUALITY", "severity": 0.5},
            "all_proposals": [
                {"agent_type": "fallback", "estimated_business_savings": 300},
                {"agent_type": "data_repair", "estimated_business_savings": 3000},
            ],
            "winner": {"agent_type": "fallback"},
            "execution_result": {"success": True, "actual_business_savings": 280},
            "reconciliation": None,
        },
    ]

    for i, bundle in enumerate(bundles):
        bundle_dir = traces_dir / f"inc-{i+1}"
        bundle_dir.mkdir()
        with open(bundle_dir / "evidence_bundle.json", "w") as f:
            json.dump(bundle, f)


def run_demo() -> None:
    """Run meta-harness demo."""
    logger.info("=" * 80)
    logger.info("DAY 21 META-HARNESS DEMO: Self-Optimization Loop")
    logger.info("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        traces_dir = Path(tmpdir) / "traces"
        traces_dir.mkdir()
        versions_dir = Path(tmpdir) / "weight_versions"

        # Step 1: Create sample bundles
        create_sample_bundles(traces_dir)
        logger.info("\nStep 1: Created 5 sample evidence bundles")
        logger.info("  - Threshold: 2 wins, 100% success (estimates: 500, 700)")
        logger.info("  - Retrain: 1 win, 100% success (estimates: 5000)")
        logger.info("  - Rollback: 1 win, 100% success (estimates: 1500)")
        logger.info("  - Fallback: 1 win, 100% success (estimates: 300)")

        # Step 2: Analyze bundles
        analyzer = EvidenceBundleAnalyzer(traces_dir)
        analysis = analyzer.analyze()
        logger.info("\nStep 2: Analyzed evidence bundles")
        logger.info(f"  Total incidents: {analysis.total_incidents}")
        logger.info(f"  Reconciliations triggered: {analysis.reconciliations_triggered}")
        logger.info(f"  High performers (>80%): {analysis.high_performers}")
        logger.info(f"  Low performers (<50%): {analysis.low_performers}")

        # Show per-agent metrics
        logger.info("\n  Per-agent metrics:")
        for agent_type, metrics in sorted(analysis.agent_metrics.items()):
            logger.info(
                f"    {agent_type}: {metrics.incidents_successful}/{metrics.incidents_selected} success, "
                f"estimate_accuracy={metrics.estimate_accuracy:.2f}"
            )

        # Step 3: Tune weights with significance testing
        current_weights = ScoringWeights()
        new_weights, significance = WeightTuner.tune(analysis, current_weights, aggressiveness=0.1, alpha=0.05)
        adjustment_reason = WeightTuner.compute_adjustment_reason(analysis, significance)
        logger.info("\nStep 3: Tuned weights (with significance testing)")
        logger.info(f"  Adjustment reason: {adjustment_reason}")
        logger.info(f"  Significance results: {significance}")
        logger.info("\n  Weight changes:")
        logger.info(f"    confidence:        {current_weights.confidence:.3f} → {new_weights.confidence:.3f}")
        logger.info(
            f"    business_value:    {current_weights.business_value:.3f} → {new_weights.business_value:.3f}"
        )
        logger.info(
            f"    historical_success: {current_weights.historical_success:.3f} → {new_weights.historical_success:.3f}"
        )

        # Step 4: Save version
        vc = WeightVersionControl(versions_dir)
        version = vc.save_version(
            new_weights.to_dict(),
            adjustment_reason,
            validation_stats={
                "high_performers": len(analysis.high_performers),
                "low_performers": len(analysis.low_performers),
                "reconciliations": analysis.reconciliations_triggered,
            },
        )
        logger.info(f"\nStep 4: Saved weight version {version.version}")
        logger.info(f"  Timestamp: {version.timestamp}")

        # Step 5: Show version history
        all_versions = vc.list_versions()
        logger.info(f"\nStep 5: Version history ({len(all_versions)} version(s))")
        for v in all_versions:
            logger.info(f"  v{v.version}: {v.reason} @ {v.timestamp}")

    logger.info("\n" + "=" * 80)
    logger.info("META-HARNESS COMPLETE: Ready for next iteration")
    logger.info("Next iteration: collect more evidence → analyze → tune → save version")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_demo()
