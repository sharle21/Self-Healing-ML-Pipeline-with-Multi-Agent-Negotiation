"""Tests for evidence bundle analyzer (meta-harness)."""

import json
import tempfile
from pathlib import Path

import pytest

from self_healing_pipeline.meta_harness.analyzer import (
    AgentMetrics,
    EvidenceBundleAnalyzer,
)


@pytest.fixture
def traces_dir():
    """Create temporary traces directory with sample bundles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traces_path = Path(tmpdir)
        yield traces_path


def create_bundle(
    agent_type: str,
    success: bool,
    estimated_savings: float,
    actual_savings: float,
    reconciliation_winner: str | None = None,
) -> dict:
    """Helper to create an evidence bundle."""
    return {
        "incident": {"id": "inc-1", "tenant_id": "standard", "type": "DRIFT"},
        "all_proposals": [
            {
                "agent_type": agent_type,
                "confidence": 0.8,
                "estimated_business_savings": estimated_savings,
                "estimated_risk": 0.1,
            }
        ],
        "winner": {"agent_type": agent_type, "score": 0.8},
        "execution_result": {
            "success": success,
            "actual_business_savings": actual_savings,
            "duration": 10.0,
        },
        "reconciliation": (
            {"winner_type": reconciliation_winner} if reconciliation_winner else None
        ),
    }


class TestAgentMetrics:
    """AgentMetrics tests."""

    def test_success_rate(self):
        """Test success rate calculation."""
        metrics = AgentMetrics(
            agent_type="threshold",
            incidents_selected=5,
            incidents_successful=4,
            total_estimated_savings=5000.0,
            total_actual_savings=4500.0,
            total_estimated_risk=0.5,
            total_actual_risk=0.4,
            reconciliations_won=0,
        )
        assert metrics.success_rate == pytest.approx(0.8)

    def test_success_rate_zero(self):
        """Test success rate when no incidents selected."""
        metrics = AgentMetrics(
            agent_type="retrain",
            incidents_selected=0,
            incidents_successful=0,
            total_estimated_savings=0.0,
            total_actual_savings=0.0,
            total_estimated_risk=0.0,
            total_actual_risk=0.0,
            reconciliations_won=0,
        )
        assert metrics.success_rate == 0.0

    def test_estimate_accuracy_perfect(self):
        """Test estimate accuracy when actual equals estimated."""
        metrics = AgentMetrics(
            agent_type="threshold",
            incidents_selected=2,
            incidents_successful=2,
            total_estimated_savings=2000.0,
            total_actual_savings=2000.0,
            total_estimated_risk=0.2,
            total_actual_risk=0.2,
            reconciliations_won=0,
        )
        assert metrics.estimate_accuracy == pytest.approx(1.0)

    def test_estimate_accuracy_underestimate(self):
        """Test accuracy when actual > estimated."""
        metrics = AgentMetrics(
            agent_type="threshold",
            incidents_selected=1,
            incidents_successful=1,
            total_estimated_savings=1000.0,
            total_actual_savings=1200.0,
            total_estimated_risk=0.1,
            total_actual_risk=0.15,
            reconciliations_won=0,
        )
        # 1200 / 1000 = 1.2, clamped to 1.0
        assert metrics.estimate_accuracy == pytest.approx(1.0)

    def test_estimate_accuracy_overestimate(self):
        """Test accuracy when actual < estimated."""
        metrics = AgentMetrics(
            agent_type="threshold",
            incidents_selected=1,
            incidents_successful=1,
            total_estimated_savings=1000.0,
            total_actual_savings=500.0,
            total_estimated_risk=0.1,
            total_actual_risk=0.15,
            reconciliations_won=0,
        )
        # 500 / 1000 = 0.5
        assert metrics.estimate_accuracy == pytest.approx(0.5)


class TestEvidenceBundleAnalyzer:
    """Evidence bundle analyzer tests."""

    def test_analyze_empty_traces_dir(self, traces_dir):
        """Test analyzing empty traces directory."""
        analyzer = EvidenceBundleAnalyzer(traces_dir)
        result = analyzer.analyze()
        assert result.total_incidents == 0
        assert result.agent_metrics == {}

    def test_analyze_single_bundle(self, traces_dir):
        """Test analyzing single evidence bundle."""
        bundle_dir = traces_dir / "inc-1"
        bundle_dir.mkdir()
        bundle = create_bundle(
            "threshold", success=True, estimated_savings=1000.0, actual_savings=950.0
        )
        with open(bundle_dir / "evidence_bundle.json", "w") as f:
            json.dump(bundle, f)

        analyzer = EvidenceBundleAnalyzer(traces_dir)
        result = analyzer.analyze()

        assert result.total_incidents == 1
        assert "threshold" in result.agent_metrics
        assert result.agent_metrics["threshold"].success_rate == 1.0

    def test_analyze_multiple_bundles(self, traces_dir):
        """Test analyzing multiple bundles."""
        for i, (agent, success) in enumerate(
            [("threshold", True), ("retrain", False), ("rollback", True)]
        ):
            bundle_dir = traces_dir / f"inc-{i}"
            bundle_dir.mkdir()
            bundle = create_bundle(agent, success, 1000.0, 900.0)
            with open(bundle_dir / "evidence_bundle.json", "w") as f:
                json.dump(bundle, f)

        analyzer = EvidenceBundleAnalyzer(traces_dir)
        result = analyzer.analyze()

        assert result.total_incidents == 3
        assert "threshold" in result.agent_metrics
        assert "retrain" in result.agent_metrics
        assert "rollback" in result.agent_metrics
        assert result.agent_metrics["threshold"].success_rate == 1.0
        assert result.agent_metrics["retrain"].success_rate == 0.0
        assert result.agent_metrics["rollback"].success_rate == 1.0

    def test_high_performers(self, traces_dir):
        """Test identifying high performers (>80% success)."""
        for i in range(5):
            bundle_dir = traces_dir / f"inc-{i}"
            bundle_dir.mkdir()
            # threshold succeeds 5/5 times = 100% (>80%)
            bundle = create_bundle("threshold", True, 1000.0, 900.0)
            with open(bundle_dir / "evidence_bundle.json", "w") as f:
                json.dump(bundle, f)

        analyzer = EvidenceBundleAnalyzer(traces_dir)
        result = analyzer.analyze()

        assert "threshold" in result.high_performers

    def test_low_performers(self, traces_dir):
        """Test identifying low performers (<50% success)."""
        for i in range(5):
            bundle_dir = traces_dir / f"inc-{i}"
            bundle_dir.mkdir()
            # retrain succeeds 1/5 times = 20%
            success = i == 0
            bundle = create_bundle("retrain", success, 1000.0, 900.0)
            with open(bundle_dir / "evidence_bundle.json", "w") as f:
                json.dump(bundle, f)

        analyzer = EvidenceBundleAnalyzer(traces_dir)
        result = analyzer.analyze()

        assert "retrain" in result.low_performers

    def test_reconciliation_counting(self, traces_dir):
        """Test counting reconciliation triggers and wins."""
        # Bundle 1: reconciliation triggered, retrain won
        bundle_dir = traces_dir / "inc-1"
        bundle_dir.mkdir()
        bundle = create_bundle(
            "retrain",
            success=True,
            estimated_savings=1000.0,
            actual_savings=1000.0,
            reconciliation_winner="retrain",
        )
        with open(bundle_dir / "evidence_bundle.json", "w") as f:
            json.dump(bundle, f)

        analyzer = EvidenceBundleAnalyzer(traces_dir)
        result = analyzer.analyze()

        assert result.reconciliations_triggered == 1
        assert result.agent_metrics["retrain"].reconciliations_won == 1

    def test_malformed_bundle_ignored(self, traces_dir):
        """Test that malformed bundles are ignored."""
        bundle_dir = traces_dir / "inc-1"
        bundle_dir.mkdir()
        with open(bundle_dir / "evidence_bundle.json", "w") as f:
            f.write("invalid json {")

        analyzer = EvidenceBundleAnalyzer(traces_dir)
        result = analyzer.analyze()

        assert result.total_incidents == 0
