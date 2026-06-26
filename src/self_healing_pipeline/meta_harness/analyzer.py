"""Meta-harness: analyze evidence bundles to optimize Commander weights.

Offline analysis of incident traces to measure agent performance:
- win rate (% incidents where agent was picked)
- success rate (% where agent's solution worked)
- estimate accuracy (actual vs predicted savings/risk)
- reconciliation impact (when it triggered, who won)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AgentMetrics:
    """Aggregated metrics for one agent."""

    agent_type: str
    incidents_selected: int
    incidents_successful: int
    total_estimated_savings: float
    total_actual_savings: float
    total_estimated_risk: float
    total_actual_risk: float
    reconciliations_won: int

    @property
    def success_rate(self) -> float:
        """Actual success rate."""
        return (
            self.incidents_successful / self.incidents_selected
            if self.incidents_selected > 0
            else 0.0
        )

    @property
    def estimate_accuracy(self) -> float:
        """How close were estimates to actuals (0-1, 1=perfect)."""
        if self.incidents_selected == 0:
            return 1.0
        # Simple metric: (actual / estimated) clamped to [0, 1]
        # If estimate was 1000 and actual was 900: 900/1000 = 0.9
        # If estimate was 1000 and actual was 1200: 1200/1000 = 1.2 → clamp to 1.0
        avg_ratio = self.total_actual_savings / max(self.total_estimated_savings, 1.0)
        return min(avg_ratio, 1.0)


@dataclass(slots=True)
class AnalysisResult:
    """Result of analyzing evidence bundles."""

    total_incidents: int
    agent_metrics: dict[str, AgentMetrics]
    reconciliations_triggered: int
    high_performers: list[str]  # agents with success_rate > 0.8
    low_performers: list[str]  # agents with success_rate < 0.5


class EvidenceBundleAnalyzer:
    """Analyze evidence bundles for weight optimization."""

    def __init__(self, traces_dir: Path) -> None:
        """Init analyzer.

        Args:
            traces_dir: where evidence bundles are stored
        """
        self.traces_dir = traces_dir

    def analyze(self) -> AnalysisResult:
        """Analyze all evidence bundles in traces_dir.

        Returns:
            AnalysisResult with per-agent metrics
        """
        agent_metrics: dict[str, AgentMetrics] = {}
        total_incidents = 0
        reconciliations_triggered = 0

        # Find all evidence bundle files
        if not self.traces_dir.exists():
            return AnalysisResult(
                total_incidents=0,
                agent_metrics={},
                reconciliations_triggered=0,
                high_performers=[],
                low_performers=[],
            )

        bundle_files = sorted(self.traces_dir.glob("*/evidence_bundle.json"))

        for bundle_file in bundle_files:
            try:
                with open(bundle_file) as f:
                    bundle = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            total_incidents += 1

            # Extract winner and execution result
            winner_type = bundle.get("winner", {}).get("agent_type")
            execution = bundle.get("execution_result", {})
            reconciliation = bundle.get("reconciliation")

            if winner_type:
                # Initialize metrics if needed
                if winner_type not in agent_metrics:
                    agent_metrics[winner_type] = AgentMetrics(
                        agent_type=winner_type,
                        incidents_selected=0,
                        incidents_successful=0,
                        total_estimated_savings=0.0,
                        total_actual_savings=0.0,
                        total_estimated_risk=0.0,
                        total_actual_risk=0.0,
                        reconciliations_won=0,
                    )

                metrics = agent_metrics[winner_type]
                metrics.incidents_selected += 1

                # Check success
                if execution.get("success", False):
                    metrics.incidents_successful += 1

                # Add savings (estimated from winner proposal)
                all_proposals = bundle.get("all_proposals", [])
                winner_proposal = next(
                    (p for p in all_proposals if p.get("agent_type") == winner_type),
                    {},
                )
                estimated_savings = winner_proposal.get("estimated_business_savings", 0.0)
                actual_savings = execution.get("actual_business_savings", 0.0)
                metrics.total_estimated_savings += estimated_savings
                metrics.total_actual_savings += actual_savings

                # Add risk
                estimated_risk = winner_proposal.get("estimated_risk", 0.0)
                metrics.total_estimated_risk += estimated_risk

                # Count reconciliation wins
                if reconciliation:
                    reconciliations_triggered += 1
                    if reconciliation.get("winner_type") == winner_type:
                        metrics.reconciliations_won += 1

        # Identify high/low performers
        high_performers = [
            agent_type
            for agent_type, metrics in agent_metrics.items()
            if metrics.success_rate > 0.8
        ]
        low_performers = [
            agent_type
            for agent_type, metrics in agent_metrics.items()
            if metrics.success_rate < 0.5
        ]

        return AnalysisResult(
            total_incidents=total_incidents,
            agent_metrics=agent_metrics,
            reconciliations_triggered=reconciliations_triggered,
            high_performers=sorted(high_performers),
            low_performers=sorted(low_performers),
        )
