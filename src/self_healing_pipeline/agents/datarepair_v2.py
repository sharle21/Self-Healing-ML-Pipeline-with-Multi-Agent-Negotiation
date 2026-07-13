"""Data Repair Remediation Policy: fix data quality issues at source."""

from __future__ import annotations

from typing import Any

from self_healing_pipeline.agents.remediation_policy import RemediationPlan, RemediationPolicyAgent


class DataRepairAgent(RemediationPolicyAgent):
    """Fix data quality issues upstream (nulls, duplicates, schema violations).

    Expensive but durable fix. Prevents recurrence of quality-driven incidents.

    Cares about: missing rate severity, root cause certainty, backup data availability, historical success.

    Confidence = 0.35*missing_severity + 0.30*root_cause_certainty + 0.20*backup_available + 0.15*historical_success
    """

    agent_type = "data_repair"

    def can_handle(self, state: dict[str, Any]) -> bool:
        """Can handle if significant data quality issues."""
        missing_rate = state.get("missing_rate", 0.08)
        duplicates = state.get("duplicate_rate", 0.02)
        schema_errors = state.get("schema_error_count", 0)

        return missing_rate > 0.15 or duplicates > 0.05 or schema_errors > 10

    async def analyze(self, state: dict[str, Any]) -> RemediationPlan:
        """Analyze state and propose data repair.

        Args:
            state: DataRepairAgentState dict

        Returns:
            RemediationPlan with state-based confidence
        """
        missing_rate = state.get("missing_rate", 0.08)
        duplicate_rate = state.get("duplicate_rate", 0.02)
        schema_errors = state.get("schema_error_count", 0)
        affected_features = state.get("affected_features", [])
        backup_available = state.get("available_backup_data", True)
        pipeline_health = state.get("data_pipeline_health", 0.55)
        historical_success = state.get("historical_repair_success", 0.70)

        # Compute confidence from state
        missing_severity = min(missing_rate / 0.50, 1.0)  # 50% missing = catastrophic
        schema_severity = min(schema_errors / 100, 1.0)
        root_cause_certainty = backup_available * pipeline_health  # Backup + health = confidence
        data_issue_severity = max(missing_severity, schema_severity)

        state_features = {
            "missing_severity": missing_severity,
            "schema_severity": schema_severity,
            "root_cause_certainty": root_cause_certainty,
            "backup_availability": float(backup_available),
            "pipeline_repairability": pipeline_health,
            "historical_success": historical_success,
        }

        weights = {
            "missing_severity": 0.25,
            "schema_severity": 0.10,
            "root_cause_certainty": 0.30,
            "backup_availability": 0.15,
            "pipeline_repairability": 0.10,
            "historical_success": 0.10,
        }

        confidence = self._compute_confidence_from_state(state_features, weights)

        return RemediationPlan(
            agent_type=self.agent_type,
            action="repair_data_quality",
            confidence=confidence,
            expected_effect={
                "missing_rate_reduction": -missing_rate * 0.9,  # Remove 90% of nulls
                "duplicate_removal": -duplicate_rate,
                "schema_fix": -schema_errors,
                "future_incident_prevention": True,
            },
            reasoning=(
                f"Data quality degradation (missing={missing_rate:.2f}, duplicates={duplicate_rate:.2f}, "
                f"schema_errors={schema_errors}) affecting features {affected_features} → "
                f"repair at source (backup_available={backup_available}, pipeline_health={pipeline_health:.2f})"
            ),
            cost="$100",
            execution_time="300 seconds",
            risk=0.20,  # Risk: repair process might affect other data
        )

    async def execute(self, plan: RemediationPlan) -> Any:
        """Execute data repair (simulated)."""
        import asyncio

        await asyncio.sleep(0.01)

        from self_healing_pipeline.agents.remediation_policy import ExecutionResult

        return ExecutionResult(
            success=True,
            actual_improvement={
                "missing_rate_reduction": -0.07,
                "duplicate_removal": -0.02,
                "schema_fix": -15,
                "future_incident_prevention": True,
            },
            duration=300.0,
            logs=[f"data repair completed: {plan.reasoning}"],
        )
