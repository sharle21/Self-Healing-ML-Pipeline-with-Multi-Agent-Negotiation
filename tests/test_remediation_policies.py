"""Tests for remediation policy agents with state-based confidence."""

import pytest

from self_healing_pipeline.agents.datarepair_v2 import DataRepairAgent
from self_healing_pipeline.agents.fallback_v2 import FallbackAgent
from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
from self_healing_pipeline.agents.rollback_v2 import RollbackAgent
from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
from self_healing_pipeline.observability import StateConstructor, TelemetryCollector


class TestThresholdAdjustmentAgent:
    """Test threshold adjustment remediation policy."""

    @pytest.mark.asyncio
    async def test_analyze_recall_degradation(self):
        """Test threshold agent proposes adjustment when recall degrades."""
        agent = ThresholdAdjustmentAgent("threshold-1")

        state = {
            "current_threshold": 0.50,
            "recall_drop": 0.12,
            "cost_false_negative": 500,
            "prediction_distribution_shift": 0.15,
            "historical_threshold_success": 0.75,
        }

        plan = await agent.analyze(state)

        assert plan.agent_type == "threshold"
        assert 0 <= plan.confidence <= 1
        assert plan.action == "change_threshold"
        assert "recall" in plan.reasoning.lower()

    def test_can_handle_high_fn_cost(self):
        """Test agent can handle high false negative cost."""
        agent = ThresholdAdjustmentAgent("threshold-1")

        state = {
            "recall_drop": 0.02,
            "cost_false_negative": 600,
        }

        assert agent.can_handle(state) is True

    @pytest.mark.asyncio
    async def test_execute_returns_result(self):
        """Test execution returns ExecutionResult."""
        agent = ThresholdAdjustmentAgent("threshold-1")

        state = {
            "current_threshold": 0.50,
            "recall_drop": 0.12,
            "cost_false_negative": 500,
            "prediction_distribution_shift": 0.15,
            "historical_threshold_success": 0.75,
        }

        plan = await agent.analyze(state)
        result = await agent.execute(plan)

        assert result.success is True
        assert result.duration > 0


class TestRetrainAgent:
    """Test model retrain remediation policy."""

    @pytest.mark.asyncio
    async def test_analyze_drift_detection(self):
        """Test retrain agent proposes retraining with drift."""
        agent = RetrainAgent("retrain-1")

        state = {
            "drift_score": 1.8,
            "auc_drop": 0.08,
            "data_quality_score": 0.92,
            "model_age_days": 45,
            "historical_retrain_success": 0.72,
            "affected_features": ["income", "age"],
        }

        plan = await agent.analyze(state)

        assert plan.agent_type == "retrain"
        assert plan.action == "retrain_model"
        assert plan.confidence > 0.5  # Should be confident with drift + AUC drop
        assert "drift" in plan.reasoning.lower()

    def test_can_handle_auc_drop(self):
        """Test agent can handle AUC drop."""
        agent = RetrainAgent("retrain-1")

        state = {
            "drift_score": 0.5,
            "auc_drop": 0.07,
        }

        assert agent.can_handle(state) is True

    @pytest.mark.asyncio
    async def test_confidence_increases_with_drift(self):
        """Test confidence is higher with stronger drift signal."""
        agent = RetrainAgent("retrain-1")

        # Low drift
        low_drift_state = {
            "drift_score": 0.5,
            "auc_drop": 0.03,
            "data_quality_score": 0.95,
            "model_age_days": 10,
            "historical_retrain_success": 0.72,
            "affected_features": [],
        }
        low_plan = await agent.analyze(low_drift_state)

        # High drift
        high_drift_state = {
            "drift_score": 2.5,
            "auc_drop": 0.12,
            "data_quality_score": 0.95,
            "model_age_days": 45,
            "historical_retrain_success": 0.72,
            "affected_features": ["income", "age"],
        }
        high_plan = await agent.analyze(high_drift_state)

        assert high_plan.confidence > low_plan.confidence


class TestRollbackAgent:
    """Test model rollback remediation policy."""

    @pytest.mark.asyncio
    async def test_analyze_recent_deployment(self):
        """Test rollback agent proposes rollback for recent bad deployment."""
        agent = RollbackAgent("rollback-1")

        state = {
            "current_model": "v13",
            "previous_model": "v12",
            "deployment_age_hours": 6,
            "current_auc": 0.68,
            "previous_auc": 0.77,
            "current_error_rate": 0.18,
            "previous_error_rate": 0.09,
            "deployment_related_incident_probability": 0.8,
            "historical_rollback_success": 0.91,
        }

        plan = await agent.analyze(state)

        assert plan.agent_type == "rollback"
        assert plan.action == "rollback"
        assert plan.confidence > 0.6  # Should be confident
        assert "v12" in plan.reasoning

    def test_can_handle_recent_regression(self):
        """Test agent can handle recent deployment with regression."""
        agent = RollbackAgent("rollback-1")

        state = {
            "deployment_age_hours": 12,
            "current_auc": 0.65,
            "previous_auc": 0.80,
        }

        assert agent.can_handle(state) is True


class TestFallbackAgent:
    """Test fallback logic remediation policy."""

    @pytest.mark.asyncio
    async def test_analyze_high_error_rate(self):
        """Test fallback agent activates with high error rate."""
        agent = FallbackAgent("fallback-1")

        state = {
            "error_rate": 0.25,
            "latency_p95": 85,
            "prediction_failure_rate": 0.15,
            "confidence_distribution_mean": 0.42,
            "missing_rate": 0.10,
            "acceptable_accuracy_loss": 0.05,
            "fallback_quality": 0.70,
            "historical_fallback_success": 0.85,
        }

        plan = await agent.analyze(state)

        assert plan.agent_type == "fallback"
        assert plan.action == "activate_fallback"
        assert plan.confidence > 0.5

    def test_can_handle_critical_latency(self):
        """Test agent can handle critical latency."""
        agent = FallbackAgent("fallback-1")

        state = {
            "error_rate": 0.10,
            "latency_p95": 600,
            "missing_rate": 0.08,
        }

        assert agent.can_handle(state) is True


class TestDataRepairAgent:
    """Test data repair remediation policy."""

    @pytest.mark.asyncio
    async def test_analyze_missing_data(self):
        """Test data repair agent proposes fix for missing data."""
        agent = DataRepairAgent("datarepair-1")

        state = {
            "missing_rate": 0.25,
            "duplicate_rate": 0.12,
            "schema_error_count": 50,
            "affected_features": ["income", "credit_history"],
            "available_backup_data": True,
            "data_pipeline_health": 0.55,
            "historical_repair_success": 0.70,
        }

        plan = await agent.analyze(state)

        assert plan.agent_type == "data_repair"
        assert plan.action == "repair_data_quality"
        assert "missing" in plan.reasoning.lower()

    def test_can_handle_schema_errors(self):
        """Test agent can handle schema violations."""
        agent = DataRepairAgent("datarepair-1")

        state = {
            "missing_rate": 0.05,
            "duplicate_rate": 0.02,
            "schema_error_count": 50,
        }

        assert agent.can_handle(state) is True


class TestStateBasedConfidence:
    """Test state-based confidence computation."""

    @pytest.mark.asyncio
    async def test_confidence_reflects_state(self):
        """Test that agent confidence reflects state features."""
        agent = RetrainAgent("retrain-1")

        # Healthy state
        healthy_state = {
            "drift_score": 0.2,
            "auc_drop": 0.01,
            "data_quality_score": 0.99,
            "model_age_days": 5,
            "historical_retrain_success": 0.90,
            "affected_features": [],
        }
        healthy_plan = await agent.analyze(healthy_state)

        # Degraded state
        degraded_state = {
            "drift_score": 2.5,
            "auc_drop": 0.15,
            "data_quality_score": 0.80,
            "model_age_days": 60,
            "historical_retrain_success": 0.50,
            "affected_features": ["income", "age", "credit"],
        }
        degraded_plan = await agent.analyze(degraded_state)

        # Degraded state should have higher confidence (more signal)
        assert degraded_plan.confidence > healthy_plan.confidence
