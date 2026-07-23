from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DecisionOutcome(Base):
    __tablename__ = "decision_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    incident_type: Mapped[str] = mapped_column(String(64), nullable=False)
    agent_type: Mapped[str] = mapped_column(String(64), nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    business_savings: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    duration: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    reward: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    incident_resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        Index(
            "idx_decision_outcomes_tenant_type_ts",
            "tenant_id",
            "incident_type",
            "created_at",
        ),
    )


class AgentSummary(Base):
    __tablename__ = "agent_summary"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    incident_type: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_type: Mapped[str] = mapped_column(String(64), primary_key=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    successes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sum_business_savings: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sum_duration: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    recent_outcomes: Mapped[list[bool]] = mapped_column(JSON, nullable=False, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )


class IncidentDedup(Base):
    __tablename__ = "incident_dedup"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    incident_type: Mapped[str] = mapped_column(String(64), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), primary_key=True)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class TenantPolicy(Base):
    __tablename__ = "tenant_policy"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Governance decisions (operator-defined)
    min_acceptable_auc: Mapped[float] = mapped_column(Float, nullable=False, default=0.75)
    max_acceptable_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    max_acceptable_missing_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    daily_cost_budget: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    latency_sla_ms: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    risk_tolerance: Mapped[str] = mapped_column(String(32), nullable=False, default="standard")

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )


class ModelValidationReport(Base):
    __tablename__ = "model_validation_report"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # Validation metrics (immutable, from training)
    auc: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False)
    optimal_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    calibration_error: Mapped[float] = mapped_column(Float, nullable=False)

    validated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        Index("idx_model_validation_tenant_version", "tenant_id", "model_version"),
    )


class RuntimeDeploymentProfile(Base):
    __tablename__ = "runtime_deployment_profile"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)

    # Runtime characteristics (continuously updated from monitoring)
    latency_p95_ms: Mapped[float] = mapped_column(Float, nullable=False)
    latency_p99_ms: Mapped[float] = mapped_column(Float, nullable=False)
    throughput_rps: Mapped[float] = mapped_column(Float, nullable=False)
    memory_mb: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cpu_percent: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    deployment_region: Mapped[str] = mapped_column(String(64), nullable=False, default="us-east-1")

    measured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_runtime_profile_tenant_ts", "tenant_id", "updated_at"),
    )


class TenantTierConfig(Base):
    __tablename__ = "tenant_tier_config"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tier: Mapped[str] = mapped_column(String(32), nullable=False, default="standard")

    # Agent eligibility
    threshold_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    retrain_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    rollback_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    fallback_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    datarepair_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Weight overrides
    business_value_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.30)
    confidence_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.20)
    risk_inverse_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.20)
    cost_efficiency_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.10)
    time_inverse_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    historical_success_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.15)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )


class IncidentHistory(Base):
    __tablename__ = "incident_history"

    incident_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        Index("idx_incident_history_tenant_ts", "tenant_id", "timestamp"),
    )


class RemediationAction(Base):
    __tablename__ = "remediation_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(String(64), nullable=False)
    agent: Mapped[str] = mapped_column(String(64), nullable=False)
    proposal: Mapped[dict] = mapped_column(JSON, nullable=False)
    chosen: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reward: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        Index("idx_remediation_actions_incident", "incident_id"),
        Index("idx_remediation_actions_agent_ts", "agent", "created_at"),
    )


class AgentWeights(Base):
    __tablename__ = "agent_weights"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    incident_type: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_type: Mapped[str] = mapped_column(String(64), primary_key=True)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        Index("idx_agent_weights_tenant_version", "tenant_id", "version"),
    )


class PolicyWeights(Base):
    __tablename__ = "policy_weights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.35)
    cost_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.25)
    risk_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.20)
    latency_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.20)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        Index("idx_policy_weights_version", "version"),
    )


class TenantThresholdOverride(Base):
    """Live decision threshold written by ThresholdAgent; read by ModelServer on each predict."""

    __tablename__ = "tenant_threshold_overrides"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    updated_by: Mapped[str] = mapped_column(String(64), nullable=False, default="system")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )


class ModelRegistry(Base):
    """Tracks trained model versions for retrain + rollback."""

    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(256), nullable=False)
    backup_path: Mapped[str] = mapped_column(String(256), nullable=True)
    overall_auc: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        Index("idx_model_registry_version", "model_version"),
        Index("idx_model_registry_status_ts", "status", "created_at"),
    )
