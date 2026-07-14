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


class TenantConfig(Base):
    __tablename__ = "tenant_config"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, default="v1")
    latency_sla: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    accuracy_target: Mapped[float] = mapped_column(Float, nullable=False, default=0.75)
    cost_budget: Mapped[float] = mapped_column(Float, nullable=False, default=0.10)
    last_training_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
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
