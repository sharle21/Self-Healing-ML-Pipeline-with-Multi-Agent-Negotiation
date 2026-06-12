from datetime import UTC, datetime

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from self_healing_pipeline.db import AgentSummary, Base, DecisionOutcome


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine, future=True)


def test_decision_outcome_roundtrip() -> None:
    session = _make_session()
    row = DecisionOutcome(
        tenant_id="tenant_a",
        incident_type="DRIFT",
        agent_type="retrain",
        success=True,
        business_savings=1234.5,
        duration=45.2,
    )
    session.add(row)
    session.commit()

    fetched = session.scalars(select(DecisionOutcome)).one()
    assert fetched.id is not None
    assert fetched.tenant_id == "tenant_a"
    assert fetched.success is True
    assert fetched.business_savings == 1234.5
    assert fetched.created_at is not None
    delta = abs((datetime.now(UTC).replace(tzinfo=None) - fetched.created_at).total_seconds())
    assert delta < 5


def test_decision_outcomes_composite_index_present() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    indexes = {idx["name"] for idx in inspect(engine).get_indexes("decision_outcomes")}
    assert "idx_decision_outcomes_tenant_type_ts" in indexes


def test_agent_summary_json_roundtrip() -> None:
    session = _make_session()
    summary = AgentSummary(
        tenant_id="tenant_a",
        incident_type="DRIFT",
        agent_type="retrain",
        attempts=3,
        successes=2,
        sum_business_savings=900.0,
        sum_duration=120.0,
        recent_outcomes=[True, False, True],
    )
    session.add(summary)
    session.commit()

    fetched = session.scalars(select(AgentSummary)).one()
    assert fetched.recent_outcomes == [True, False, True]
    assert isinstance(fetched.recent_outcomes, list)


def test_agent_summary_composite_primary_key() -> None:
    session = _make_session()
    pk = {"tenant_id": "tenant_a", "incident_type": "DRIFT", "agent_type": "retrain"}
    session.add(
        AgentSummary(
            **pk,
            attempts=1,
            successes=1,
            sum_business_savings=100.0,
            sum_duration=10.0,
            recent_outcomes=[True],
        )
    )
    session.commit()

    stmt = (
        sqlite_insert(AgentSummary)
        .values(
            **pk,
            attempts=5,
            successes=3,
            sum_business_savings=500.0,
            sum_duration=80.0,
            recent_outcomes=[True, True, False, True, True],
            updated_at=datetime.now(UTC),
        )
        .on_conflict_do_update(
            index_elements=["tenant_id", "incident_type", "agent_type"],
            set_={
                "attempts": 5,
                "successes": 3,
                "sum_business_savings": 500.0,
                "sum_duration": 80.0,
                "recent_outcomes": [True, True, False, True, True],
            },
        )
    )
    session.execute(stmt)
    session.commit()

    rows = session.scalars(select(AgentSummary)).all()
    assert len(rows) == 1
    assert rows[0].attempts == 5
    assert rows[0].recent_outcomes == [True, True, False, True, True]
