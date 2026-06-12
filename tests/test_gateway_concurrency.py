from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from self_healing_pipeline.db.models import Base, IncidentDedup
from self_healing_pipeline.gateway import EventGateway, Incident, IncidentType


@pytest.fixture
def session_factory() -> Callable[[], Session]:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def _incident(tenant: str, payload: dict[str, object] | None = None) -> Incident:
    return Incident(tenant_id=tenant, type=IncidentType.DRIFT, payload=payload or {})


async def test_per_tenant_serialization(session_factory: Callable[[], Session]) -> None:
    gateway = EventGateway(session_factory)
    timeline: list[tuple[str, str]] = []

    async def handler(incident: Incident) -> None:
        timeline.append(("start", incident.id))
        await asyncio.sleep(0.05)
        timeline.append(("end", incident.id))

    a = _incident("tenant_a", {"n": 1})
    b = _incident("tenant_a", {"n": 2})
    await asyncio.gather(gateway.dispatch(a, handler), gateway.dispatch(b, handler))

    # per-tenant lock means sequential: start1 end1 start2 end2 (order of a/b may vary)
    starts = [i for i, (k, _) in enumerate(timeline) if k == "start"]
    ends = [i for i, (k, _) in enumerate(timeline) if k == "end"]
    assert starts == [0, 2]
    assert ends == [1, 3]


async def test_global_concurrency_cap(session_factory: Callable[[], Session]) -> None:
    gateway = EventGateway(session_factory, global_concurrency=3)
    active = 0
    peak = 0

    async def handler(_: Incident) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1

    incidents = [_incident(f"tenant_{i}") for i in range(6)]
    await asyncio.gather(*(gateway.dispatch(inc, handler) for inc in incidents))
    assert peak == 3


async def test_submit_dedups_within_window(session_factory: Callable[[], Session]) -> None:
    gateway = EventGateway(session_factory, dedup_window_seconds=60.0)
    inc = _incident("tenant_a", {"feature": "AGE", "severity": 0.8})

    assert await gateway.submit(inc) is True
    duplicate = Incident(
        tenant_id=inc.tenant_id, type=inc.type, payload=inc.payload
    )  # same content, new id/timestamp
    assert await gateway.submit(duplicate) is False


async def test_submit_accepts_after_window(session_factory: Callable[[], Session]) -> None:
    gateway = EventGateway(session_factory, dedup_window_seconds=0.0)
    inc = _incident("tenant_a", {"feature": "AGE"})
    assert await gateway.submit(inc) is True
    assert await gateway.submit(_incident("tenant_a", {"feature": "AGE"})) is True


async def test_submit_distinct_payload_not_deduped(session_factory: Callable[[], Session]) -> None:
    gateway = EventGateway(session_factory, dedup_window_seconds=60.0)
    assert await gateway.submit(_incident("tenant_a", {"feature": "AGE"})) is True
    assert await gateway.submit(_incident("tenant_a", {"feature": "LIMIT_BAL"})) is True


async def test_submit_persists_dedup_row(session_factory: Callable[[], Session]) -> None:
    gateway = EventGateway(session_factory)
    inc = _incident("tenant_a", {"feature": "AGE"})
    await gateway.submit(inc)
    with session_factory() as s:
        rows = s.scalars(select(IncidentDedup)).all()
    assert len(rows) == 1
    assert rows[0].tenant_id == "tenant_a"
    assert rows[0].incident_type == IncidentType.DRIFT.value
    assert rows[0].fingerprint == inc.fingerprint()
