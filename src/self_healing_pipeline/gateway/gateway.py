from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy.orm import Session

from self_healing_pipeline.db.models import IncidentDedup
from self_healing_pipeline.gateway.events import Incident

IncidentHandler = Callable[[Incident], Awaitable[None]]


class SessionFactory(Protocol):
    def __call__(self) -> Session: ...


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


class EventGateway:
    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        global_concurrency: int = 3,
        dedup_window_seconds: float = 60.0,
    ) -> None:
        if global_concurrency < 1:
            raise ValueError(f"global_concurrency must be >= 1, got {global_concurrency}")
        if dedup_window_seconds < 0:
            raise ValueError(f"dedup_window_seconds must be >= 0, got {dedup_window_seconds}")
        self._session_factory = session_factory
        self._sem = asyncio.Semaphore(global_concurrency)
        self._tenant_locks: dict[str, asyncio.Lock] = {}
        self.dedup_window_seconds = dedup_window_seconds

    def _lock_for(self, tenant_id: str) -> asyncio.Lock:
        lock = self._tenant_locks.get(tenant_id)
        if lock is None:
            lock = asyncio.Lock()
            self._tenant_locks[tenant_id] = lock
        return lock

    async def submit(self, incident: Incident) -> bool:
        fp = incident.fingerprint()
        now = datetime.now(UTC)
        return await asyncio.to_thread(self._dedup_upsert, incident, fp, now)

    def _dedup_upsert(self, incident: Incident, fp: str, now: datetime) -> bool:
        with self._session_factory() as session:
            existing = session.get(
                IncidentDedup, (incident.tenant_id, incident.type.value, fp)
            )
            if existing is not None:
                age = (now - _aware(existing.last_seen_at)).total_seconds()
                if age < self.dedup_window_seconds:
                    return False
                existing.last_seen_at = now
            else:
                session.add(
                    IncidentDedup(
                        tenant_id=incident.tenant_id,
                        incident_type=incident.type.value,
                        fingerprint=fp,
                        last_seen_at=now,
                    )
                )
            session.commit()
            return True

    async def dispatch(self, incident: Incident, handler: IncidentHandler) -> None:
        async with self._sem, self._lock_for(incident.tenant_id):
            await handler(incident)
