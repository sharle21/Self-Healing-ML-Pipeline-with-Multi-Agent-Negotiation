from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class IncidentType(StrEnum):
    DRIFT = "drift"
    DATA_QUALITY = "data_quality"
    COST_THRESHOLD = "cost_threshold"
    LATENCY_BREACH = "latency_breach"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True, slots=True)
class Incident:
    tenant_id: str
    type: IncidentType
    payload: dict[str, Any]
    severity: float = 0.0
    affected_features: tuple[str, ...] = ()
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utcnow)

    def fingerprint(self) -> str:
        canonical = json.dumps(self.payload, sort_keys=True, default=str, separators=(",", ":"))
        material = f"{self.tenant_id}|{self.type.value}|{canonical}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()
