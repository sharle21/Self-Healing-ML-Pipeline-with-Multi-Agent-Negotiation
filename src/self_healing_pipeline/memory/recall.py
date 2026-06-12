from __future__ import annotations

from typing import Any, Protocol


class MemoryRecall(Protocol):
    def recall(self, tenant_id: str, incident_type: str) -> dict[str, Any]: ...


class NullMemory:
    def recall(self, tenant_id: str, incident_type: str) -> dict[str, Any]:
        return {"cold_start": True, "agents": {}}
