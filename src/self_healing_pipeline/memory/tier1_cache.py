"""Tier 1: Hot cache (in-memory dict, microsecond access)."""

from dataclasses import dataclass


@dataclass(slots=True)
class CacheEntry:
    """Single cache entry for (tenant_id, incident_type)."""

    last_winner: str | None
    success_rate: float
    recent_success_rate: float
    avg_time: float


class Tier1Cache:
    """In-memory hot cache, invalidated after each resolution."""

    def __init__(self) -> None:
        self.cache: dict[tuple[str, str], CacheEntry] = {}

    def get(self, tenant_id: str, incident_type: str) -> CacheEntry | None:
        """Get cache entry if exists."""
        key = (tenant_id, incident_type)
        return self.cache.get(key)

    def set(self, tenant_id: str, incident_type: str, entry: CacheEntry) -> None:
        """Set cache entry."""
        key = (tenant_id, incident_type)
        self.cache[key] = entry

    def invalidate(self, tenant_id: str, incident_type: str) -> None:
        """Invalidate cache entry (called after resolution)."""
        key = (tenant_id, incident_type)
        self.cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()

    def size(self) -> int:
        """Current cache size."""
        return len(self.cache)
