"""Weight version control: persist and track weight evolution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WeightVersion:
    """Persisted weight version with metadata."""

    version: int
    timestamp: str
    weights: dict[str, float]
    reason: str
    validation_stats: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


class WeightVersionControl:
    """Manage weight versions: persist, load, validate."""

    def __init__(self, versions_dir: Path) -> None:
        """Init version control.

        Args:
            versions_dir: where to store weight versions
        """
        self.versions_dir = versions_dir
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def save_version(
        self,
        weights: dict[str, float],
        reason: str,
        validation_stats: dict[str, Any] | None = None,
    ) -> WeightVersion:
        """Save a weight version.

        Args:
            weights: the weight dict
            reason: why weights changed
            validation_stats: optional validation metrics

        Returns:
            WeightVersion that was saved
        """
        # Find next version number
        existing = list(self.versions_dir.glob("weights_v*.json"))
        next_version = len(existing) + 1

        version = WeightVersion(
            version=next_version,
            timestamp=datetime.now(UTC).isoformat(),
            weights=weights,
            reason=reason,
            validation_stats=validation_stats,
        )

        # Save to file
        version_file = self.versions_dir / f"weights_v{next_version}.json"
        with open(version_file, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

        return version

    def load_latest(self) -> WeightVersion | None:
        """Load latest weight version.

        Returns:
            Latest WeightVersion or None if no versions exist
        """
        existing = sorted(self.versions_dir.glob("weights_v*.json"))
        if not existing:
            return None

        latest_file = existing[-1]
        with open(latest_file) as f:
            data = json.load(f)

        return WeightVersion(
            version=data["version"],
            timestamp=data["timestamp"],
            weights=data["weights"],
            reason=data["reason"],
            validation_stats=data.get("validation_stats"),
        )

    def load_version(self, version: int) -> WeightVersion | None:
        """Load specific weight version.

        Args:
            version: version number to load

        Returns:
            WeightVersion or None if not found
        """
        version_file = self.versions_dir / f"weights_v{version}.json"
        if not version_file.exists():
            return None

        with open(version_file) as f:
            data = json.load(f)

        return WeightVersion(
            version=data["version"],
            timestamp=data["timestamp"],
            weights=data["weights"],
            reason=data["reason"],
            validation_stats=data.get("validation_stats"),
        )

    def list_versions(self) -> list[WeightVersion]:
        """List all saved versions in order.

        Returns:
            List of WeightVersion objects, oldest first
        """
        existing = sorted(self.versions_dir.glob("weights_v*.json"))
        versions = []

        for version_file in existing:
            with open(version_file) as f:
                data = json.load(f)
            versions.append(
                WeightVersion(
                    version=data["version"],
                    timestamp=data["timestamp"],
                    weights=data["weights"],
                    reason=data["reason"],
                    validation_stats=data.get("validation_stats"),
                )
            )

        return versions
