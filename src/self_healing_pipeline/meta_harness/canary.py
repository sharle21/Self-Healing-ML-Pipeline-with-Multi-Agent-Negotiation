"""Canary weight management: gradual rollout with automatic rollback."""

import hashlib
import logging
from dataclasses import dataclass

from self_healing_pipeline.meta_harness.tuner import ScoringWeights
from self_healing_pipeline.meta_harness.version_control import WeightVersionControl

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CanaryConfig:
    """Canary rollout configuration."""

    stable_version: int
    canary_version: int
    canary_percentage: float  # 0-100
    min_incidents: int = 10  # Minimum before considering rollback
    rollback_threshold: float = 0.95  # If success_rate < this, rollback


class CanaryWeightManager:
    """Manage canary rollout of weight versions."""

    def __init__(self, version_control: WeightVersionControl) -> None:
        self.vc = version_control
        self.canary_config: CanaryConfig | None = None
        self.canary_metrics = {
            "stable_successes": 0,
            "stable_attempts": 0,
            "canary_successes": 0,
            "canary_attempts": 0,
        }

    def select_weight_version(self, incident_id: str) -> ScoringWeights:
        """Select weight version for incident (stable or canary).

        Uses incident_id hash to deterministically route to canary or stable.

        Args:
            incident_id: unique incident identifier
            config: canary configuration

        Returns:
            ScoringWeights to use for this incident
        """
        if not self.canary_config:
            # No canary; use latest
            latest = self.vc.load_latest()
            return ScoringWeights.from_dict(latest.weights) if latest else ScoringWeights()

        # Deterministic hash-based routing
        hash_val = int(hashlib.md5(incident_id.encode()).hexdigest(), 16)
        canary_bucket = (hash_val % 100) < self.canary_config.canary_percentage

        if canary_bucket:
            canary = self.vc.load_version(self.canary_config.canary_version)
            if canary:
                return ScoringWeights.from_dict(canary.weights)

        # Fallback to stable
        stable = self.vc.load_version(self.canary_config.stable_version)
        return ScoringWeights.from_dict(stable.weights) if stable else ScoringWeights()

    def record_outcome(self, incident_id: str, success: bool) -> None:
        """Record incident outcome (used for canary metrics).

        Args:
            incident_id: incident that was executed
            success: whether execution succeeded
        """
        if not self.canary_config:
            return

        hash_val = int(hashlib.md5(incident_id.encode()).hexdigest(), 16)
        canary_bucket = (hash_val % 100) < self.canary_config.canary_percentage

        if canary_bucket:
            self.canary_metrics["canary_attempts"] += 1
            if success:
                self.canary_metrics["canary_successes"] += 1
        else:
            self.canary_metrics["stable_attempts"] += 1
            if success:
                self.canary_metrics["stable_successes"] += 1

    def check_rollback(self) -> bool:
        """Check if canary should be rolled back.

        Returns:
            True if rollback recommended
        """
        if not self.canary_config:
            return False

        if self.canary_metrics["canary_attempts"] < self.canary_config.min_incidents:
            return False

        canary_rate = (
            self.canary_metrics["canary_successes"]
            / self.canary_metrics["canary_attempts"]
        )
        stable_rate = (
            self.canary_metrics["stable_successes"]
            / self.canary_metrics["stable_attempts"]
            if self.canary_metrics["stable_attempts"] > 0
            else 1.0
        )

        # Rollback if canary significantly worse than stable
        if canary_rate < stable_rate * self.canary_config.rollback_threshold:
            logger.warning(
                f"Canary rollback recommended: "
                f"canary={canary_rate:.2%} < stable={stable_rate:.2%} * {self.canary_config.rollback_threshold}"
            )
            return True

        return False

    def promote_canary(self) -> None:
        """Promote canary to stable (100% rollout)."""
        if not self.canary_config:
            return

        logger.info(
            f"Promoting canary v{self.canary_config.canary_version} to stable"
        )
        self.canary_config.stable_version = self.canary_config.canary_version
        self.canary_config.canary_percentage = 0.0
        self.canary_metrics = {
            "stable_successes": 0,
            "stable_attempts": 0,
            "canary_successes": 0,
            "canary_attempts": 0,
        }

    def start_canary(
        self,
        canary_version: int,
        percentage: float,
        min_incidents: int = 10,
    ) -> None:
        """Start canary rollout of a new weight version.

        Args:
            canary_version: version number to test
            percentage: % of traffic to route to canary (0-100)
            min_incidents: min incidents before considering rollback
        """
        stable = self.vc.load_latest()
        stable_version = stable.version if stable else 1

        self.canary_config = CanaryConfig(
            stable_version=stable_version,
            canary_version=canary_version,
            canary_percentage=percentage,
            min_incidents=min_incidents,
        )
        logger.info(
            f"Canary started: v{canary_version} at {percentage}% "
            f"(stable=v{stable_version})"
        )
