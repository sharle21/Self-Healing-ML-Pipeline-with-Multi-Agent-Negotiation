from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from self_healing_pipeline.commander.commander import CommanderResult
from self_healing_pipeline.gateway.events import Incident


class BundleWriter:
    """Writes Tier 3 evidence bundles to filesystem."""

    def __init__(self, traces_dir: Path | str = "traces") -> None:
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    def write(self, incident_id: str, bundle: dict[str, Any]) -> Path:
        """Write raw bundle (for flexibility)."""
        run_dir = self.traces_dir / f"run_{incident_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "evidence_bundle.json"
        path.write_text(json.dumps(bundle, indent=2, default=str, sort_keys=True))
        return path

    def write_commander_result(
        self,
        incident: Incident,
        commander_result: CommanderResult,
    ) -> Path:
        """Write evidence bundle from CommanderResult."""

        bundle = {
            "incident": {
                "id": incident.id,
                "tenant_id": incident.tenant_id,
                "type": incident.type.value,
                "severity": incident.severity,
                "detected_at": incident.created_at.isoformat(),
                "affected_features": list(incident.affected_features),
            },
            "all_proposals": commander_result.all_proposals,
            "scoring_breakdown": commander_result.scoring_breakdown,
            "winner": {
                "agent_type": commander_result.winning_agent_type,
                "score": commander_result.winning_proposal.get("score", 0.0),
            },
            "execution_result": commander_result.execution_result,
            "reconciliation_triggered": commander_result.reconciliation_triggered,
            "fallback_used": commander_result.fallback_used,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        return self.write(incident.id, bundle)

    def read(self, incident_id: str) -> dict[str, Any] | None:
        """Read evidence bundle for incident."""
        bundle_path = self.traces_dir / f"run_{incident_id}" / "evidence_bundle.json"
        if not bundle_path.exists():
            return None
        return json.loads(bundle_path.read_text())
