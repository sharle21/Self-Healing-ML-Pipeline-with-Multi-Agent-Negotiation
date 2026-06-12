from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BundleWriter:
    def __init__(self, traces_dir: Path | str) -> None:
        self.traces_dir = Path(traces_dir)

    def write(self, incident_id: str, bundle: dict[str, Any]) -> Path:
        run_dir = self.traces_dir / f"run_{incident_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "evidence_bundle.json"
        path.write_text(json.dumps(bundle, indent=2, default=str, sort_keys=True))
        return path
