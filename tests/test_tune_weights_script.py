"""scripts/tune_weights.py: real evidence -> tuned weights -> live TenantTierConfig."""

import json

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from self_healing_pipeline.db.models import Base, IncidentHistory, TenantTierConfig

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from tune_weights import run_tune_cycle  # noqa: E402


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine, future=True)


def _write_bundle(traces_dir: Path, incident_id: str, winner: str, success: bool) -> None:
    bundle_dir = traces_dir / f"run_{incident_id}"
    bundle_dir.mkdir(parents=True)
    bundle = {
        "incident": {"id": incident_id, "type": "drift", "severity": 0.5},
        "all_proposals": [{"agent_type": winner}],
        "winner": {"agent_type": winner},
        "execution_result": {"success": success},
        "reconciliation": None,
    }
    (bundle_dir / "evidence_bundle.json").write_text(json.dumps(bundle))


def test_no_incidents_is_a_noop(tmp_path):
    session = _make_session()
    summary = run_tune_cycle(
        tmp_path / "traces", session, versions_dir=tmp_path / "versions"
    )
    assert summary == {"total_incidents": 0, "tenants_updated": []}
    assert session.query(TenantTierConfig).count() == 0


def test_tune_cycle_applies_weights_to_tenants_with_incidents(tmp_path):
    session = _make_session()
    traces_dir = tmp_path / "traces"

    for i in range(6):
        _write_bundle(traces_dir, f"inc-{i}", winner="retrain", success=True)

    session.add(IncidentHistory(incident_id="a", tenant_id="enterprise", type="drift", severity=0.5))
    session.add(IncidentHistory(incident_id="b", tenant_id="standard", type="drift", severity=0.3))
    session.commit()

    summary = run_tune_cycle(
        traces_dir, session, versions_dir=tmp_path / "versions", aggressiveness=0.1, alpha=0.05
    )

    assert summary["total_incidents"] == 6
    assert set(summary["tenants_updated"]) == {"enterprise", "standard"}

    rows = {r.tenant_id: r for r in session.query(TenantTierConfig).all()}
    assert set(rows) == {"enterprise", "standard"}
    assert rows["enterprise"].business_value_weight == summary["new_weights"]["business_value"]

    versions = list((tmp_path / "versions").glob("weights_v*.json"))
    assert len(versions) == 1


def test_dry_run_leaves_db_and_versions_untouched(tmp_path):
    session = _make_session()
    traces_dir = tmp_path / "traces"

    for i in range(6):
        _write_bundle(traces_dir, f"inc-{i}", winner="retrain", success=True)
    session.add(IncidentHistory(incident_id="a", tenant_id="enterprise", type="drift", severity=0.5))
    session.commit()

    summary = run_tune_cycle(
        traces_dir, session, versions_dir=tmp_path / "versions", dry_run=True
    )

    assert summary["total_incidents"] == 6
    assert summary["tenants_updated"] == []
    assert session.query(TenantTierConfig).count() == 0
    assert list((tmp_path / "versions").glob("weights_v*.json")) == []
