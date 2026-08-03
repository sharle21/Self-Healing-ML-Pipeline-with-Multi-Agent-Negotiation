"""CommanderV3 writes evidence bundles the offline meta-harness can analyze."""

import json

import pytest

from self_healing_pipeline.agents.rollback_v2 import RollbackAgent
from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
from self_healing_pipeline.commander.commander_v3 import CommanderV3
from self_healing_pipeline.gateway.events import Incident, IncidentType
from self_healing_pipeline.memory.tier3_traces import BundleWriter
from self_healing_pipeline.meta_harness.analyzer import EvidenceBundleAnalyzer


@pytest.mark.asyncio
async def test_handle_incident_writes_evidence_bundle(tmp_path):
    agents = [ThresholdAdjustmentAgent("threshold-1"), RollbackAgent("rollback-1")]
    commander = CommanderV3(agents, bundle_writer=BundleWriter(tmp_path))

    incident = Incident(
        tenant_id="test-tenant",
        type=IncidentType.DRIFT,
        payload={"accuracy": 0.68},
        severity=0.6,
    )

    result = await commander.handle_incident(incident)

    bundle_path = tmp_path / f"run_{incident.id}" / "evidence_bundle.json"
    assert bundle_path.exists()

    bundle = json.loads(bundle_path.read_text())
    assert bundle["incident"]["id"] == incident.id
    assert bundle["incident"]["tenant_id"] == "test-tenant"
    assert bundle["winner"]["agent_type"] == result.winning_agent_type
    assert bundle["execution_result"]["success"] == result.execution_result["success"]
    assert bundle["all_proposals"]
    assert result.winning_agent_type in {p["agent_type"] for p in bundle["all_proposals"]}


@pytest.mark.asyncio
async def test_handle_incident_skips_bundle_write_by_default(tmp_path, monkeypatch):
    """No bundle_writer passed -> no writes anywhere, including the real traces_dir.

    This is what keeps ordinary unit tests (which mostly don't pass bundle_writer)
    from polluting the project's actual traces/ directory on every run.
    """
    monkeypatch.chdir(tmp_path)
    agents = [ThresholdAdjustmentAgent("threshold-1"), RollbackAgent("rollback-1")]
    commander = CommanderV3(agents)

    incident = Incident(
        tenant_id="test-tenant",
        type=IncidentType.DRIFT,
        payload={"accuracy": 0.68},
        severity=0.6,
    )
    await commander.handle_incident(incident)

    assert not (tmp_path / "traces").exists()


@pytest.mark.asyncio
async def test_written_bundle_is_analyzable(tmp_path):
    agents = [ThresholdAdjustmentAgent("threshold-1"), RollbackAgent("rollback-1")]
    commander = CommanderV3(agents, bundle_writer=BundleWriter(tmp_path))

    for _ in range(3):
        incident = Incident(
            tenant_id="test-tenant",
            type=IncidentType.DRIFT,
            payload={"accuracy": 0.68},
            severity=0.6,
        )
        await commander.handle_incident(incident)

    analysis = EvidenceBundleAnalyzer(tmp_path).analyze()

    assert analysis.total_incidents == 3
    assert sum(m.incidents_selected for m in analysis.agent_metrics.values()) == 3


@pytest.mark.asyncio
async def test_reconciliation_log_round_trips_winner_type(tmp_path):
    """When reconciliation fires, the bundle's reconciliation.winner_type must be
    readable by EvidenceBundleAnalyzer (it increments reconciliations_won on match)."""
    agents = [ThresholdAdjustmentAgent("threshold-1"), RollbackAgent("rollback-1")]
    commander = CommanderV3(agents, bundle_writer=BundleWriter(tmp_path))

    incident = Incident(
        tenant_id="test-tenant",
        type=IncidentType.DRIFT,
        payload={"accuracy": 0.68},
        severity=0.6,
    )
    result = await commander.handle_incident(incident)

    bundle_path = tmp_path / f"run_{incident.id}" / "evidence_bundle.json"
    bundle = json.loads(bundle_path.read_text())

    if result.reconciliation_triggered:
        assert bundle["reconciliation"]["winner_type"] == result.winning_agent_type
    else:
        assert bundle["reconciliation"] is None
