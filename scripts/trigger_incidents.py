"""Watch Prometheus metrics and fire real incidents into CommanderV3.

Polls Prometheus every N seconds. When drift or AUC breach thresholds,
creates an Incident and hands it to CommanderV3. This populates:
  - incidents_total       (Incident Feed dashboard)
  - agent_wins_total      (Agent Leaderboard dashboard)
  - agent_proposals_total (Memory State dashboard)

Usage:
    uv run python scripts/trigger_incidents.py
    uv run python scripts/trigger_incidents.py --poll 10 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import httpx

logger = logging.getLogger("trigger")

DRIFT_THRESHOLD = 1.5    # σ from tenant baseline → DRIFT incident
AUC_THRESHOLD = 0.70     # below → DRIFT incident
MISSING_THRESHOLD = 0.10  # above → DATA_QUALITY incident
DUPLICATE_THRESHOLD = 0.05  # above → DATA_QUALITY incident
LATENCY_THRESHOLD = 50   # ms p95 above → LATENCY_BREACH incident

# Business cost thresholds per tenant (expected $ loss per prediction).
# Set above each tenant's normal baseline so alert only fires on degradation.
# standard normal ~$8, free normal ~$15, enterprise normal ~$760.
COST_THRESHOLDS: dict[str, float] = {
    "standard": 25.0,   # normal ~$8; fires when drift causes more wrong predictions
    "free": 40.0,       # normal ~$15; fires when drift causes more wrong predictions
    "enterprise": 660.0, # normal ~$530-$630; max possible ~$770 (15.4% positive × $5000 fn_cost)
}
COST_THRESHOLD_DEFAULT = 50.0  # fallback for unknown tenants


def _query(prometheus_url: str, promql: str) -> list[dict]:
    """Run PromQL instant query, return result list."""
    try:
        r = httpx.get(f"{prometheus_url}/api/v1/query", params={"query": promql}, timeout=5)
        return r.json().get("data", {}).get("result", [])
    except Exception as exc:
        logger.warning("prometheus query failed: %s", exc)
        return []


def _detect_incidents(prometheus_url: str) -> list[dict]:
    """Check metrics and return list of incidents to fire."""
    incidents = []

    # Drift: any tenant × feature where score > threshold
    for row in _query(prometheus_url, f"feature_drift_score > {DRIFT_THRESHOLD}"):
        tenant = row["metric"].get("tenant_id", "unknown")
        feature = row["metric"].get("feature", "unknown")
        score = float(row["value"][1])
        incidents.append({
            "tenant_id": tenant,
            "type": "DRIFT",
            "payload": {"feature": feature, "drift_score": score},
            "severity": min(score / 5.0, 1.0),
            "reason": f"{feature} drift={score:.2f} > {DRIFT_THRESHOLD}",
        })

    # AUC degradation → also a drift incident
    for row in _query(prometheus_url, f"model_auc < {AUC_THRESHOLD}"):
        tenant = row["metric"].get("tenant_id", "unknown")
        auc = float(row["value"][1])
        # avoid duplicate if already flagged for drift
        if not any(i["tenant_id"] == tenant and i["type"] == "DRIFT" for i in incidents):
            incidents.append({
                "tenant_id": tenant,
                "type": "DRIFT",
                "payload": {"auc": auc},
                "severity": max(0.0, (AUC_THRESHOLD - auc) * 5),
                "reason": f"AUC={auc:.3f} < {AUC_THRESHOLD}",
            })

    # Missing rate → DATA_QUALITY
    for row in _query(prometheus_url, f"data_missing_rate > {MISSING_THRESHOLD}"):
        tenant = row["metric"].get("tenant_id", "unknown")
        rate = float(row["value"][1])
        incidents.append({
            "tenant_id": tenant,
            "type": "DATA_QUALITY",
            "payload": {"missing_rate": rate},
            "severity": min(rate * 5, 1.0),
            "reason": f"missing_rate={rate:.2%} > {MISSING_THRESHOLD:.0%}",
        })

    # Duplicate rate → DATA_QUALITY (from DataQualityMonitor via replay)
    for row in _query(prometheus_url, f"data_duplicate_rate > {DUPLICATE_THRESHOLD}"):
        tenant = row["metric"].get("tenant_id", "unknown")
        rate = float(row["value"][1])
        if not any(i["tenant_id"] == tenant and i["type"] == "DATA_QUALITY" for i in incidents):
            incidents.append({
                "tenant_id": tenant,
                "type": "DATA_QUALITY",
                "payload": {"duplicate_rate": rate},
                "severity": min(rate * 10, 1.0),
                "reason": f"duplicate_rate={rate:.2%} > {DUPLICATE_THRESHOLD:.0%}",
            })

    # Latency breach → LATENCY_BREACH
    for row in _query(prometheus_url, f"system_latency_p95_ms > {LATENCY_THRESHOLD}"):
        tenant = row["metric"].get("tenant_id", "unknown")
        latency = float(row["value"][1])
        if not any(i["tenant_id"] == tenant and i["type"] == "LATENCY_BREACH" for i in incidents):
            incidents.append({
                "tenant_id": tenant,
                "type": "LATENCY_BREACH",
                "payload": {"latency_p95_ms": latency},
                "severity": min((latency - LATENCY_THRESHOLD) / 200, 1.0),
                "reason": f"latency_p95={latency:.1f}ms > {LATENCY_THRESHOLD}ms",
            })

    # Cost threshold → COST_THRESHOLD (per-tenant business cost thresholds)
    for row in _query(prometheus_url, "cost_per_prediction"):
        tenant = row["metric"].get("tenant_id", "unknown")
        cost = float(row["value"][1])
        threshold = COST_THRESHOLDS.get(tenant, COST_THRESHOLD_DEFAULT)
        if cost > threshold and not any(
            i["tenant_id"] == tenant and i["type"] == "COST_THRESHOLD" for i in incidents
        ):
            incidents.append({
                "tenant_id": tenant,
                "type": "COST_THRESHOLD",
                "payload": {"cost_per_prediction": cost},
                "severity": min((cost - threshold) / threshold, 1.0),
                "reason": f"cost=${cost:.1f} > threshold=${threshold:.1f} (tenant={tenant})",
            })

    return incidents


async def _fire(commander, incident_spec: dict, dry_run: bool) -> None:
    from self_healing_pipeline.gateway.events import Incident, IncidentType

    type_map = {
        "DRIFT": IncidentType.DRIFT,
        "DATA_QUALITY": IncidentType.DATA_QUALITY,
        "LATENCY_BREACH": IncidentType.LATENCY_BREACH,
        "COST_THRESHOLD": IncidentType.COST_THRESHOLD,
    }

    inc_type = type_map.get(incident_spec["type"], IncidentType.DRIFT)
    incident = Incident(
        tenant_id=incident_spec["tenant_id"],
        type=inc_type,
        payload=incident_spec["payload"],
        severity=incident_spec["severity"],
    )

    logger.info(
        "INCIDENT %s  tenant=%s  severity=%.2f  reason=%s",
        incident.id, incident.tenant_id, incident.severity, incident_spec["reason"],
    )

    if dry_run:
        logger.info("  [dry-run] skipping CommanderV3")
        return

    result = await commander.handle_incident(incident)
    logger.info(
        "  → winner=%s  reward=%.3f  resolved=%s",
        result.winning_agent_type, result.reward, result.incident_resolved,
    )
    if result.escalation_triggered:
        logger.warning("  → ESCALATED: all agents failed")


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--poll", type=int, default=15, help="seconds between checks")
    parser.add_argument("--dry-run", action="store_true", help="detect but don't fire")
    parser.add_argument("--once", action="store_true", help="run one cycle then exit")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Expose metrics on port 8002 so Prometheus can scrape this process
    from prometheus_client import start_http_server
    start_http_server(8002)
    logger.info("metrics server started on :8002")

    # Build CommanderV3 with all 5 agents (real actions wired in)
    from self_healing_pipeline.agents.datarepair_v2 import DataRepairAgent
    from self_healing_pipeline.agents.fallback_v2 import FallbackAgent
    from self_healing_pipeline.agents.retrain_v2 import RetrainAgent
    from self_healing_pipeline.agents.rollback_v2 import RollbackAgent
    from self_healing_pipeline.agents.threshold_v2 import ThresholdAdjustmentAgent
    from self_healing_pipeline.commander.commander_v3 import CommanderV3
    from self_healing_pipeline.config import get_settings
    from self_healing_pipeline.db.session import create_all, session_scope
    from self_healing_pipeline.memory.tier3_traces import BundleWriter

    settings = get_settings()
    api_url = f"http://{settings.api_host}:{settings.api_port}"

    # Ensure new DB tables exist (idempotent)
    create_all()

    agents = [
        ThresholdAdjustmentAgent("threshold-1", session_factory=session_scope),
        RetrainAgent(
            "retrain-1",
            model_path=settings.model_path,
            session_factory=session_scope,
            api_url=api_url,
        ),
        RollbackAgent(
            "rollback-1",
            model_path=settings.model_path,
            session_factory=session_scope,
            api_url=api_url,
        ),
        FallbackAgent("fallback-1"),
        DataRepairAgent("datarepair-1"),
    ]
    commander = CommanderV3(
        agents,
        session_factory=session_scope,
        use_mock_telemetry=False,     # use real Prometheus
        stabilization_seconds=15.0,   # wait 15s for Prometheus gauges to update
        bundle_writer=BundleWriter(settings.traces_dir),  # feed offline meta-harness
    )

    fired: set[str] = set()  # cooldown: (tenant, type) seen in last cycle

    logger.info(
        "watching prometheus=%s  poll=%ds  dry_run=%s",
        args.prometheus, args.poll, args.dry_run,
    )

    while True:
        incidents = _detect_incidents(args.prometheus)

        if not incidents:
            logger.info("no incidents detected")
        else:
            new_fired: set[str] = set()
            for spec in incidents:
                key = f"{spec['tenant_id']}:{spec['type']}"
                if key not in fired:
                    await _fire(commander, spec, args.dry_run)
                    new_fired.add(key)
                else:
                    logger.debug("cooldown: skipping %s (already fired this cycle)", key)
            fired = new_fired

        if args.once:
            break

        await asyncio.sleep(args.poll)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
