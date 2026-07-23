from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import REGISTRY, generate_latest
from pydantic import BaseModel

from self_healing_pipeline.config import get_settings
from self_healing_pipeline.observability.metrics import (
    cost_per_prediction,
    data_duplicate_rate,
    data_missing_rate,
    data_schema_violations,
    false_negative_rate,
    false_positive_rate,
    feature_drift_score,
    model_auc,
    model_calibration_error,
    model_drift_percentage,
    model_error_rate,
    model_precision,
    model_recall,
    prediction_count,
    prediction_latency,
    system_latency_p95,
    system_latency_p99,
)
from self_healing_pipeline.pipeline.serving import ModelServer

SERVICE_NAME = "self-healing-pipeline"


def _service_version() -> str:
    try:
        return version(SERVICE_NAME)
    except PackageNotFoundError:
        return "0.0.0"


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class PredictRequest(BaseModel):
    features: dict[str, Any]


class PredictResponse(BaseModel):
    tenant_id: str
    probability: float
    threshold: float
    label: int
    expected_cost: float


class IncidentSummary(BaseModel):
    incident_id: str
    tenant_id: str
    incident_type: str
    severity: float
    winner_agent: str | None
    execution_success: bool | None
    timestamp: str


class IncidentsResponse(BaseModel):
    incidents: list[IncidentSummary]
    total_count: int


class AgentStats(BaseModel):
    agent_type: str
    total_proposals: int
    successful_proposals: int
    success_rate: float
    total_savings: float
    avg_savings: float


class AgentSummaryResponse(BaseModel):
    agents: list[AgentStats]
    timestamp: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if not settings.model_path.exists():
        raise RuntimeError(
            f"model not found at {settings.model_path}; "
            "run `uv run python scripts/train.py` before starting the API"
        )
    app.state.model_server = ModelServer.from_path(settings.model_path)
    yield


app = FastAPI(title="Self-Healing ML Pipeline", version=_service_version(), lifespan=lifespan)


def get_model_server(request: Request) -> ModelServer:
    server = getattr(request.app.state, "model_server", None)
    if server is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    assert isinstance(server, ModelServer)
    return server


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME, version=_service_version())


@app.post("/predict/{tenant_id}", response_model=PredictResponse)
def predict(
    tenant_id: str,
    body: PredictRequest,
    server: Annotated[ModelServer, Depends(get_model_server)],
) -> PredictResponse:
    import time

    if tenant_id not in server.tenants:
        raise HTTPException(status_code=404, detail=f"unknown tenant {tenant_id!r}")

    start_time = time.time()
    try:
        pred = server.predict(tenant_id, body.features)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc).strip("'")) from exc
    finally:
        # Record metrics
        duration = time.time() - start_time
        prediction_count.labels(tenant_id=tenant_id).inc()
        prediction_latency.labels(tenant_id=tenant_id).observe(duration)

    return PredictResponse(
        tenant_id=pred.tenant_id,
        probability=pred.probability,
        threshold=pred.threshold,
        label=pred.label,
        expected_cost=pred.expected_cost,
    )


@app.get("/incidents/recent", response_model=IncidentsResponse)
def recent_incidents(limit: int = 100) -> IncidentsResponse:
    """Get recent incidents from evidence bundles."""
    settings = get_settings()
    traces_dir = settings.traces_dir

    if not traces_dir.exists():
        return IncidentsResponse(incidents=[], total_count=0)

    incidents: list[IncidentSummary] = []

    # Scan all incident directories
    for inc_dir in sorted(traces_dir.glob("inc-*"), reverse=True)[:limit]:
        bundle_file = inc_dir / "evidence_bundle.json"
        if not bundle_file.exists():
            continue

        try:
            with open(bundle_file) as f:
                bundle = json.load(f)

            incident = bundle.get("incident", {})
            execution = bundle.get("execution_result", {})
            winner = bundle.get("winner", {})

            incidents.append(
                IncidentSummary(
                    incident_id=inc_dir.name,
                    tenant_id=incident.get("tenant_id", "unknown"),
                    incident_type=incident.get("type", "UNKNOWN"),
                    severity=incident.get("severity", 0.0),
                    winner_agent=winner.get("agent_type"),
                    execution_success=execution.get("success"),
                    timestamp=bundle.get("timestamp", ""),
                )
            )
        except (json.JSONDecodeError, KeyError):
            continue

    return IncidentsResponse(incidents=incidents, total_count=len(incidents))


@app.get("/agents/summary", response_model=AgentSummaryResponse)
def agent_summary() -> AgentSummaryResponse:
    """Get agent performance summary from all evidence bundles."""
    from datetime import UTC, datetime

    settings = get_settings()
    traces_dir = settings.traces_dir

    if not traces_dir.exists():
        return AgentSummaryResponse(agents=[], timestamp=datetime.now(UTC).isoformat())

    # Collect stats per agent
    agent_stats: dict[str, dict[str, float | int]] = {}

    # Scan all incident directories
    for inc_dir in sorted(traces_dir.glob("inc-*")):
        bundle_file = inc_dir / "evidence_bundle.json"
        if not bundle_file.exists():
            continue

        try:
            with open(bundle_file) as f:
                bundle = json.load(f)

            # Get winner info
            winner = bundle.get("winner", {})
            winner_agent = winner.get("agent_type")
            if not winner_agent:
                continue

            execution = bundle.get("execution_result", {})
            success = execution.get("success", False)
            savings = execution.get("actual_business_savings", 0.0)

            # Initialize agent stats if needed
            if winner_agent not in agent_stats:
                agent_stats[winner_agent] = {
                    "total_proposals": 0,
                    "successful_proposals": 0,
                    "total_savings": 0.0,
                }

            # Update stats
            agent_stats[winner_agent]["total_proposals"] += 1
            if success:
                agent_stats[winner_agent]["successful_proposals"] += 1
            agent_stats[winner_agent]["total_savings"] += savings
        except (json.JSONDecodeError, KeyError):
            continue

    # Convert to response format
    agents: list[AgentStats] = []
    for agent_type, stats in sorted(agent_stats.items()):
        total = stats["total_proposals"]
        successful = stats["successful_proposals"]
        success_rate = successful / total if total > 0 else 0.0
        avg_savings = stats["total_savings"] / total if total > 0 else 0.0

        agents.append(
            AgentStats(
                agent_type=agent_type,
                total_proposals=int(total),
                successful_proposals=int(successful),
                success_rate=success_rate,
                total_savings=stats["total_savings"],
                avg_savings=avg_savings,
            )
        )

    return AgentSummaryResponse(agents=agents, timestamp=datetime.now(UTC).isoformat())


class ModelQualityUpdate(BaseModel):
    tenant_id: str
    auc: float
    precision: float
    recall: float
    error_rate: float
    calibration_error: float
    missing_rate: float
    duplicate_rate: float
    schema_violations: int
    latency_p95_ms: float
    latency_p99_ms: float
    cost_per_prediction: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    drift_percentage: float = 0.0
    feature_drift: dict[str, float] = {}


@app.post("/internal/metrics/update", status_code=204)
def update_model_metrics(body: ModelQualityUpdate) -> None:
    """Update model quality Gauges from replay script (has ground truth labels)."""
    tid = body.tenant_id
    model_auc.labels(tenant_id=tid).set(body.auc)
    model_precision.labels(tenant_id=tid).set(body.precision)
    model_recall.labels(tenant_id=tid).set(body.recall)
    model_error_rate.labels(tenant_id=tid).set(body.error_rate)
    model_calibration_error.labels(tenant_id=tid).set(body.calibration_error)
    data_missing_rate.labels(tenant_id=tid).set(body.missing_rate)
    data_duplicate_rate.labels(tenant_id=tid).set(body.duplicate_rate)
    data_schema_violations.labels(tenant_id=tid).set(body.schema_violations)
    system_latency_p95.labels(tenant_id=tid).set(body.latency_p95_ms)
    system_latency_p99.labels(tenant_id=tid).set(body.latency_p99_ms)
    cost_per_prediction.labels(tenant_id=tid).set(body.cost_per_prediction)
    false_positive_rate.labels(tenant_id=tid).set(body.false_positive_rate)
    false_negative_rate.labels(tenant_id=tid).set(body.false_negative_rate)
    model_drift_percentage.labels(tenant_id=tid).set(body.drift_percentage)
    for feat, score in body.feature_drift.items():
        feature_drift_score.labels(tenant_id=tid, feature=feat).set(score)


@app.post("/internal/reload-model", status_code=204)
def reload_model(request: Request) -> None:
    """Reload model from disk after retraining or rollback."""
    server = getattr(request.app.state, "model_server", None)
    if server is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    settings = get_settings()
    server.reload(settings.model_path)


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(generate_latest(REGISTRY), media_type="text/plain")
