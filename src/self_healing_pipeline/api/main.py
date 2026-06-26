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
    if tenant_id not in server.tenants:
        raise HTTPException(status_code=404, detail=f"unknown tenant {tenant_id!r}")
    try:
        pred = server.predict(tenant_id, body.features)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc).strip("'")) from exc
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


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(generate_latest(REGISTRY), media_type="text/plain")
