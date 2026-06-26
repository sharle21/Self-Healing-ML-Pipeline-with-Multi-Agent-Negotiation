from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
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


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(generate_latest(REGISTRY), media_type="text/plain")
