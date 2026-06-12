from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from self_healing_pipeline.api.main import app, get_model_server
from self_healing_pipeline.pipeline.loader import LABEL_COL, TENANT_COL
from self_healing_pipeline.pipeline.serving import ModelServer
from self_healing_pipeline.pipeline.trainer import persist_model, train_model


def _training_frame(n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    limit = rng.uniform(10_000, 500_000, size=n)
    age = rng.integers(20, 70, size=n)
    pay = rng.integers(-2, 8, size=n)
    signal = (limit < 100_000).astype(int) * 0.7 + (pay > 2).astype(int) * 0.5
    noise = rng.normal(0, 0.15, size=n)
    probs = np.clip(signal + noise, 0, 1)
    y = (rng.uniform(size=n) < probs).astype(int)
    tenant = np.where(limit < 100_000, "standard", np.where(limit > 300_000, "enterprise", "free"))
    return pd.DataFrame(
        {"LIMIT_BAL": limit, "AGE": age, "PAY_0": pay, TENANT_COL: tenant, LABEL_COL: y}
    )


@pytest.fixture
def client(tmp_path: Path) -> Any:
    result = train_model(_training_frame(), random_state=0)
    persist_model(result, tmp_path / "lgbm.joblib")
    server = ModelServer.from_path(tmp_path / "lgbm.joblib")
    app.dependency_overrides[get_model_server] = lambda: server
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_model_server, None)


def test_predict_happy_path(client: TestClient) -> None:
    resp = client.post(
        "/predict/standard",
        json={"features": {"LIMIT_BAL": 50_000, "AGE": 30, "PAY_0": 4}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "standard"
    assert 0.0 <= body["probability"] <= 1.0
    assert body["threshold"] == 0.5
    assert body["label"] in (0, 1)
    assert body["expected_cost"] >= 0.0


def test_predict_unknown_tenant_returns_404(client: TestClient) -> None:
    resp = client.post(
        "/predict/premium",
        json={"features": {"LIMIT_BAL": 1, "AGE": 1, "PAY_0": 0}},
    )
    assert resp.status_code == 404


def test_predict_missing_feature_returns_422(client: TestClient) -> None:
    resp = client.post("/predict/standard", json={"features": {"LIMIT_BAL": 50_000}})
    assert resp.status_code == 422


def test_predict_threshold_is_per_tenant(client: TestClient) -> None:
    features = {"LIMIT_BAL": 50_000, "AGE": 30, "PAY_0": 4}
    std = client.post("/predict/standard", json={"features": features}).json()
    ent = client.post("/predict/enterprise", json={"features": features}).json()
    assert std["threshold"] == 0.5
    assert ent["threshold"] == 0.8
