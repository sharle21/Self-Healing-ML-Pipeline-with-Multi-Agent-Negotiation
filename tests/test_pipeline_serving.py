from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
def server(tmp_path: Path) -> ModelServer:
    df = _training_frame()
    result = train_model(df, random_state=0)
    persist_model(result, tmp_path / "lgbm.joblib")
    return ModelServer.from_path(tmp_path / "lgbm.joblib")


def test_predict_uses_per_tenant_threshold(server: ModelServer) -> None:
    features = {"LIMIT_BAL": 50_000, "AGE": 30, "PAY_0": 4}
    std = server.predict("standard", features)
    ent = server.predict("enterprise", features)
    assert std.threshold == 0.5
    assert ent.threshold == 0.8
    # probabilities may differ (tenant_id is a feature); thresholds differ by design
    assert 0.0 <= std.probability <= 1.0
    assert 0.0 <= ent.probability <= 1.0
    assert std.label == int(std.probability >= std.threshold)
    assert ent.label == int(ent.probability >= ent.threshold)


def test_predict_unknown_tenant_raises(server: ModelServer) -> None:
    with pytest.raises(KeyError):
        server.predict("premium", {"LIMIT_BAL": 1, "AGE": 1, "PAY_0": 0})


def test_predict_missing_feature_raises(server: ModelServer) -> None:
    with pytest.raises(KeyError):
        server.predict("standard", {"LIMIT_BAL": 50_000})


def test_expected_cost_sign(server: ModelServer) -> None:
    # Push probability very high via features known to drive positive signal
    features = {"LIMIT_BAL": 20_000, "AGE": 40, "PAY_0": 6}
    pred = server.predict("standard", features)
    assert pred.expected_cost >= 0.0
