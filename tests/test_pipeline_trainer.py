from pathlib import Path

import numpy as np
import pandas as pd

from self_healing_pipeline.pipeline.loader import LABEL_COL, TENANT_COL
from self_healing_pipeline.pipeline.trainer import load_model, persist_model, train_model


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


def test_train_returns_auc_and_per_tenant() -> None:
    result = train_model(_training_frame(), random_state=0)
    assert 0.5 < result.overall_auc <= 1.0
    assert set(result.per_tenant_auc.keys()) <= {"standard", "enterprise", "free"}
    assert TENANT_COL in result.feature_names


def test_persist_and_load_roundtrip(tmp_path: Path) -> None:
    df = _training_frame()
    result = train_model(df, random_state=0)
    path = persist_model(result, tmp_path / "lgbm.joblib")
    assert path.exists()

    model, names = load_model(path)
    features = df.drop(columns=[LABEL_COL]).head(5).copy()
    features[TENANT_COL] = features[TENANT_COL].astype("category")
    proba = model.predict_proba(features[names])
    assert proba.shape == (5, 2)
