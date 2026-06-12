from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from self_healing_pipeline.pipeline.loader import LABEL_COL, TENANT_COL


@dataclass
class TrainResult:
    model: LGBMClassifier
    feature_names: list[str]
    overall_auc: float
    per_tenant_auc: dict[str, float]


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if LABEL_COL not in df.columns:
        raise KeyError(f"{LABEL_COL!r} missing from training frame")
    if TENANT_COL not in df.columns:
        raise KeyError(f"{TENANT_COL!r} missing; call split_by_tenant first")

    features = df.drop(columns=[LABEL_COL]).copy()
    features[TENANT_COL] = features[TENANT_COL].astype("category")
    labels = df[LABEL_COL].astype(int)
    return features, labels, list(features.columns)


def train_model(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 0,
    model_params: dict[str, Any] | None = None,
) -> TrainResult:
    features, labels, names = _prepare_features(df)
    strat = labels if labels.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=strat
    )

    params: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": random_state,
        "verbose": -1,
    }
    if model_params:
        params.update(model_params)

    model: LGBMClassifier = LGBMClassifier(**params)
    model.fit(x_train, y_train, categorical_feature=[TENANT_COL])

    proba = model.predict_proba(x_test)[:, 1]
    overall = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan")

    per_tenant: dict[str, float] = {}
    for tenant in sorted(x_test[TENANT_COL].unique()):
        mask = x_test[TENANT_COL] == tenant
        y_sub = y_test[mask]
        if y_sub.nunique() < 2:
            per_tenant[str(tenant)] = float("nan")
            continue
        per_tenant[str(tenant)] = float(roc_auc_score(y_sub, proba[np.asarray(mask)]))

    return TrainResult(
        model=model, feature_names=names, overall_auc=overall, per_tenant_auc=per_tenant
    )


def persist_model(result: TrainResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": result.model, "feature_names": result.feature_names}, path)
    return path


def load_model(path: Path) -> tuple[LGBMClassifier, list[str]]:
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_names"]
