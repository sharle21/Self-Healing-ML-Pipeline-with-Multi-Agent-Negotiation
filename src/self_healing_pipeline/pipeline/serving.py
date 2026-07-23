from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier

from self_healing_pipeline.config import load_tenants_config
from self_healing_pipeline.pipeline.loader import TENANT_COL
from self_healing_pipeline.pipeline.trainer import load_model

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    tenant_id: str
    probability: float
    threshold: float
    label: int
    expected_cost: float


class ModelServer:
    def __init__(
        self,
        model: LGBMClassifier,
        feature_names: list[str],
        tenants: dict[str, dict[str, Any]],
        db_path: str | None = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.tenants = tenants
        self._db_path = db_path  # sqlite file path for live threshold reads

    @classmethod
    def from_path(cls, path: Path, db_path: str | None = None) -> ModelServer:
        if db_path is None:
            from self_healing_pipeline.config import get_settings
            url = get_settings().db_url
            if url.startswith("sqlite:///"):
                db_path = url[len("sqlite:///"):]
        model, names = load_model(path)
        return cls(model, names, load_tenants_config(), db_path=db_path)

    def reload(self, path: Path) -> None:
        """Reload model from disk (called after retraining or rollback)."""
        model, names = load_model(path)
        self.model = model
        self.feature_names = names
        logger.info("model reloaded from %s", path)

    def _live_threshold(self, tenant_id: str, default: float) -> float:
        """Read tenant threshold from SQLite override table; fall back to default."""
        if not self._db_path:
            return default
        try:
            import sqlite3
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT threshold FROM tenant_threshold_overrides WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchone()
                return float(row[0]) if row else default
        except Exception as exc:
            logger.debug("threshold DB read failed: %s", exc)
            return default

    def predict(self, tenant_id: str, features: dict[str, Any]) -> Prediction:
        if tenant_id not in self.tenants:
            raise KeyError(f"unknown tenant {tenant_id!r}")

        row = {**features, TENANT_COL: tenant_id}
        missing = [c for c in self.feature_names if c not in row]
        if missing:
            raise KeyError(f"missing features: {missing}")

        frame = pd.DataFrame([row])[self.feature_names]
        frame[TENANT_COL] = frame[TENANT_COL].astype("category")

        proba = float(self.model.predict_proba(frame)[0, 1])
        cfg = self.tenants[tenant_id]
        default_threshold = float(cfg["confidence_threshold"])
        threshold = self._live_threshold(tenant_id, default_threshold)
        label = int(proba >= threshold)
        fp_cost = float(cfg["false_positive_cost"])
        fn_cost = float(cfg["false_negative_cost"])
        # Expected cost under the chosen label:
        # label=1 risks FP (actual negative) weighted by (1-proba).
        # label=0 risks FN (actual positive) weighted by proba.
        expected_cost = (1 - proba) * fp_cost if label == 1 else proba * fn_cost
        return Prediction(
            tenant_id=tenant_id,
            probability=proba,
            threshold=threshold,
            label=label,
            expected_cost=expected_cost,
        )
