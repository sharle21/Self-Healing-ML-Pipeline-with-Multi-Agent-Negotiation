from __future__ import annotations

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

UCI_CREDIT_DEFAULT_ID = 350
TARGET_COL = "default payment next month"
LABEL_COL = "y"
TENANT_COL = "tenant_id"
TIER_STANDARD = "standard"
TIER_ENTERPRISE = "enterprise"
TIER_FREE = "free"


def fetch_uci_credit_default() -> pd.DataFrame:
    bundle = fetch_ucirepo(id=UCI_CREDIT_DEFAULT_ID)
    features = bundle.data.features.copy()
    targets = bundle.data.targets.copy()
    features[LABEL_COL] = targets[TARGET_COL].astype(int).to_numpy()
    return features


def split_by_tenant(
    df: pd.DataFrame,
    *,
    limit_col: str = "LIMIT_BAL",
    cold_start_frac: float = 0.05,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if limit_col not in df.columns:
        raise KeyError(f"{limit_col!r} missing; cannot tertile-split")
    if not 0.0 < cold_start_frac <= 1.0:
        raise ValueError(f"cold_start_frac must be in (0, 1], got {cold_start_frac}")

    rng = rng if rng is not None else np.random.default_rng(0)
    out = df.copy()
    q_low, q_high = out[limit_col].quantile([1 / 3, 2 / 3])

    tenant = pd.Series(TIER_FREE, index=out.index, dtype="object")
    tenant[out[limit_col] <= q_low] = TIER_STANDARD
    tenant[out[limit_col] >= q_high] = TIER_ENTERPRISE
    out[TENANT_COL] = tenant

    middle_idx = out.index[out[TENANT_COL] == TIER_FREE]
    target_size = max(1, int(round(cold_start_frac * len(out))))
    if len(middle_idx) > target_size:
        drop = rng.choice(middle_idx.to_numpy(), size=len(middle_idx) - target_size, replace=False)
        out = out.drop(index=drop).reset_index(drop=True)
    return out
