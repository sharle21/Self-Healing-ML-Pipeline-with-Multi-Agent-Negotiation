from __future__ import annotations

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

UCI_CREDIT_DEFAULT_ID = 350
LABEL_COL = "y"
TENANT_COL = "tenant_id"
TIER_STANDARD = "standard"
TIER_ENTERPRISE = "enterprise"
TIER_FREE = "free"

# X1-X23 column mapping for UCI Default of Credit Card Clients (ID=350)
_UCI_RENAME = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": LABEL_COL,
}


def fetch_uci_credit_default() -> pd.DataFrame:
    bundle = fetch_ucirepo(id=UCI_CREDIT_DEFAULT_ID)
    features = bundle.data.features.copy()
    targets = bundle.data.targets.copy()

    # UCI lib returns X1-X23 / Y; rename to domain column names
    if "X1" in features.columns:
        features = features.rename(columns=_UCI_RENAME)
    if "Y" in targets.columns:
        targets = targets.rename(columns={"Y": LABEL_COL})

    label_col = LABEL_COL if LABEL_COL in targets.columns else targets.columns[0]
    features[LABEL_COL] = targets[label_col].astype(int).to_numpy()
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
