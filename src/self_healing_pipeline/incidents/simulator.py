from __future__ import annotations

import numpy as np
import pandas as pd


class IncidentSimulator:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def inject_concept_drift(
        self,
        df: pd.DataFrame,
        *,
        feature: str,
        threshold: float,
        label_col: str = "y",
        flip_prob: float = 1.0,
    ) -> pd.DataFrame:
        if feature not in df.columns:
            raise KeyError(f"feature {feature!r} not in dataframe")
        if label_col not in df.columns:
            raise KeyError(f"label_col {label_col!r} not in dataframe")
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError(f"flip_prob must be in [0, 1], got {flip_prob}")

        out = df.copy()
        region = out[feature] > threshold
        if flip_prob == 0.0 or not region.any():
            return out

        if flip_prob >= 1.0:
            flip_mask = region
        else:
            draws = self.rng.random(int(region.sum()))
            flip_mask = region.copy()
            flip_mask.loc[region] = draws < flip_prob

        out.loc[flip_mask, label_col] = 1 - out.loc[flip_mask, label_col]
        return out
