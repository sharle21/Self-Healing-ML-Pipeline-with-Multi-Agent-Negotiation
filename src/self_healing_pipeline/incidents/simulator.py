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

    def inject_missing_values(
        self,
        df: pd.DataFrame,
        *,
        features: str | list[str] | None = None,
        missing_rate: float = 0.1,
    ) -> pd.DataFrame:
        """Introduce missing values (NaN) in specified columns."""
        if not 0.0 <= missing_rate <= 1.0:
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")

        out = df.copy()
        if features is None:
            features = out.columns.tolist()
        elif isinstance(features, str):
            features = [features]

        for feature in features:
            if feature not in out.columns:
                raise KeyError(f"feature {feature!r} not in dataframe")
            n_missing = int(len(out) * missing_rate)
            if n_missing > 0:
                idx = self.rng.choice(len(out), n_missing, replace=False)
                out.loc[idx, feature] = np.nan

        return out

    def inject_duplicates(
        self,
        df: pd.DataFrame,
        *,
        duplicate_rate: float = 0.05,
    ) -> pd.DataFrame:
        """Introduce duplicate rows."""
        if not 0.0 <= duplicate_rate <= 1.0:
            raise ValueError(f"duplicate_rate must be in [0, 1], got {duplicate_rate}")

        out = df.copy()
        n_duplicates = int(len(out) * duplicate_rate)
        if n_duplicates > 0:
            idx = self.rng.choice(len(out), n_duplicates, replace=True)
            duplicates = out.iloc[idx].reset_index(drop=True)
            out = pd.concat([out, duplicates], ignore_index=True)

        return out

    def inject_schema_violation(
        self,
        df: pd.DataFrame,
        *,
        feature: str,
        violation_rate: float = 0.05,
    ) -> pd.DataFrame:
        """Introduce schema violations (wrong type/format in a column)."""
        if not 0.0 <= violation_rate <= 1.0:
            raise ValueError(f"violation_rate must be in [0, 1], got {violation_rate}")

        if feature not in df.columns:
            raise KeyError(f"feature {feature!r} not in dataframe")

        out = df.copy()
        n_violations = int(len(out) * violation_rate)
        if n_violations > 0:
            out[feature] = out[feature].astype(object)
            idx = self.rng.choice(len(out), n_violations, replace=False)
            out.loc[idx, feature] = "INVALID"

        return out

    def inject_volume_drop(
        self,
        df: pd.DataFrame,
        *,
        drop_rate: float = 0.3,
    ) -> pd.DataFrame:
        """Simulate sudden volume drop by removing rows."""
        if not 0.0 <= drop_rate <= 1.0:
            raise ValueError(f"drop_rate must be in [0, 1], got {drop_rate}")

        n_keep = int(len(df) * (1.0 - drop_rate))
        if n_keep <= 0:
            return df.iloc[:0].copy()

        idx = self.rng.choice(len(df), n_keep, replace=False)
        return df.iloc[idx].reset_index(drop=True)
