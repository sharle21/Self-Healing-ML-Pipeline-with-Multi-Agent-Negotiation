import numpy as np
import pandas as pd
import pytest

from self_healing_pipeline.config import load_tenants_config
from self_healing_pipeline.pipeline.loader import (
    LABEL_COL,
    TENANT_COL,
    split_by_tenant,
)


def _synthetic_uci(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "LIMIT_BAL": np.tile(np.arange(1, 301), 3)[:n],
            "AGE": rng.integers(20, 70, size=n),
            LABEL_COL: rng.integers(0, 2, size=n),
        }
    )


def test_split_tertiles_and_cold_start_downsample() -> None:
    df = _synthetic_uci(900)
    out = split_by_tenant(df, cold_start_frac=0.05, rng=np.random.default_rng(0))
    counts = out[TENANT_COL].value_counts().to_dict()
    # standard=bottom third (~300), enterprise=top third (~300), free=5% of 900 = 45
    assert counts["standard"] >= 290
    assert counts["enterprise"] >= 290
    assert 40 <= counts["free"] <= 50


def test_split_tiers_map_to_limit_bal() -> None:
    df = _synthetic_uci(900)
    out = split_by_tenant(df, cold_start_frac=0.05, rng=np.random.default_rng(0))
    enterprise_max = out.loc[out[TENANT_COL] == "enterprise", "LIMIT_BAL"].min()
    standard_max = out.loc[out[TENANT_COL] == "standard", "LIMIT_BAL"].max()
    assert enterprise_max > standard_max


def test_split_missing_limit_col_raises() -> None:
    with pytest.raises(KeyError):
        split_by_tenant(pd.DataFrame({"foo": [1, 2, 3]}))


def test_split_bad_frac_raises() -> None:
    with pytest.raises(ValueError):
        split_by_tenant(_synthetic_uci(30), cold_start_frac=1.5)


def test_tenants_config_has_three_tiers() -> None:
    tenants = load_tenants_config()
    assert set(tenants.keys()) == {"standard", "enterprise", "free"}
    for cfg in tenants.values():
        assert {"confidence_threshold", "false_positive_cost", "false_negative_cost"} <= cfg.keys()
