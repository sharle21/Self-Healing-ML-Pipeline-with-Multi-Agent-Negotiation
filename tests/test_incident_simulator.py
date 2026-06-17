import numpy as np
import pandas as pd
import pytest

from self_healing_pipeline.incidents import IncidentSimulator


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "y": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )


def test_full_flip_above_threshold() -> None:
    sim = IncidentSimulator()
    df = _sample_df()
    out = sim.inject_concept_drift(df, feature="income", threshold=50)
    # income > 50 flipped: y was [1,1,1,1,1] -> [0,0,0,0,0]; income <= 50 untouched
    assert out.loc[df["income"] > 50, "y"].tolist() == [0, 0, 0, 0, 0]
    assert out.loc[df["income"] <= 50, "y"].tolist() == [0, 0, 0, 0, 0]


def test_input_not_mutated() -> None:
    sim = IncidentSimulator()
    df = _sample_df()
    snapshot = df.copy()
    sim.inject_concept_drift(df, feature="income", threshold=50)
    pd.testing.assert_frame_equal(df, snapshot)


def test_flip_prob_zero_is_noop() -> None:
    sim = IncidentSimulator()
    df = _sample_df()
    out = sim.inject_concept_drift(df, feature="income", threshold=50, flip_prob=0.0)
    pd.testing.assert_frame_equal(out, df)


def test_partial_flip_deterministic_with_seed() -> None:
    df = _sample_df()
    out_a = IncidentSimulator(rng=np.random.default_rng(42)).inject_concept_drift(
        df, feature="income", threshold=50, flip_prob=0.5
    )
    out_b = IncidentSimulator(rng=np.random.default_rng(42)).inject_concept_drift(
        df, feature="income", threshold=50, flip_prob=0.5
    )
    pd.testing.assert_frame_equal(out_a, out_b)
    # below-threshold rows untouched
    assert out_a.loc[df["income"] <= 50, "y"].tolist() == [0, 0, 0, 0, 0]
    # some subset of above-threshold rows flipped
    above = out_a.loc[df["income"] > 50, "y"].tolist()
    assert 0 <= sum(1 for v in above if v == 0) <= 5


def test_missing_feature_raises() -> None:
    sim = IncidentSimulator()
    with pytest.raises(KeyError):
        sim.inject_concept_drift(_sample_df(), feature="missing", threshold=0)


def test_bad_flip_prob_raises() -> None:
    sim = IncidentSimulator()
    with pytest.raises(ValueError):
        sim.inject_concept_drift(_sample_df(), feature="income", threshold=50, flip_prob=1.5)


def test_inject_missing_values() -> None:
    sim = IncidentSimulator(rng=np.random.default_rng(42))
    df = _sample_df()
    result = sim.inject_missing_values(df, features=["income"], missing_rate=0.2)

    assert len(result) == len(df)
    missing_count = result["income"].isna().sum()
    assert missing_count > 0
    assert missing_count <= 3


def test_inject_duplicates() -> None:
    sim = IncidentSimulator(rng=np.random.default_rng(42))
    df = _sample_df()
    result = sim.inject_duplicates(df, duplicate_rate=0.2)

    assert len(result) > len(df)
    assert len(result) <= len(df) * 1.3


def test_inject_schema_violation() -> None:
    sim = IncidentSimulator(rng=np.random.default_rng(42))
    df = _sample_df()
    result = sim.inject_schema_violation(df, feature="income", violation_rate=0.2)

    assert len(result) == len(df)
    invalid_count = (result["income"] == "INVALID").sum()
    assert invalid_count > 0
    assert invalid_count <= 3


def test_inject_volume_drop() -> None:
    sim = IncidentSimulator(rng=np.random.default_rng(42))
    df = _sample_df()
    result = sim.inject_volume_drop(df, drop_rate=0.3)

    assert len(result) < len(df)
    assert len(result) >= int(len(df) * 0.65)
    assert len(result) <= int(len(df) * 0.75)
