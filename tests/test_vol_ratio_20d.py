import numpy as np
import pandas as pd
from dfi_features.vol_ratio_20d import compute


def _daily_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    ts  = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    vol = rng.uniform(1000, 5000, n)
    return pd.DataFrame({"volume": vol}, index=ts)


def test_warmup_is_nan():
    df = _daily_df(n=300)
    s  = compute(df, {"window_d": 20, "window_norm": 252})
    assert s.iloc[:270].isna().all()
    assert not pd.isna(s.iloc[270])


def test_output_is_float_series():
    assert compute(_daily_df(), {}).dtype == float


def test_output_index_matches_input():
    df = _daily_df()
    assert compute(df, {}).index.equals(df.index)


def test_volume_spike_gives_positive():
    n   = 300
    ts  = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    vol = np.ones(n) * 1000.0
    vol[-1] = 50000.0
    df  = pd.DataFrame({"volume": vol}, index=ts)
    s   = compute(df, {"window_d": 20, "window_norm": 252})
    assert s.iloc[-1] > 0


def test_missing_column_returns_empty():
    assert compute(pd.DataFrame({"close": [1.0]}), {}).empty


def test_no_mutation_of_input():
    df   = _daily_df()
    cols = list(df.columns)
    compute(df, {})
    assert list(df.columns) == cols
