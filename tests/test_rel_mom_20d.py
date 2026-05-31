import numpy as np
import pandas as pd
from dfi_features.rel_mom_20d import compute


def _daily_df(n=300, seed=42, vol=0.02):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    return pd.DataFrame({"close": close}, index=ts)


def test_warmup_is_nan():
    df = _daily_df(n=300)
    s = compute(df, {"window_mom": 20, "window_norm": 252})
    assert s.iloc[:271].isna().all(), "warmup bars must be NaN"
    assert not pd.isna(s.iloc[271]), "bar 272 must be valid"


def test_output_is_float_series():
    df = _daily_df()
    s = compute(df, {})
    assert isinstance(s, pd.Series)
    assert s.dtype == float


def test_output_index_matches_input():
    df = _daily_df()
    s = compute(df, {})
    assert s.index.equals(df.index)


def test_strong_momentum_gives_positive():
    n = 300
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = np.ones(n) * 100.0
    close[-1] = 150.0
    df = pd.DataFrame({"close": close}, index=ts)
    s = compute(df, {"window_mom": 20, "window_norm": 252})
    assert s.iloc[-1] > 0, "strong positive return → positive z-score"


def test_missing_column_returns_empty():
    df = pd.DataFrame({"volume": [1.0, 2.0]})
    assert compute(df, {}).empty


def test_no_mutation_of_input():
    df = _daily_df()
    cols = list(df.columns)
    compute(df, {})
    assert list(df.columns) == cols
