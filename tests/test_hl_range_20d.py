import numpy as np
import pandas as pd
from dfi_features.hl_range_20d import compute


def _daily_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    high  = close * (1 + rng.uniform(0, 0.03, n))
    low   = close * (1 - rng.uniform(0, 0.03, n))
    return pd.DataFrame({"close": close, "high": high, "low": low}, index=ts)


def test_warmup_is_nan():
    df = _daily_df(n=300)
    s = compute(df, {"window_d": 20, "window_norm": 252})
    assert s.iloc[:270].isna().all()
    assert not pd.isna(s.iloc[270])


def test_output_is_float_series():
    df = _daily_df()
    assert isinstance(compute(df, {}), pd.Series)


def test_output_index_matches_input():
    df = _daily_df()
    assert compute(df, {}).index.equals(df.index)


def test_wide_range_gives_positive():
    """Day with unusually wide H-L range → positive z-score."""
    n = 300
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = np.ones(n) * 100.0
    high  = close * 1.01
    low   = close * 0.99
    high[-1] = close[-1] * 1.20
    low[-1]  = close[-1] * 0.80
    df = pd.DataFrame({"close": close, "high": high, "low": low}, index=ts)
    s = compute(df, {"window_d": 20, "window_norm": 252})
    assert s.iloc[-1] > 0


def test_missing_columns_returns_empty():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    assert compute(df, {}).empty


def test_no_mutation_of_input():
    df = _daily_df()
    cols = list(df.columns)
    compute(df, {})
    assert list(df.columns) == cols
