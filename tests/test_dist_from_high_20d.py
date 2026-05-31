import numpy as np
import pandas as pd
from dfi_features.dist_from_high_20d import compute


def _daily_df(n=30, seed=42):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.DataFrame({"close": close}, index=ts)


def test_warmup_is_nan():
    df = _daily_df(n=30)
    s = compute(df, {"window_d": 20})
    assert s.iloc[:19].isna().all()
    assert not pd.isna(s.iloc[19])


def test_always_non_positive():
    """Signal is always <= 0 — close can only be at or below its rolling max."""
    df = _daily_df(n=60)
    s = compute(df, {}).dropna()
    assert (s <= 0).all(), "dist_from_high must always be <= 0"


def test_at_high_gives_zero():
    """When close equals the rolling max, signal = 0."""
    n = 25
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = np.ones(n) * 100.0
    close[-1] = 200.0
    df = pd.DataFrame({"close": close}, index=ts)
    s = compute(df, {"window_d": 20})
    assert s.iloc[-1] == 0.0, "at the 20d high → signal = 0"


def test_far_below_high_is_negative():
    n = 25
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = np.ones(n) * 100.0
    close[0] = 200.0
    close[-1] = 50.0
    df = pd.DataFrame({"close": close}, index=ts)
    s = compute(df, {"window_d": 20})
    assert s.iloc[-1] <= -0.5, "50% below high → signal <= -0.5"


def test_missing_column_returns_empty():
    df = pd.DataFrame({"volume": [1.0, 2.0]})
    assert compute(df, {}).empty


def test_no_mutation_of_input():
    df = _daily_df()
    cols = list(df.columns)
    compute(df, {})
    assert list(df.columns) == cols
