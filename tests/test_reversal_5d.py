import numpy as np
import pandas as pd
from dfi_features.reversal_5d import compute


def _daily_df(n=60, seed=42):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.DataFrame({"close": close}, index=ts)


def test_warmup_is_nan():
    df = _daily_df()
    s = compute(df, {"window_d": 5})
    assert s.iloc[:5].isna().all(), "first 5 bars must be NaN"
    assert not pd.isna(s.iloc[5]), "bar 6 must be valid"


def test_sign_is_inverted():
    """Asset that fell → positive reversal signal."""
    n = 10
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = np.ones(n) * 100.0
    close[-1] = 80.0
    df = pd.DataFrame({"close": close}, index=ts)
    s = compute(df, {"window_d": 5})
    assert s.iloc[-1] > 0, "price drop → positive reversal"


def test_rally_gives_negative():
    n = 10
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = np.ones(n) * 100.0
    close[-1] = 120.0
    df = pd.DataFrame({"close": close}, index=ts)
    s = compute(df, {"window_d": 5})
    assert s.iloc[-1] < 0, "price rally → negative reversal"


def test_output_index_matches_input():
    df = _daily_df()
    s = compute(df, {})
    assert s.index.equals(df.index)


def test_missing_column_returns_empty():
    df = pd.DataFrame({"volume": [1.0, 2.0]})
    assert compute(df, {}).empty


def test_no_mutation_of_input():
    df = _daily_df()
    cols = list(df.columns)
    compute(df, {})
    assert list(df.columns) == cols
