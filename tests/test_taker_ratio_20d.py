import numpy as np
import pandas as pd
from dfi_features.taker_ratio_20d import compute


def _daily_df(n=60, seed=42):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    volume = rng.uniform(1000, 5000, n)
    taker  = volume * rng.uniform(0.3, 0.7, n)
    return pd.DataFrame({"volume": volume, "taker_buy_vol": taker}, index=ts)


def test_warmup_is_nan():
    df = _daily_df()
    s = compute(df, {"window_d": 20})
    assert s.iloc[:19].isna().all(), "first 19 bars must be NaN"
    assert not pd.isna(s.iloc[19]), "bar 20 must be valid"


def test_output_is_float_series():
    df = _daily_df()
    s = compute(df, {})
    assert isinstance(s, pd.Series)
    assert s.dtype == float


def test_output_index_matches_input():
    df = _daily_df()
    s = compute(df, {})
    assert s.index.equals(df.index)


def test_high_taker_ratio_gives_positive():
    """Day with taker ratio far above recent mean → positive z-score."""
    n = 40
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    volume = np.ones(n) * 1000.0
    taker  = np.ones(n) * 500.0   # 50% ratio normally
    taker[-1] = 950.0              # spike to 95%
    df = pd.DataFrame({"volume": volume, "taker_buy_vol": taker}, index=ts)
    s = compute(df, {"window_d": 20})
    assert s.iloc[-1] > 0, "taker ratio spike → positive z-score"


def test_missing_columns_returns_empty():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    assert compute(df, {}).empty


def test_missing_taker_returns_empty():
    df = pd.DataFrame({"volume": [1.0, 2.0]})
    assert compute(df, {}).empty


def test_no_mutation_of_input():
    df = _daily_df()
    cols = list(df.columns)
    compute(df, {})
    assert list(df.columns) == cols
