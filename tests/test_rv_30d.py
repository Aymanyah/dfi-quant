import numpy as np
import pandas as pd
from dfi_features.rv_30d import compute


def _daily_df(n=60, seed=42):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"close": close}, index=ts)


def test_warmup_is_nan():
    df = _daily_df()
    s = compute(df, {"window_d": 30})
    # log_ret[0] is NaN from shift(1), so rolling(30) needs 31 rows to get 30 valid returns
    assert s.iloc[:30].isna().all(), "first 30 bars must be NaN (warmup)"
    assert not pd.isna(s.iloc[30]), "bar 31 must be valid"


def test_output_positive():
    df = _daily_df()
    s = compute(df, {"window_d": 30})
    assert (s.dropna() > 0).all(), "realized vol must be positive"


def test_higher_vol_raises_rv():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2023-01-01", periods=60, freq="1D", tz="UTC")
    low_vol  = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 60)))}, index=ts)
    high_vol = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.05,  60)))}, index=ts)
    s_low  = compute(low_vol,  {"window_d": 30}).dropna()
    s_high = compute(high_vol, {"window_d": 30}).dropna()
    assert s_high.mean() > s_low.mean(), "higher daily moves => higher RV"


def test_missing_column_returns_empty():
    df = pd.DataFrame({"price": [1.0, 2.0]})
    s = compute(df, {})
    assert s.empty
