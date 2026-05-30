import numpy as np
import pandas as pd
from dfi_features.cvd_20d import compute


def _daily_df(n=40, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    volume = rng.uniform(1000, 2000, n)
    taker_buy_vol = volume * rng.uniform(0.3, 0.7, n)
    return pd.DataFrame({"volume": volume, "taker_buy_vol": taker_buy_vol}, index=ts)


def _const_df(n, vol, tbv):
    ts = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"volume": [vol] * n, "taker_buy_vol": [tbv] * n}, index=ts)


def test_warmup_is_nan():
    df = _daily_df()
    s = compute(df, {"window_d": 20})
    assert s.iloc[:19].isna().all(), "first 19 bars must be NaN (warmup)"
    assert not pd.isna(s.iloc[19]), "bar 20 must be valid"


def test_all_buys_positive():
    df = _const_df(30, vol=100.0, tbv=100.0)
    s = compute(df, {"window_d": 20})
    assert (s.dropna() > 0).all(), "all-buy bars => positive CVD"


def test_all_sells_negative():
    df = _const_df(30, vol=100.0, tbv=0.0)
    s = compute(df, {"window_d": 20})
    assert (s.dropna() < 0).all(), "all-sell bars => negative CVD"


def test_balanced_near_zero():
    df = _const_df(30, vol=100.0, tbv=50.0)
    s = compute(df, {"window_d": 20})
    assert (s.dropna().abs() < 1e-9).all(), "balanced buy/sell => CVD ~ 0"


def test_missing_column_returns_empty():
    df = pd.DataFrame({"price": [1.0, 2.0]})
    s = compute(df, {})
    assert s.empty
