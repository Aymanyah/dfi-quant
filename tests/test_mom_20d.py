import numpy as np
import pandas as pd
from dfi_features.mom_20d import compute


def _daily_df(n=40, growth=0.01, seed=None):
    ts = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    if seed is not None:
        rng = np.random.default_rng(seed)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    else:
        close = 100 * np.exp(np.arange(n) * growth)
    return pd.DataFrame({"close": close}, index=ts)


def test_warmup_is_nan():
    df = _daily_df(n=40)
    s = compute(df, {"window_d": 20})
    # log_ret[0]=NaN, rolling(20) needs 20 valid returns → first valid at index 20
    assert s.iloc[:20].isna().all(), "20 premiers bars doivent être NaN"
    assert not pd.isna(s.iloc[20]), "bar 21 doit être valide"


def test_positive_momentum_uptrend():
    df = _daily_df(n=40, growth=0.02)
    s = compute(df, {"window_d": 20})
    assert (s.dropna() > 0).all(), "prix qui montent => momentum positif"


def test_negative_momentum_downtrend():
    df = _daily_df(n=40, growth=-0.02)
    s = compute(df, {"window_d": 20})
    assert (s.dropna() < 0).all(), "prix qui baissent => momentum négatif"


def test_equivalent_to_log_ratio():
    df = _daily_df(n=40, seed=42)
    s = compute(df, {"window_d": 20})
    close = df["close"]
    expected = np.log(close / close.shift(20))
    valid = s.dropna()
    exp_valid = expected.loc[valid.index]
    assert np.allclose(valid.values, exp_valid.values, atol=1e-10)


def test_missing_column_returns_empty():
    df = pd.DataFrame({"price": [1.0, 2.0]})
    assert compute(df, {}).empty
