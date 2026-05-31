import numpy as np
import pandas as pd
from dfi_features.trades_zscore_20d import compute


def _daily_df(n=60, seed=42):
    rng     = np.random.default_rng(seed)
    ts      = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    n_trades = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame({"n_trades": n_trades}, index=ts)


def test_warmup_is_nan():
    df = _daily_df()
    s  = compute(df, {"window_d": 20})
    assert s.iloc[:19].isna().all()
    assert not pd.isna(s.iloc[19])


def test_output_is_float_series():
    assert compute(_daily_df(), {}).dtype == float


def test_output_index_matches_input():
    df = _daily_df()
    assert compute(df, {}).index.equals(df.index)


def test_trades_spike_gives_positive():
    n        = 40
    ts       = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    n_trades = np.ones(n) * 1000.0
    n_trades[-1] = 20000.0
    df = pd.DataFrame({"n_trades": n_trades}, index=ts)
    s  = compute(df, {"window_d": 20})
    assert s.iloc[-1] > 0


def test_missing_column_returns_empty():
    assert compute(pd.DataFrame({"volume": [1.0]}), {}).empty


def test_no_mutation_of_input():
    df   = _daily_df()
    cols = list(df.columns)
    compute(df, {})
    assert list(df.columns) == cols
