import numpy as np
import pandas as pd
from dfi_features.risk_adj_ret_20d import compute


def _daily_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.DataFrame({"close": close}, index=ts)


def test_warmup_is_nan():
    df = _daily_df(n=300)
    s = compute(df, {"window_ret": 20, "window_norm": 252})
    assert s.iloc[:271].isna().all()
    assert not pd.isna(s.iloc[271])


def test_output_is_float_series():
    df = _daily_df()
    assert isinstance(compute(df, {}), pd.Series)
    assert compute(df, {}).dtype == float


def test_output_index_matches_input():
    df = _daily_df()
    assert compute(df, {}).index.equals(df.index)


def test_missing_column_returns_empty():
    df = pd.DataFrame({"volume": [1.0, 2.0]})
    assert compute(df, {}).empty


def test_no_mutation_of_input():
    df = _daily_df()
    cols = list(df.columns)
    compute(df, {})
    assert list(df.columns) == cols
