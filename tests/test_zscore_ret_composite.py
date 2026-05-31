import numpy as np
import pandas as pd
import pytest
from dfi_features.zscore_ret_composite import compute


def _daily_df(n=150, seed=42, vol=0.02):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2021-01-01", periods=n, freq="1D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    return pd.DataFrame({"close": close}, index=ts)


# ── Warmup ────────────────────────────────────────────────────────────────────

def test_warmup_is_nan():
    """First max(windows)-1 values must be NaN — largest window drives warmup."""
    df = _daily_df(n=350)
    windows = [15, 30, 60, 80, 100, 150, 200, 250, 300]
    s = compute(df, {"windows": windows, "method": "rolling"})
    assert s.iloc[:300].isna().all(), "first 300 bars must be NaN (warmup)"
    assert not pd.isna(s.iloc[300]), "bar 301 must be valid"


def test_warmup_single_window():
    """With a single small window the warmup is shorter."""
    df = _daily_df()
    s = compute(df, {"windows": [15], "method": "rolling"})
    assert s.iloc[:15].isna().all()
    assert not pd.isna(s.iloc[15])


# ── Output properties ─────────────────────────────────────────────────────────

def test_output_is_float_series():
    df = _daily_df()
    s = compute(df, {})
    assert isinstance(s, pd.Series)
    assert s.dtype == float


def test_output_index_matches_input():
    df = _daily_df()
    s = compute(df, {})
    assert s.index.equals(df.index)


def test_output_is_bounded():
    """Z-scores on normal data should sit well within [-5, 5]."""
    df = _daily_df(n=300)
    s = compute(df, {}).dropna()
    assert s.abs().max() < 10, "z-score should not explode on normal returns"


# ── Sign / magnitude sanity ───────────────────────────────────────────────────

def test_strong_positive_return_gives_positive_zscore():
    """A day with a return far above the rolling mean → composite > 0."""
    rng = np.random.default_rng(0)
    n = 150
    ts = pd.date_range("2021-01-01", periods=n, freq="1D", tz="UTC")
    # flat prices then one big up day at the end
    close = np.ones(n) * 100.0
    close[-1] = 120.0
    df = pd.DataFrame({"close": close}, index=ts)
    s = compute(df, {"windows": [15, 30], "method": "rolling"})
    assert s.iloc[-1] > 0, "large positive return → composite z-score > 0"


def test_strong_negative_return_gives_negative_zscore():
    """A day with a return far below the rolling mean → composite < 0."""
    n = 150
    ts = pd.date_range("2021-01-01", periods=n, freq="1D", tz="UTC")
    close = np.ones(n) * 100.0
    close[-1] = 80.0
    df = pd.DataFrame({"close": close}, index=ts)
    s = compute(df, {"windows": [15, 30], "method": "rolling"})
    assert s.iloc[-1] < 0, "large negative return → composite z-score < 0"


# ── EWM method ────────────────────────────────────────────────────────────────

def test_ewm_method_runs():
    """EWM method should produce valid output without errors."""
    df = _daily_df()
    s = compute(df, {"windows": [15, 30], "method": "ewm"})
    assert not s.dropna().empty


def test_ewm_warmup_is_nan():
    df = _daily_df()
    s = compute(df, {"windows": [15, 30], "method": "ewm"})
    assert s.iloc[:30].isna().all()


# ── Numerical correctness ─────────────────────────────────────────────────────

def test_value_matches_formula():
    """Manual rolling z-score computation must match compute()."""
    df = _daily_df(n=60)
    windows = [15, 20]
    s = compute(df, {"windows": windows, "method": "rolling"})

    close = df["close"].astype(float)
    ret = close / close.shift(1) - 1

    zscores = []
    for w in windows:
        mu = ret.rolling(w, min_periods=w).mean()
        sigma = ret.rolling(w, min_periods=w).std().replace(0.0, np.nan)
        zscores.append((ret - mu) / sigma)

    expected = pd.concat(zscores, axis=1).mean(axis=1, skipna=False)
    pd.testing.assert_series_equal(s, expected.rename("value"), check_names=False)


# ── Graceful degradation ──────────────────────────────────────────────────────

def test_missing_column_returns_empty():
    df = pd.DataFrame({"volume": [1.0, 2.0, 3.0]})
    s = compute(df, {})
    assert s.empty


def test_empty_dataframe_returns_empty():
    df = pd.DataFrame({"close": pd.Series(dtype=float)})
    s = compute(df, {})
    assert s.empty


def test_no_mutation_of_input():
    """compute() must not modify the input DataFrame."""
    df = _daily_df()
    cols_before = list(df.columns)
    compute(df, {})
    assert list(df.columns) == cols_before, "compute() must not add columns to df"
