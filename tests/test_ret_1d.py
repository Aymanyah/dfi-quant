import numpy as np
import pandas as pd
from dfi_features.ret_1d import compute


def _daily_df(closes):
    ts = pd.date_range("2023-01-01", periods=len(closes), freq="1D", tz="UTC")
    return pd.DataFrame({"close": closes}, index=ts)


def test_first_bar_is_nan():
    df = _daily_df([100.0, 110.0, 105.0])
    s = compute(df, {})
    assert pd.isna(s.iloc[0]), "premier bar doit être NaN (pas de close précédent)"


def test_positive_return_when_price_rises():
    df = _daily_df([100.0, 110.0])
    s = compute(df, {})
    assert s.iloc[1] > 0, "prix monte => rendement positif"


def test_negative_return_when_price_falls():
    df = _daily_df([100.0, 90.0])
    s = compute(df, {})
    assert s.iloc[1] < 0, "prix baisse => rendement négatif"


def test_log_return_value():
    df = _daily_df([100.0, 110.0])
    s = compute(df, {})
    assert abs(s.iloc[1] - np.log(110.0 / 100.0)) < 1e-10


def test_missing_column_returns_empty():
    df = pd.DataFrame({"price": [1.0, 2.0]})
    assert compute(df, {}).empty
