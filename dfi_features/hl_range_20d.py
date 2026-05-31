"""20-day rolling high-low range ratio.

Measures average daily price range (high - low) / close over 20 days,
then z-scores it over a 252-day window. High values indicate elevated
intraday volatility — a microstructure activity signal.

Requires: high, low, close columns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    if not {"high", "low", "close"}.issubset(df.columns):
        return pd.Series(dtype=float)

    window:      int = params.get("window_d",   20)
    window_norm: int = params.get("window_norm", 252)

    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    daily_range = (high - low) / close.replace(0.0, np.nan)
    roll_range  = daily_range.rolling(window, min_periods=window).mean()

    mu    = roll_range.rolling(window_norm, min_periods=window_norm).mean()
    sigma = roll_range.rolling(window_norm, min_periods=window_norm).std()

    zscore = (roll_range - mu) / sigma.replace(0.0, np.nan)
    zscore.name = "value"
    return zscore
