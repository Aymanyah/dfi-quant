"""Z-scored volume ratio over 20 days.

Measures whether today's volume is unusually high or low relative to
the recent 20-day average, then z-scores that ratio over a 252-day
rolling window to make it comparable across assets and time.

A spike in volume often precedes or accompanies large price moves.
Positive = unusually high activity, negative = unusually quiet.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(dtype=float)

    window:      int = params.get("window_d",   20)
    window_norm: int = params.get("window_norm", 252)

    vol = df["volume"].astype(float)

    avg_vol = vol.rolling(window, min_periods=window).mean()
    ratio   = vol / avg_vol.replace(0.0, np.nan)

    mu    = ratio.rolling(window_norm, min_periods=window_norm).mean()
    sigma = ratio.rolling(window_norm, min_periods=window_norm).std()

    zscore = (ratio - mu) / sigma.replace(0.0, np.nan)
    zscore.name = "value"
    return zscore
