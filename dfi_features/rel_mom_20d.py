"""Z-scored 20-day momentum.

Computes the 20-day cumulative simple return, then normalises it by its
own 252-day rolling mean and std. This makes the signal comparable across
assets and time — an asset with a z-score of +2 has unusually strong
momentum relative to its own history.

True cross-sectional demeaning (asset minus universe mean) cannot be done
inside compute() since it receives one asset at a time. Z-scoring achieves
the same effect for IC ranking purposes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series(dtype=float)

    window_mom: int = params.get("window_mom", 20)
    window_norm: int = params.get("window_norm", 252)

    close = df["close"].astype(float)
    ret_20d = close / close.shift(window_mom) - 1

    mu    = ret_20d.rolling(window_norm, min_periods=window_norm).mean()
    sigma = ret_20d.rolling(window_norm, min_periods=window_norm).std()

    zscore = (ret_20d - mu) / sigma.replace(0.0, np.nan)
    zscore.name = "value"
    return zscore
