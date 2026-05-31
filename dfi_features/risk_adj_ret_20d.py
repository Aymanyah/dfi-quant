"""Risk-adjusted 20-day return (Sharpe-like).

Computes the 20-day cumulative simple return divided by the 20-day
realized volatility. Assets with high returns relative to their own
volatility rank higher — rewards consistent movers over volatile ones.

The raw ratio is then z-scored over a 252-day rolling window to make
it comparable across assets and time.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series(dtype=float)

    window_ret:  int = params.get("window_ret",  20)
    window_norm: int = params.get("window_norm", 252)

    close   = df["close"].astype(float)
    log_ret = np.log(close / close.shift(1))

    ret_20d = close / close.shift(window_ret) - 1
    rv_20d  = log_ret.rolling(window_ret, min_periods=window_ret).std() * np.sqrt(252)

    ratio = ret_20d / rv_20d.replace(0.0, np.nan)

    mu    = ratio.rolling(window_norm, min_periods=window_norm).mean()
    sigma = ratio.rolling(window_norm, min_periods=window_norm).std()

    zscore = (ratio - mu) / sigma.replace(0.0, np.nan)
    zscore.name = "value"
    return zscore
