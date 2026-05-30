"""Z-score composite of daily simple returns.

Computes a z-score for each window in `params['windows']` (rolling or EWM),
then returns their cross-window mean as a single composite signal.

The composite averages out noise that is specific to any one window length
while retaining the directional signal that is persistent across scales.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _zscore(series: pd.Series, window: int, method: str) -> pd.Series:
    if method == "ewm":
        mu = series.ewm(span=window, min_periods=window).mean()
        sigma = series.ewm(span=window, min_periods=window).std()
    else:
        mu = series.rolling(window, min_periods=window).mean()
        sigma = series.rolling(window, min_periods=window).std()
    return (series - mu) / sigma.replace(0.0, np.nan)


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series(dtype=float)

    windows: list[int] = params.get("windows", [15, 30, 60, 80, 100])
    method: str = params.get("method", "rolling")

    close = df["close"].astype(float)
    ret = close / close.shift(1) - 1

    zscores = [_zscore(ret, w, method) for w in windows]
    composite = pd.concat(zscores, axis=1).mean(axis=1, skipna=False)
    composite.name = "value"
    return composite
