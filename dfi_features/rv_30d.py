"""Realized volatility: 30-day rolling annualized std of daily log-returns."""
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    window_d = int(params.get("window_d", 30))
    if "close" not in df.columns:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(window_d, min_periods=window_d).std() * np.sqrt(365)
    return rv.rename("rv_30d")
