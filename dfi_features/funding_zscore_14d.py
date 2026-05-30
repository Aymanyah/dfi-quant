"""Funding rate z-score: rolling z-score of the 8h funding rate.

Returns empty series when funding_rate column is absent (e.g. ohlcv_1d source).
"""
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    if "funding_rate" not in df.columns:
        return pd.Series(dtype=float)
    lookback = int(params.get("lookback_d", 14))
    fr = df["funding_rate"].astype(float)
    mu = fr.rolling(lookback, min_periods=lookback).mean()
    sd = fr.rolling(lookback, min_periods=lookback).std()
    zscore = (fr - mu) / sd.replace(0, float("nan"))
    return zscore.rename("funding_zscore_14d")
