"""Cumulative Volume Delta: 20-day rolling net buy pressure, normalized by volume."""
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    window_d = int(params.get("window_d", 20))
    if "taker_buy_vol" not in df.columns or "volume" not in df.columns:
        return pd.Series(dtype=float)
    vol = df["volume"].astype(float)
    tbv = df["taker_buy_vol"].astype(float)
    cvd_bar = 2 * tbv - vol
    cvd_roll = cvd_bar.rolling(window_d, min_periods=window_d).sum()
    avg_vol = vol.rolling(window_d, min_periods=window_d).mean()
    return (cvd_roll / avg_vol.replace(0, float("nan"))).rename("cvd_20d")
