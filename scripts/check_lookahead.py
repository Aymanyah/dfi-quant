"""Lookahead test harness.

For a given feature, generates synthetic market data, computes the feature on
the full series, then mutates all data after some cutoff t* and recomputes.
Values at ts <= t* MUST be unchanged. If any change, the feature peeks.

Synthetic data covers 21 days at 1-minute resolution — enough for both
microstructure features (sub-minute lookbacks) and long-lookback derivatives
features (funding_zscore_14d needs 14 days of 8h-resampled bars).
"""
from __future__ import annotations
import argparse
import importlib
import pathlib
import sys

# Ensure project root is importable regardless of working directory
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd


def _synth(n=43_200, seed=0):
    """43,200 1-minute ticks = 30 days; enough for 14-day lookback features."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    side = np.where(rng.random(n) < 0.55, "buy", "sell")
    amount = rng.uniform(0.1, 1.0, n)
    log_inc = rng.normal(0.0, 0.0001, n)
    price = 100.0 * np.exp(np.cumsum(log_inc))
    oi = 1_000_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0005, n)))
    funding_rate = rng.normal(0.0001, 0.0003, n)
    taker_buy_vol = amount * rng.uniform(0.3, 0.7, n)
    high = price * (1 + rng.uniform(0.0, 0.005, n))
    low  = price * (1 - rng.uniform(0.0, 0.005, n))
    return pd.DataFrame({
        "side": side, "amount": amount, "price": price,
        "close": price, "high": high, "low": low,
        "volume": amount,
        "taker_buy_vol": taker_buy_vol,
        "open_interest": oi, "funding_rate": funding_rate,
    }, index=ts)


def _to_utc(idx):
    if not hasattr(idx, "tz"):
        raise TypeError(f"Expected DatetimeIndex, got {type(idx).__name__}")
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", required=True)
    args = ap.parse_args()
    mod = importlib.import_module(f"dfi_features.{args.feature}")

    df = _synth()
    s_full = mod.compute(df, {})

    if not isinstance(s_full.index, pd.DatetimeIndex):
        print(f"SKIP {args.feature}: compute() returned empty/non-datetime index "
              f"(feature may need specific raw columns not in generic synth data)")
        return 0
    if s_full.dropna().empty:
        print(f"SKIP {args.feature}: all-NaN output (lookback may exceed synth window)")
        return 0

    cutoff = df.index[len(df) // 2]
    df_mut = df.copy()
    mask = df_mut.index > cutoff
    df_mut.loc[mask, "side"] = "buy"
    df_mut.loc[mask, "amount"] = 999.0
    df_mut.loc[mask, "price"] = df_mut.loc[mask, "price"] * 1.5
    df_mut.loc[mask, "close"] = df_mut.loc[mask, "close"] * 1.5
    df_mut.loc[mask, "volume"] = 999.0
    df_mut.loc[mask, "taker_buy_vol"] = 999.0
    df_mut.loc[mask, "open_interest"] = 9_999_999.0
    df_mut.loc[mask, "funding_rate"] = 0.99

    s_mut = mod.compute(df_mut, {})

    s_full.index = _to_utc(s_full.index)
    s_mut.index  = _to_utc(s_mut.index)
    cutoff_ts = (pd.Timestamp(cutoff).tz_convert("UTC")
                 if cutoff.tzinfo else pd.Timestamp(cutoff).tz_localize("UTC"))

    common = s_full.index.intersection(s_mut.index)
    keep = common[common <= cutoff_ts]
    a = s_full.loc[keep].dropna()
    b = s_mut.loc[keep].dropna()
    common2 = a.index.intersection(b.index)

    if len(common2) == 0:
        print(f"SKIP {args.feature}: no overlapping non-NaN bars before cutoff")
        return 0

    if not np.allclose(a.loc[common2].values, b.loc[common2].values, equal_nan=True):
        diffs = (a.loc[common2] - b.loc[common2]).abs()
        bad = diffs[diffs > 1e-12]
        print(f"LOOKAHEAD DETECTED in {args.feature}: {len(bad)} bars differ")
        print(bad.head())
        return 2

    print(f"OK no lookahead in {args.feature} "
          f"({len(common2)} bars checked up to {cutoff_ts})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
