"""Backtest / IC agent — CLI entry point.

Usage:
    python -m agents.backtest_ic.runner --features mom_20d rv_30d cvd_20d
    python -m agents.backtest_ic.runner --features zscore_ret_composite
    python -m agents.backtest_ic.runner --no-ortho
    python -m agents.backtest_ic.runner  # runs all ohlcv_1d features

Outputs (per feature):
    reports/ic/<feature_id>.json   — IC / ICIR / regime stats, raw + ortho
    reports/ic/<feature_id>.png    — 6-panel plot
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.backtest_ic.ic_engine import (
    orthogonalize,
    split_regimes,
    ic_by_regime,
)
from agents.backtest_ic.report import save_json, save_plots

RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "ic"

HORIZONS_BY_FAMILY = {
    "returns":        [1, 3, 5, 7],
    "volatility":     [1, 3, 5, 7],
    "momentum":       [1, 3, 5, 7],
    "derivatives":    [1, 3, 5],
    "microstructure": [1, 3],
}
DEFAULT_HORIZONS = [1, 3, 5, 7]
ORTHO_CONTROLS = ["mom_20d", "rv_30d"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ohlcv(symbol: str, exchange: str = "binance-futures") -> pd.DataFrame:
    base = RAW / f"exchange={exchange}" / "data_type=ohlcv_1d" / f"symbol={symbol}"
    parts = sorted(base.rglob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["date"] = pd.to_datetime(df["ts_open"], unit="us", utc=True).dt.normalize()
    df = df.sort_values("date").set_index("date")
    df["log_ret"] = np.log(df["close"].astype(float) / df["close"].astype(float).shift(1))
    return df


def load_features_registry() -> list[dict]:
    with open(ROOT / "configs" / "features.yaml") as f:
        return yaml.safe_load(f)["features"]


def load_universe() -> list[dict]:
    with open(ROOT / "configs" / "universe.yaml") as f:
        return yaml.safe_load(f)["assets"]


# ── Feature computation ───────────────────────────────────────────────────────

def _compute_feature(feature_cfg: dict, df: pd.DataFrame) -> pd.Series:
    import importlib
    try:
        mod = importlib.import_module(f"dfi_features.{feature_cfg['id']}")
        return mod.compute(df, feature_cfg.get("params", {}))
    except Exception:
        return pd.Series(dtype=float)


def _compute_controls(
    asset: str,
    prices: dict[str, pd.DataFrame],
    registry: list[dict],
    exclude_id: str,
) -> pd.DataFrame:
    """Build a DataFrame of control feature values for one asset."""
    cols = {}
    for ctrl_id in ORTHO_CONTROLS:
        if ctrl_id == exclude_id:
            continue
        cfg = next((f for f in registry if f["id"] == ctrl_id), None)
        if cfg is None:
            continue
        sig = _compute_feature(cfg, prices[asset])
        if not sig.empty:
            cols[ctrl_id] = sig
    return pd.DataFrame(cols)


# ── IC time-series computation ────────────────────────────────────────────────

def _compute_ic_ts(
    signals: dict[str, pd.Series],
    prices: dict[str, pd.DataFrame],
    horizons: list[int],
) -> dict[int, pd.Series]:
    """Cross-asset IC at each date for each horizon."""
    all_dates = sorted(set.union(*[set(s.index) for s in signals.values()]))
    ic_ts: dict[int, pd.Series] = {}

    for h in horizons:
        daily: list[tuple] = []
        for t in pd.DatetimeIndex(all_dates):
            xs, xf = [], []
            for asset, sig in signals.items():
                df = prices[asset]
                if t not in sig.index or t not in df.index:
                    continue
                sv = sig.loc[t]
                if pd.isna(sv):
                    continue
                loc = df.index.get_loc(t)
                if loc + h >= len(df):
                    continue
                fwd = float(df["log_ret"].iloc[loc + 1: loc + h + 1].sum())
                if pd.isna(fwd):
                    continue
                xs.append(sv)
                xf.append(fwd)
            ic = (
                pd.Series(xs).corr(pd.Series(xf), method="spearman")
                if len(xs) >= 3 else float("nan")
            )
            daily.append((t, ic))

        dates, vals = zip(*daily)
        ic_ts[h] = pd.Series(vals, index=pd.DatetimeIndex(dates), name=h)
        mean_ic = pd.Series(vals).mean()
        icir = (
            mean_ic / pd.Series(vals).std()
            if pd.Series(vals).std() > 0 else float("nan")
        )
        print(f"  h={h:2d}d  IC={mean_ic:+.4f}  ICIR={icir:+.3f}"
              f"  n={pd.Series(vals).notna().sum()}")

    return ic_ts


# ── Main feature runner ───────────────────────────────────────────────────────

def run_feature(
    feature_cfg: dict,
    assets: list[str],
    prices: dict[str, pd.DataFrame],
    horizons: list[int],
    registry: list[dict],
    do_ortho: bool,
) -> dict:
    fid = feature_cfg["id"]

    # ── Raw signals ───────────────────────────────────────────────────────────
    raw_signals: dict[str, pd.Series] = {}
    for asset in assets:
        if prices[asset].empty:
            continue
        sig = _compute_feature(feature_cfg, prices[asset])
        if sig.empty:
            print(f"  {asset}: empty (source non disponible)")
            continue
        raw_signals[asset] = sig

    if not raw_signals:
        print("  Aucun asset disponible — skip")
        return {}

    print(f"\n── {fid}  [raw] ──")
    raw_ic = _compute_ic_ts(raw_signals, prices, horizons)

    # ── Orthogonalized signals ────────────────────────────────────────────────
    ortho_ic: dict[int, pd.Series] = {}
    if do_ortho:
        ortho_signals: dict[str, pd.Series] = {}
        for asset, sig in raw_signals.items():
            controls = _compute_controls(asset, prices, registry, exclude_id=fid)
            ortho_signals[asset] = (
                orthogonalize(sig, controls) if not controls.empty else sig
            )
        print(f"\n── {fid}  [ortho: {ORTHO_CONTROLS}] ──")
        ortho_ic = _compute_ic_ts(ortho_signals, prices, horizons)

    # ── BTC regime split ──────────────────────────────────────────────────────
    btc_rets = prices.get("BTCUSDT", pd.DataFrame()).get(
        "log_ret", pd.Series(dtype=float)
    )
    regime_labels = (
        split_regimes(btc_rets) if not btc_rets.empty else pd.Series(dtype=str)
    )

    regime_by_horizon: dict[int, dict] = {}
    for h in horizons:
        aligned = raw_ic[h].reindex(regime_labels.index)
        regime_by_horizon[h] = ic_by_regime(aligned, regime_labels)

    return {
        "ic_ts":             raw_ic,
        "ic_ts_ortho":       ortho_ic,
        "regime_labels":     regime_labels,
        "regime_by_horizon": regime_by_horizon,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Backtest IC agent")
    ap.add_argument("--features", nargs="*",
                    help="Feature IDs to analyse (default: all ohlcv_1d features)")
    ap.add_argument("--horizons", nargs="*", type=int,
                    help="Forward horizons in days (default: per feature family)")
    ap.add_argument("--no-ortho", action="store_true",
                    help="Disable orthogonalization pass")
    args = ap.parse_args()

    registry = load_features_registry()
    universe = load_universe()
    assets = [a["symbol"] for a in universe]

    ohlcv_features = [f for f in registry if f.get("source") == "ohlcv_1d"]
    if args.features:
        ohlcv_features = [f for f in ohlcv_features if f["id"] in args.features]

    print(f"Assets   : {assets}")
    print(f"Features : {[f['id'] for f in ohlcv_features]}")

    print("\nChargement des données OHLCV...")
    prices: dict[str, pd.DataFrame] = {}
    for asset in assets:
        df = load_ohlcv(asset)
        if not df.empty:
            prices[asset] = df
            print(f"  {asset}: {len(df)} jours")

    for feat_cfg in ohlcv_features:
        fid = feat_cfg["id"]
        family = feat_cfg.get("family", "returns")
        horizons = args.horizons or HORIZONS_BY_FAMILY.get(family, DEFAULT_HORIZONS)

        result = run_feature(
            feat_cfg,
            list(prices.keys()),
            prices,
            horizons,
            registry,
            do_ortho=not args.no_ortho,
        )
        if not result:
            continue

        ic_ts        = result["ic_ts"]
        ic_ts_ortho  = result["ic_ts_ortho"]
        regime_labels        = result["regime_labels"]
        regime_by_horizon    = result["regime_by_horizon"]

        def _stats(ts_dict: dict[int, pd.Series]) -> tuple[dict, dict]:
            means = {h: float(s.mean()) for h, s in ts_dict.items()}
            icirs = {
                h: float(s.mean() / s.std()) if s.std() > 0 else float("nan")
                for h, s in ts_dict.items()
            }
            return means, icirs

        ic_mean,      icir      = _stats(ic_ts)
        ic_mean_ortho, icir_ortho = _stats(ic_ts_ortho) if ic_ts_ortho else ({}, {})

        json_path = save_json(
            fid, list(prices.keys()), horizons,
            ic_mean, icir,
            ic_mean_ortho, icir_ortho,
            regime_by_horizon, OUT,
        )
        plot_path = save_plots(
            fid, ic_ts, ic_ts_ortho, regime_labels, OUT
        )
        print(f"  → {json_path.relative_to(ROOT)}")
        print(f"  → {plot_path.relative_to(ROOT)}")

    print("\n── Terminé ──")


if __name__ == "__main__":
    main()
