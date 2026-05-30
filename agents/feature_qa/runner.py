"""Feature QA agent — CLI entry point.

Runs statistical sanity checks on every ohlcv_1d feature and writes
a Markdown report per feature to reports/qa/.

Usage:
    python -m agents.feature_qa.runner
    python -m agents.feature_qa.runner --features mom_20d cvd_20d
"""
from __future__ import annotations
import argparse
import importlib
import pathlib
import sys
import datetime as dt

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.feature_qa.checks import (
    check_distribution,
    check_coverage,
    check_lookahead,
    check_explosion,
)

RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "qa"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ohlcv(symbol: str, exchange: str = "binance-futures") -> pd.DataFrame:
    base = RAW / f"exchange={exchange}" / "data_type=ohlcv_1d" / f"symbol={symbol}"
    parts = sorted(base.rglob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["date"] = pd.to_datetime(df["ts_open"], unit="us", utc=True).dt.normalize()
    return df.sort_values("date").set_index("date")


def load_registry() -> list[dict]:
    with open(ROOT / "configs" / "features.yaml") as f:
        return yaml.safe_load(f)["features"]


def load_universe() -> list[str]:
    with open(ROOT / "configs" / "universe.yaml") as f:
        return [a["symbol"] for a in yaml.safe_load(f)["assets"]]


# ── Markdown report ───────────────────────────────────────────────────────────

def _flag(passed: bool) -> str:
    return "✅ PASS" if passed else "❌ FAIL"


def write_report(
    feature_id: str,
    family: str,
    results: dict,          # {check_name: {passed, message, detail}}
    out_dir: pathlib.Path,
) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# QA Report — `{feature_id}`",
        f"",
        f"**Family:** {family}  ",
        f"**Date:** {dt.date.today().isoformat()}",
        f"",
        f"## Summary",
        f"",
        f"| Check | Status | Message |",
        f"|---|---|---|",
    ]

    for check_name, res in results.items():
        lines.append(f"| {check_name} | {_flag(res['passed'])} | {res['message']} |")

    overall = all(r["passed"] for r in results.values())
    lines += [
        f"",
        f"**Overall: {_flag(overall)}**",
        f"",
    ]

    # Detail sections
    for check_name, res in results.items():
        lines += [f"## {check_name}", f""]
        detail = res.get("detail", {})
        if isinstance(detail, dict):
            for k, v in detail.items():
                if isinstance(v, dict):
                    lines.append(f"**{k}:**")
                    for kk, vv in v.items():
                        lines.append(f"- {kk}: {vv}")
                elif isinstance(v, list):
                    if v:
                        lines.append(f"**{k}:** {v}")
                else:
                    lines.append(f"- **{k}:** {v}")
        lines.append("")

    path = out_dir / f"{feature_id}.md"
    path.write_text("\n".join(lines))
    return path


# ── Main QA loop ──────────────────────────────────────────────────────────────

def run_qa(feature_cfg: dict, assets: list[str], prices: dict) -> dict:
    fid = feature_cfg["id"]
    params = feature_cfg.get("params", {})
    warmup = int(params.get("window_d", params.get("lookback_d", 1)))

    mod = importlib.import_module(f"dfi_features.{fid}")
    compute = mod.compute

    # Compute feature for all assets
    series_by_asset: dict[str, pd.Series] = {}
    for asset in assets:
        df = prices.get(asset, pd.DataFrame())
        if df.empty:
            series_by_asset[asset] = pd.Series(dtype=float)
            continue
        s = compute(df, params)
        series_by_asset[asset] = s

    # Pick BTC as reference for single-series checks
    ref_series = series_by_asset.get("BTCUSDT", pd.Series(dtype=float))
    ref_df = prices.get("BTCUSDT", pd.DataFrame())

    results = {}

    # 1. Distribution
    print(f"  distribution...", end=" ", flush=True)
    results["Distribution"] = check_distribution(ref_series)
    print(_flag(results["Distribution"]["passed"]))

    # 2. Coverage
    print(f"  coverage...", end=" ", flush=True)
    results["Coverage"] = check_coverage(series_by_asset, warmup_bars=warmup)
    print(_flag(results["Coverage"]["passed"]))

    # 3. Lookahead
    print(f"  lookahead...", end=" ", flush=True)
    if not ref_df.empty:
        results["Lookahead"] = check_lookahead(compute, params, ref_df)
    else:
        results["Lookahead"] = {"passed": True, "message": "Pas de données BTCUSDT", "detail": {}}
    print(_flag(results["Lookahead"]["passed"]))

    # 4. Explosion
    print(f"  explosion...", end=" ", flush=True)
    results["Explosion"] = check_explosion(ref_series)
    print(_flag(results["Explosion"]["passed"]))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="*")
    args = ap.parse_args()

    registry = load_registry()
    assets = load_universe()

    ohlcv_features = [f for f in registry if f.get("source") == "ohlcv_1d"]
    if args.features:
        ohlcv_features = [f for f in ohlcv_features if f["id"] in args.features]

    print(f"Assets  : {assets}")
    print(f"Features: {[f['id'] for f in ohlcv_features]}\n")

    print("Chargement OHLCV...")
    prices = {}
    for asset in assets:
        df = load_ohlcv(asset)
        if not df.empty:
            prices[asset] = df
            print(f"  {asset}: {len(df)} jours")

    all_passed = True
    for feat_cfg in ohlcv_features:
        fid = feat_cfg["id"]
        print(f"\n── {fid} ──")
        results = run_qa(feat_cfg, assets, prices)
        path = write_report(fid, feat_cfg.get("family", "?"), results, OUT)
        feature_passed = all(r["passed"] for r in results.values())
        all_passed = all_passed and feature_passed
        print(f"  → {path.relative_to(ROOT)}  [{_flag(feature_passed)}]")

    print(f"\n── Résultat global : {_flag(all_passed)} ──\n")


if __name__ == "__main__":
    main()
