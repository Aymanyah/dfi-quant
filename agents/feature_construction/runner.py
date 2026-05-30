"""Feature construction runner.

Discovers features declared in configs/features.yaml, loads their raw
inputs from data/raw/, calls dfi_features.<feature_id>.compute() and writes
the result to data/features/<...>/_SUCCESS.

Usage:
    python -m agents.feature_construction.runner \
        --from 2024-03-01 --to 2024-03-01 \
        --features trade_imbalance_5m --symbols BTCUSDT
"""
from __future__ import annotations
import argparse
import importlib
import os
import sys
import datetime as dt
import yaml
import pathlib
import uuid
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def _daterange(d0, d1):
    cur = d0
    while cur <= d1:
        yield cur
        cur += dt.timedelta(days=1)


def _raw_path(root, exchange, data_type, symbol, day):
    return (pathlib.Path(root) / "raw" /
            f"exchange={exchange}" / f"data_type={data_type}" /
            f"symbol={symbol}" / f"year={day:%Y}" /
            f"month={day:%m}" / f"day={day:%d}")


def _feat_path(root, family, fid, version, asset, day):
    return (pathlib.Path(root) / "features" /
            f"feature_family={family}" / f"feature_id={fid}" /
            f"version={version}" / f"asset={asset}" /
            f"year={day:%Y}" / f"month={day:%m}" / f"day={day:%d}")


def _success(p): return (p / "_SUCCESS").exists()


def _atomic_write(out_dir: pathlib.Path, table: pa.Table):
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_dir.parent / f"_tmp_{uuid.uuid4().hex}"
    tmp.mkdir()
    pq.write_table(table, tmp / "part-00000.parquet",
                   compression="zstd", use_dictionary=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    (out_dir / "_SUCCESS").touch()


def _load_raw(raw_root, source, exchange, symbol, d0, d1):
    """Load raw parquet for [d0, d1] inclusive, normalised to ts_utc index."""
    frames = []
    for day in _daterange(d0, d1):
        p = _raw_path(raw_root, exchange, source, symbol, day)
        for f in sorted(p.glob("*.parquet")):
            frames.append(pd.read_parquet(f))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if "ts_utc" not in df.columns:
        for col in ("ts", "timestamp", "ts_open"):
            if col in df.columns:
                df["ts_utc"] = pd.to_datetime(df[col], unit="us", utc=True)
                break
    if "ts_utc" not in df.columns:
        return None
    return df.sort_values("ts_utc").set_index("ts_utc")


def _extra_days(feat: dict) -> int:
    """Days of raw history to load before the target day for warmup."""
    params = feat.get("params", {})
    if "window_d" in params:
        return int(params["window_d"])
    if "lookback_d" in params:
        return int(params["lookback_d"])
    if "window" in params:
        w = str(params["window"]).strip().lower()
        if w.endswith("d"):
            return int(w[:-1])
        if w.endswith("h"):
            return 1
    if "window_min" in params:
        # sub-day windows need at most 1 previous day for cross-midnight warmup
        return max(1, int(params["window_min"]) // (24 * 60))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", required=True)
    ap.add_argument("--to",   dest="d_to",   required=True)
    ap.add_argument("--features", default="")
    ap.add_argument("--symbols",  default="")
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()

    settings = _load_yaml(ROOT / "configs" / "settings.yaml")
    features = _load_yaml(ROOT / "configs" / "features.yaml")["features"]
    universe = _load_yaml(ROOT / "configs" / "universe.yaml")["assets"]

    if args.features:
        wanted = set(args.features.split(","))
        features = [f for f in features if f["id"] in wanted]
    if args.symbols:
        wsym = set(args.symbols.split(","))
        universe = [a for a in universe if a["symbol"] in wsym]

    data_root = pathlib.Path(os.path.expanduser(
        settings["storage"]["local_cache"]))

    d0 = dt.date.fromisoformat(args.d_from)
    d1 = dt.date.fromisoformat(args.d_to)

    plan = []
    for feat in features:
        for asset in universe:
            for day in _daterange(d0, d1):
                out = _feat_path(data_root, feat.get("family", "misc"),
                                 feat["id"], feat["version"],
                                 asset["symbol"], day)
                if _success(out):
                    continue
                plan.append((feat, asset, day, out))

    print(f"Planned {len(plan)} feature partitions")
    if args.dry_run:
        for feat, asset, day, out in plan[:20]:
            print(f"  {feat['id']} v{feat['version']} {asset['symbol']} {day}")
        return 0

    for feat, asset, day, out in plan:
        mod = importlib.import_module(f"dfi_features.{feat['id']}")
        extra = _extra_days(feat)
        d_load = day - dt.timedelta(days=extra)
        df = _load_raw(data_root, feat["source"], asset["exchange"],
                       asset["symbol"], d_load, day)
        if df is None:
            print(f"  skip {feat['id']} {asset['symbol']} {day} : no raw")
            continue

        series = mod.compute(df, feat.get("params", {}))

        # Keep only target day — compute() may have used prior days for warmup
        day_start = pd.Timestamp(day, tz="UTC")
        day_end   = day_start + pd.Timedelta(days=1)
        series = series[(series.index >= day_start) & (series.index < day_end)]
        series = series.rename("value").dropna()

        if len(series) == 0:
            print(f"  skip {feat['id']} {asset['symbol']} {day} : no output rows")
            continue

        out_df = pd.DataFrame({
            "ts":         (series.index.astype("int64") // 1000),  # microseconds UTC
            "feature_id": feat["id"],
            "version":    feat["version"],
            "asset":      asset["symbol"],
            "value":      series.values.astype("float64"),
        })
        table = pa.Table.from_pandas(out_df, preserve_index=False)
        _atomic_write(out, table)
        print(f"  wrote {feat['id']} {asset['symbol']} {day}: "
              f"{len(out_df)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
