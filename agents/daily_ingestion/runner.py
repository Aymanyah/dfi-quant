"""Daily OHLCV ingestion from the Binance USD-M Futures public REST API.

No API key required. One partition per (exchange, symbol, day), mirroring
the same hive layout as the rest of the raw store.

Usage:
    python -m agents.daily_ingestion.runner \\
        --from 2022-01-01 --to 2024-12-31 \\
        --symbols BTCUSDT,ETHUSDT,SOLUSDT
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time
import urllib.request
import yaml

import pyarrow as pa

from .writer import HivePartitionWriter, PartitionKey
from .schemas import SCHEMAS

ROOT = pathlib.Path(__file__).resolve().parents[2]

_FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"


def _load_settings():
    with open(ROOT / "configs" / "settings.yaml") as f:
        return yaml.safe_load(f)


def _load_universe():
    with open(ROOT / "configs" / "universe.yaml") as f:
        return yaml.safe_load(f)["assets"]


def _daterange(d0: dt.date, d1: dt.date):
    cur = d0
    while cur <= d1:
        yield cur
        cur += dt.timedelta(days=1)


def _fetch_klines_batch(symbol: str, start_ms: int, end_ms: int) -> list:
    url = (f"{_FUTURES_KLINES}?symbol={symbol}&interval=1d"
           f"&startTime={start_ms}&endTime={end_ms}&limit=1000")
    req = urllib.request.Request(url, headers={"User-Agent": "dfi-quant/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _fetch_all_klines(symbol: str, d0: dt.date, d1: dt.date) -> list:
    """Paginate through the Binance API to cover the full [d0, d1] range."""
    start_ms = int(dt.datetime(d0.year, d0.month, d0.day,
                               tzinfo=dt.timezone.utc).timestamp() * 1000)
    end_ms   = int(dt.datetime(d1.year, d1.month, d1.day, 23, 59, 59,
                               tzinfo=dt.timezone.utc).timestamp() * 1000)
    rows, cur = [], start_ms
    while cur <= end_ms:
        batch = _fetch_klines_batch(symbol, cur, end_ms)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        cur = int(batch[-1][6]) + 1  # close_time of last bar + 1 ms
        time.sleep(0.1)
    return rows


def _row_to_table(row: list, exchange: str, symbol: str) -> pa.Table:
    """Convert a single Binance kline row to a 1-row Arrow table."""
    schema = SCHEMAS["ohlcv_1d"]
    return pa.table({
        "ts_open":       pa.array([int(row[0])  * 1000], type=pa.int64()),
        "ts_close":      pa.array([int(row[6])  * 1000], type=pa.int64()),
        "open":          pa.array([float(row[1])],        type=pa.float64()),
        "high":          pa.array([float(row[2])],        type=pa.float64()),
        "low":           pa.array([float(row[3])],        type=pa.float64()),
        "close":         pa.array([float(row[4])],        type=pa.float64()),
        "volume":        pa.array([float(row[5])],        type=pa.float64()),
        "quote_vol":     pa.array([float(row[7])],        type=pa.float64()),
        "n_trades":      pa.array([int(row[8])],          type=pa.int64()),
        "taker_buy_vol": pa.array([float(row[9])],        type=pa.float64()),
        "exchange":      pa.array([exchange],             type=pa.string()),
        "symbol":        pa.array([symbol],               type=pa.string()),
    }, schema=schema)


def _row_date(row: list) -> dt.date:
    """Return the UTC date of a kline row from its open timestamp (ms)."""
    return dt.datetime.fromtimestamp(int(row[0]) / 1000,
                                     tz=dt.timezone.utc).date()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", required=True)
    ap.add_argument("--to",   dest="d_to",   required=True)
    ap.add_argument("--symbols", default=None,
                    help="comma-separated override of universe.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    settings = _load_settings()
    universe = _load_universe()
    if args.symbols:
        wanted = {s.strip() for s in args.symbols.split(",")}
        universe = [a for a in universe if a["symbol"] in wanted]

    raw_root = os.path.expanduser(settings["storage"]["local_cache"]) + "/raw"
    writer   = HivePartitionWriter(root=raw_root)

    d0 = dt.date.fromisoformat(args.d_from)
    d1 = dt.date.fromisoformat(args.d_to)

    plan = []
    for asset in universe:
        for day in _daterange(d0, d1):
            key = PartitionKey(asset["exchange"], "ohlcv_1d", asset["symbol"], day)
            if writer.is_complete(key):
                continue
            plan.append((asset, key))

    print(f"Planned {len(plan)} partitions")
    if args.dry_run:
        for asset, key in plan[:20]:
            print(" ", key)
        return 0

    # Group by asset to fetch in bulk (1 API call per asset covers the full range)
    assets_done: set = set()
    rows_by_asset: dict[str, list] = {}

    for asset, key in plan:
        aid = (asset["exchange"], asset["symbol"])
        if aid not in assets_done:
            print(f"fetching {aid[0]} {aid[1]} {d0} -> {d1} ...", flush=True)
            try:
                rows_by_asset[aid] = _fetch_all_klines(asset["symbol"], d0, d1)
            except Exception as e:
                print(f"  ERROR: {e}")
                rows_by_asset[aid] = []
            assets_done.add(aid)

        rows = rows_by_asset.get(aid, [])
        # Find the row for this specific day
        day_rows = [r for r in rows if _row_date(r) == key.day]
        if not day_rows:
            print(f"  skip {key.symbol} {key.day} : no data from API")
            continue

        table = _row_to_table(day_rows[0], key.exchange, key.symbol)
        if not table.schema.equals(SCHEMAS["ohlcv_1d"], check_metadata=False):
            raise ValueError(f"Schema mismatch for {key}")

        path = writer.write(key, table)
        print(f"  wrote {key.symbol} {key.day} -> {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
