"""Determinism smoke test for the hive partition writer.

Running the same write twice must:
- not raise
- not change the resulting parquet file (mtime may change, contents must not)
- short-circuit on the second call (idempotency)
"""
import datetime as dt
import hashlib
import pathlib

import pyarrow as pa

from agents.daily_ingestion.writer import HivePartitionWriter, PartitionKey


def _digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def test_idempotent_write(tmp_path: pathlib.Path):
    table = pa.table({
        'ts_open':       [1_709_251_200_000_000],
        'ts_close':      [1_709_337_599_000_000],
        'open':          [61_203.3],
        'high':          [62_000.0],
        'low':           [60_500.0],
        'close':         [61_800.0],
        'volume':        [12_345.6],
        'quote_vol':     [7.6e8],
        'n_trades':      [420_000],
        'taker_buy_vol': [6_200.0],
        'exchange':      ['binance-futures'],
        'symbol':        ['BTCUSDT'],
    })
    w   = HivePartitionWriter(root=str(tmp_path))
    key = PartitionKey('binance-futures', 'ohlcv_1d', 'BTCUSDT', dt.date(2024, 3, 1))

    p1 = pathlib.Path(w.write(key, table)) / 'part-00000.parquet'
    d1 = _digest(p1)

    # Second call must short-circuit — file contents unchanged.
    w.write(key, table)
    d2 = _digest(p1)

    assert d1 == d2, 'parquet contents changed on second write'
    assert (p1.parent / '_SUCCESS').exists()
