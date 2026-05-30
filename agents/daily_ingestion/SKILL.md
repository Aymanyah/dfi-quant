# SKILL: Daily OHLCV ingestion agent

## Job
Fetch daily OHLCV bars from the Binance USD-M Futures public REST API for
every asset in the universe and write them to the raw store as hive-partitioned
parquet. No API key required. Idempotent.

## Inputs (read-only)
- `configs/settings.yaml`          -> storage.local_cache
- `configs/universe.yaml`          -> list of (symbol, exchange) pairs
- date range from CLI              -> `--from`, `--to` (UTC, ISO date)

## Outputs (write-only, append/overwrite by partition)
Local layout (mirrored on S3 as `s3://dfi-tardis-raw/` in prod):
```
data/raw/
  exchange=<exchange>/
    data_type=ohlcv_1d/
      symbol=<SYMBOL>/
        year=YYYY/month=MM/day=DD/
          part-00000.parquet
          _SUCCESS
```

Each parquet file contains exactly **1 row** with these columns:
- `ts_open`       (int64, microseconds UTC, bar open timestamp)
- `ts_close`      (int64, microseconds UTC, bar close timestamp)
- `open`          (float64, USDT)
- `high`          (float64, USDT)
- `low`           (float64, USDT)
- `close`         (float64, USDT)
- `volume`        (float64, base asset — e.g. BTC)
- `quote_vol`     (float64, USDT notional)
- `n_trades`      (int64)
- `taker_buy_vol` (float64, taker buy volume in base asset)
- `exchange`      (string)
- `symbol`        (string)

## Contract
1. **Idempotency**: a partition with a `_SUCCESS` marker is never re-fetched.
   Re-running the same date range is a no-op for completed days.
2. **Atomic writes**: data lands in `_tmp_<uuid>/` first; on success the
   directory is renamed and `_SUCCESS` is touched.
3. **Raw only**: no derived columns, no feature computation here.
4. **Source**: Binance USD-M Futures endpoint `fapi.binance.com/fapi/v1/klines`,
   interval `1d`. Only `binance-futures` exchange is supported.
5. **Bulk fetch**: the API is called once per asset for the full date range
   (paginated at 1000 bars), then written day by day. Reduces API calls.
6. **Rate limits**: 0.1 s sleep between paginated requests.

## Tools the agent may use
- Binance public REST API (`fapi.binance.com`) — no auth required
- `urllib.request` (stdlib), `pyarrow`, `pyyaml`
- Read access to `configs/`

## Tools the agent must NOT use
- Any external API other than `fapi.binance.com`
- Any feature store / DuckDB write APIs
- Mutations of `configs/` files

## Definition of done
- All `(symbol, day)` partitions in the requested range have a `_SUCCESS` marker.
- Each parquet file validates against `SCHEMAS["ohlcv_1d"]` in `schemas.py`.
- Row count = 1 per partition.
