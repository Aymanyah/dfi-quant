# dfi-quant — Claude Code Agent Guide

This file is the canonical reference for Claude Code operating as a feature
construction agent in this repository. Read it before touching any code.

---

## Repository layout

```
dfi-quant/
├── agents/
│   ├── daily_ingestion/        # Fetches daily OHLCV from Binance → data/raw/
│   │   ├── SKILL.md            # Agent contract
│   │   ├── runner.py
│   │   ├── writer.py           # HivePartitionWriter (shared)
│   │   └── schemas.py          # Arrow schemas
│   └── feature_construction/   # raw → features
│       └── runner.py
├── configs/
│   ├── settings.yaml           # storage.local_cache path
│   ├── universe.yaml           # list of {symbol, exchange} pairs
│   └── features.yaml           # feature registry (source of truth)
├── dfi_features/               # One .py per feature; each exports compute()
├── tests/                      # pytest; one file per feature
├── scripts/
│   └── check_lookahead.py      # Causal-correctness harness
├── orchestration/
│   └── flows.py                # End-to-end pipeline runner
└── notebooks/                  # Exploration only — not part of the pipeline
```

---

## Data layout (hive-partitioned parquet)

```
data/raw/
  exchange=<exchange>/
    data_type=<type>/
      symbol=<SYMBOL>/
        year=YYYY/month=MM/day=DD/
          part-00000.parquet
          _SUCCESS

data/features/
  feature_family=<family>/
    feature_id=<id>/
      version=<version>/
        asset=<SYMBOL>/
          year=YYYY/month=MM/day=DD/
            part-00000.parquet
            _SUCCESS
```

A `_SUCCESS` marker means the partition is complete. The runner skips
partitions that already have `_SUCCESS` — this makes every run idempotent.

---

## Available raw sources

| data_type | Key columns | Status |
|---|---|---|
| `ohlcv_1d` | `ts_open`, `ts_close`, `open`, `high`, `low`, `close`, `volume`, `quote_vol`, `n_trades`, `taker_buy_vol`, `exchange`, `symbol` | **Populated** — Binance USD-M Futures daily bars |
| `derivative_ticker` | `funding_rate`, `open_interest` | Not yet populated |
| `book_snapshot_25` | `bid_price_0`…`bid_price_24`, `ask_price_0`…`ask_price_24`, `bid_size_0`…`bid_size_24`, `ask_size_0`…`ask_size_24` | Not yet populated |

---

## Feature contract

Every feature lives in `dfi_features/<feature_id>.py` and must export:

```python
def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    ...
```

### Rules

1. **Input** — `df` is indexed by a `DatetimeIndex` (UTC). Columns come from
   the raw source declared in `features.yaml`.

2. **Output** — a `pd.Series` with the same `DatetimeIndex` and float dtype.
   Return `pd.Series(dtype=float)` if required columns are absent.

3. **Point-in-time safety** — the value at index `t` must depend only on data
   at index `≤ t`. Use `rolling(window, min_periods=window)` — this
   automatically produces NaN during warmup without peeking forward.

4. **NaN for warmup** — the first `window - 1` values are NaN by convention.
   The runner drops NaN before writing to the feature store.

5. **No side effects** — do not read files, call APIs, or mutate `df`.

6. **Graceful degradation** — if a required column is missing, return
   `pd.Series(dtype=float)` rather than raising an exception. This allows
   features to be registered for future data sources without breaking the
   pipeline today.

7. **Default params** — `params` may be `{}` (e.g. in the lookahead harness).
   Always use `params.get("key", default)`.

---

## Feature registry (`configs/features.yaml`)

```yaml
- id: <feature_id>        # must match filename in dfi_features/
  version: v1             # bump only on breaking schema change
  family: <family>        # volatility | momentum | microstructure | derivatives
  source: <data_type>     # matches data_type in the hive path
  requires: [col1, col2]  # columns that must be present
  params:
    window_d: 30          # passed to compute() verbatim
  description: |
    What it measures, range, sign convention.
```

The runner uses `window_d` and `lookback_d` from `params` to determine how
many extra days of history to load before the target day (warmup lookback).

---

## Existing features

| id | source | what it measures |
|---|---|---|
| `rv_30d` | `ohlcv_1d` | 30d rolling annualized realized volatility |
| `cvd_20d` | `ohlcv_1d` | 20d rolling cumulative volume delta, normalized |
| `funding_zscore_14d` | `derivative_ticker` | 14d z-score of 8h funding rate; empty for ohlcv_1d |
| `obi_0_5pct` | `book_snapshot_25` | Order book imbalance within 0.5% of mid; empty without L2 |

---

## How to add a new feature

1. Create `dfi_features/<feature_id>.py` implementing `compute(df, params)`.
2. Add an entry to `configs/features.yaml`.
3. Add `tests/test_<feature_id>.py` with at minimum:
   - warmup NaN test
   - sign / magnitude sanity test
   - missing-column returns-empty test
4. Verify no lookahead: `python scripts/check_lookahead.py --feature <feature_id>`
5. Run tests: `pytest tests/test_<feature_id>.py -v`

Both commands must succeed before the feature is considered done.

---

## Running the pipeline

```bash
# Ingest one day of OHLCV
python -m agents.daily_ingestion.runner --from 2024-01-01 --to 2024-01-01

# Compute all features for one day
python -m agents.feature_construction.runner --from 2024-01-01 --to 2024-01-01

# Compute a specific feature for a specific symbol
python -m agents.feature_construction.runner \
    --from 2024-01-01 --to 2024-01-31 \
    --features rv_30d --symbols BTCUSDT

# Full pipeline (yesterday)
python orchestration/flows.py

# Lookahead checks
python scripts/check_lookahead.py --feature rv_30d
python scripts/check_lookahead.py --feature cvd_20d
python scripts/check_lookahead.py --feature funding_zscore_14d

# Full test suite
pytest tests/ -v
```

---

## Agent constraints

- **Do not** modify `configs/settings.yaml` or `configs/universe.yaml`.
- **Do not** write to `data/raw/` — that belongs to `agents/daily_ingestion/`.
- **Do not** call external APIs from inside `dfi_features/` modules.
- **Do not** import from `agents/` inside `dfi_features/`.
- **Do not** bump a feature version without a schema change — version bumps
  invalidate all previously written partitions.
- When a required data source is unavailable, return empty — never raise.
