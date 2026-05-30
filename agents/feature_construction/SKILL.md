# SKILL: Feature construction agent

## Job
Compute deterministic, point-in-time-correct features from raw market data.
For each feature declared in `configs/features.yaml`, read the required raw
parquet partitions, run the corresponding `compute()` function from the
`dfi_features` package, and write the resulting Series to the feature store
as hive-partitioned parquet, idempotently.

## Inputs (read-only)
- `configs/features.yaml`           -> registry of features (formula, lookback, deps, version)
- `configs/universe.yaml`           -> per-feature applicable assets
- `data/raw/exchange=.../...`       -> raw partitions written by tardis_ingestion
- date range from CLI                -> `--from`, `--to` (UTC, ISO date)

## Outputs (write-only)
```
data/features/
  feature_family=<family>/
    feature_id=<id>/
      version=<vN>/
        asset=<SYMBOL>/
          year=YYYY/month=MM/day=DD/
            part-00000.parquet
            _SUCCESS
```
Each parquet MUST contain:
- `ts`        (int64, microseconds UTC, end-of-bar timestamp)
- `feature_id` (string, redundant)
- `version`    (string)
- `asset`      (string)
- `value`      (float64; NaN allowed at warmup)

## Contract
1. Determinism: `(feature_id, version, asset, ts)` is the primary key.
   Re-running the same range produces byte-identical parquet.
2. **No lookahead.** At time t, the feature value MUST depend only on raw
   data with `ts <= t - 1 unit_of_bar`. Verified by `scripts/check_lookahead.py`.
3. Versions are immutable. Code change to `compute()` => bump version in
   `features.yaml`. Old versions stay readable forever.
4. NaN policy: leave NaNs during warmup; never forward-fill across the
   first valid value. No imputation here.
5. One feature, one module. `compute(df, params) -> pd.Series` only.
6. Atomic writes: tmp prefix + rename + `_SUCCESS` marker.

## API contract for each feature module
```python
# dfi_features/<feature_id>.py
def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    """One-line description.

    Formula:
        $$ value_t = ... $$

    Inputs:
        df: tidy DataFrame indexed by ts (datetime64[ns, UTC]) with
            the columns listed under `requires` in features.yaml.
        params: dict of hyperparameters (lookback, etc.).

    Returns:
        pd.Series indexed by ts, name = feature_id.
    """
```
- Type hints required.
- Docstring with LaTeX formula required.
- Unit test required at `tests/test_<feature_id>.py` using synthetic data.

## Tools the agent may use
- Filesystem under the repo (read raw, write features, read configs)
- Sample raw data already on disk (1 month of trades)
- `pyarrow`, `pandas`, `numpy`, `pytest`, `ruff`

## Tools the agent must NOT use
- Production / live data feeds
- S3 write access
- Any credentials, including `TARDIS_API_KEY`
- Mutations of `configs/features.yaml` itself (only humans bump versions)

## Tests it MUST run before declaring done
```
pytest tests/test_<feature>.py
python scripts/check_lookahead.py --feature <feature_id>
```

## Definition of done
- All requested `(feature_id, version, asset, day)` partitions have `_SUCCESS`.
- pytest passes for every feature touched.
- `check_lookahead.py` exits 0 for every feature touched.
- `ruff check dfi_features/` returns 0.
