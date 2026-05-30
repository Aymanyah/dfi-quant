# Claude Code — Feature Construction

You are working in `dfi_features/`. Your job is to implement new features.

## Before writing any code

1. Check `configs/features.yaml` — the feature must be registered there first.
2. Check `configs/universe.yaml` — know which assets will use this feature.
3. Read an existing feature (e.g. `cvd_20d.py`) to match the style.

## Rules for every feature

- One file per feature: `dfi_features/<feature_id>.py`
- One public function: `def compute(df: pd.DataFrame, params: dict) -> pd.Series`
- Return `pd.Series(dtype=float)` if a required column is missing — never raise
- Use `params.get("key", default)` — never assume params is populated
- Use `rolling(window, min_periods=window)` — never `min_periods=1`
- Never read files, call APIs, or mutate `df`

## After writing the feature

Run in this exact order — do not skip steps:

```bash
pytest tests/test_<feature_id>.py -v
python scripts/check_lookahead.py --feature <feature_id>
```

Both must exit 0 before the feature is considered done.

## Test file requirements

Every feature needs `tests/test_<feature_id>.py` with at minimum:
- `test_warmup_is_nan` — first `window-1` values must be NaN
- `test_sign_correct` — rising prices → positive signal (or correct sign)
- `test_missing_column_returns_empty` — no crash on missing columns
- `test_value_matches_formula` — numerical correctness against manual calc

## Common mistakes to avoid

- Using `min_periods=1` → produces values during warmup → lookahead test fails
- Forgetting to handle missing columns → pipeline crashes on derivative_ticker source
- Using `df.index[-1]` inside compute → implicit lookahead
- Mutating `df` with `df['new_col'] = ...` → side effects across calls
