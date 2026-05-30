# Claude Code — Feature QA Agent

You are working in `agents/feature_qa/`. Your job is to audit features
statistically — never to modify them.

## What this agent does

Runs 4 checks on every ohlcv_1d feature and writes a Markdown report to `reports/qa/`:

| Check | Fail condition |
|---|---|
| Distribution | ADF p > 0.05 (non-stationnaire) ou outlier rate > 1% |
| Coverage | NaN rate > 20% après warmup sur n'importe quel asset |
| Lookahead | Valeur à t change si on masque les données après t |
| Explosion | std finale > 5× std initiale |

## How to run

```bash
python -m agents.feature_qa.runner                      # toutes les features
python -m agents.feature_qa.runner --features mom_20d   # une seule
```

## When modifying checks.py

- Each check returns `{"passed": bool, "message": str, "detail": dict}`
- Never raise exceptions — catch and return `passed=False` with a message
- Keep checks deterministic: lookahead uses `seed=42`, ADF uses `maxlag=5`
- Do not modify anything in `data/` or `configs/`

## When a check fails

A FAIL writes to the report but does not crash the pipeline (exit 0).
If you need to investigate why a feature fails, read its report at
`reports/qa/<feature_id>.md` and trace back to `dfi_features/<feature_id>.py`.
Do not patch the check to make it pass — fix the feature.
