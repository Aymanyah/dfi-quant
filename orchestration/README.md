# orchestration

The orchestrator (Prefect / Airflow / cron) is the only thing allowed to
schedule agent runs. Agents never trigger each other directly. A run
looks like:

```
day = T-1 UTC
  tardis_ingestion --from day --to day
  check_raw_coverage --from day --to day      # gate
  feature_construction --as-of day
  feature_qa --as-of day                       # gate
  backtest_ic --as-of day --window 90d
  research_synthesis --as-of day
```

Gates exit non-zero on failure; the orchestrator decides whether to
retry or page a human. Agents are invoked as subprocesses (no shared
Python interpreter) so they cannot mutate orchestrator state.
