"""Prefect-style flows. Kept dependency-light here; replace with
`from prefect import flow, task` once Prefect is installed.

The orchestrator owns scheduling, retries, and SLA tracking. Agents
are invoked as subprocesses so they cannot mutate orchestrator state.

Usage:
    python orchestration/flows.py                  # daily pipeline (yesterday)
    python orchestration/flows.py --research       # QA + IC + synthesis only
    python orchestration/flows.py --from 2024-01-01 --to 2024-01-31
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import datetime as dt
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print('\n$', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def daily_pipeline(d_from: dt.date, d_to: dt.date) -> None:
    """Ingest + construct features for a date range."""
    run([sys.executable, '-m', 'agents.daily_ingestion.runner',
         '--from', d_from.isoformat(), '--to', d_to.isoformat()])
    run([sys.executable, '-m', 'agents.feature_construction.runner',
         '--from', d_from.isoformat(), '--to', d_to.isoformat()])


def research_pipeline(features: list[str] | None = None) -> None:
    """QA → IC → Synthesis — évalue toutes les features et génère le rapport."""
    qa_cmd = [sys.executable, '-m', 'agents.feature_qa.runner']
    ic_cmd = [sys.executable, '-m', 'agents.backtest_ic.runner']
    if features:
        qa_cmd += ['--features'] + features
        ic_cmd += ['--features'] + features

    run(qa_cmd)
    run(ic_cmd)
    run([sys.executable, '-m', 'agents.research_synthesis.runner'])


def full_pipeline(d_from: dt.date, d_to: dt.date) -> None:
    """Pipeline complet : ingestion → features → QA → IC → synthèse."""
    daily_pipeline(d_from, d_to)
    research_pipeline()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--from',     dest='d_from', default=None)
    ap.add_argument('--to',       dest='d_to',   default=None)
    ap.add_argument('--research', action='store_true',
                    help='Lancer uniquement QA + IC + synthèse (sans ingestion)')
    ap.add_argument('--features', nargs='*',
                    help='Limiter le research pipeline à certaines features')
    args = ap.parse_args()

    yesterday = dt.date.today() - dt.timedelta(days=1)
    d_from = dt.date.fromisoformat(args.d_from) if args.d_from else yesterday
    d_to   = dt.date.fromisoformat(args.d_to)   if args.d_to   else yesterday

    if args.research:
        print('── Research pipeline ──')
        research_pipeline(args.features)
    else:
        print(f'── Full pipeline : {d_from} → {d_to} ──')
        full_pipeline(d_from, d_to)
