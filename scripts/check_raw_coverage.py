"""Check that every (exchange, symbol, data_type, day) partition expected
for a date range has a `_SUCCESS` marker.

Exits 0 iff fully covered. Used as a gate before the feature
construction agent runs.
"""
from __future__ import annotations
import argparse, os, sys, datetime as dt, pathlib, yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--from', dest='d_from', required=True)
    p.add_argument('--to',   dest='d_to',   required=True)
    args = p.parse_args()

    settings = yaml.safe_load(open(ROOT / 'configs' / 'settings.yaml'))
    universe = yaml.safe_load(open(ROOT / 'configs' / 'universe.yaml'))['assets']
    data_types = settings['tardis']['data_types']
    raw_root = pathlib.Path(os.path.expanduser(settings['storage']['local_cache'])) / 'raw'

    d0 = dt.date.fromisoformat(args.d_from)
    d1 = dt.date.fromisoformat(args.d_to)

    missing = []
    cur = d0
    while cur <= d1:
        for asset in universe:
            for dtp in data_types:
                rel = (
                    f"exchange={asset['exchange']}/data_type={dtp}/"
                    f"symbol={asset['symbol']}/year={cur.year:04d}/"
                    f"month={cur.month:02d}/day={cur.day:02d}"
                )
                marker = raw_root / rel / '_SUCCESS'
                if not marker.exists():
                    missing.append(str(marker))
        cur += dt.timedelta(days=1)

    if missing:
        print(f'MISSING {len(missing)} partitions')
        for m in missing[:20]:
            print(' ', m)
        sys.exit(1)
    print('OK: all partitions covered')
    sys.exit(0)


if __name__ == '__main__':
    main()
