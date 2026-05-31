"""Read-path: DuckDB on parquet feature store.

Usage:
    from dfi_features.store import FeatureStore
    store = FeatureStore("~/work/dfi-quant/data/features")
    df = store.load("zscore_ret_composite", asset="BTCUSDT")
    df = store.load("rv_30d", asset="BTCUSDT", d_from="2024-01-01", d_to="2024-12-31")
    df = store.query("SELECT feature_id, asset, avg(value) FROM features GROUP BY 1, 2")
"""
from __future__ import annotations
import os
import pathlib
import duckdb
import pandas as pd


class FeatureStore:
    def __init__(self, features_root: str):
        self.root = pathlib.Path(os.path.expanduser(features_root))
        self._con = duckdb.connect()

    def query(self, sql: str) -> pd.DataFrame:
        """Run arbitrary SQL. The 'features' view covers all parquet partitions."""
        glob = str(self.root / "**" / "*.parquet")
        self._con.execute(
            f"CREATE OR REPLACE VIEW features AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
        )
        return self._con.execute(sql).df()

    def load(
        self,
        feature_id: str,
        asset: str,
        version: str = "v1",
        d_from: str | None = None,
        d_to:   str | None = None,
    ) -> pd.DataFrame:
        """Load a single (feature_id, asset, version) slice, sorted by ts.

        Args:
            feature_id : e.g. "zscore_ret_composite"
            asset      : e.g. "BTCUSDT"
            version    : e.g. "v1"
            d_from     : ISO date string inclusive, e.g. "2024-01-01"
            d_to       : ISO date string inclusive, e.g. "2024-12-31"
        """
        glob = str(
            self.root / "*" / f"feature_id={feature_id}" /
            f"version={version}" / f"asset={asset}" / "**" / "*.parquet"
        )
        base = (
            f"SELECT * FROM read_parquet('{glob}') "
            f"WHERE feature_id = '{feature_id}' AND asset = '{asset}'"
        )
        if d_from:
            us = int(pd.Timestamp(d_from, tz="UTC").timestamp() * 1_000_000)
            base += f" AND ts >= {us}"
        if d_to:
            us = int(
                (pd.Timestamp(d_to, tz="UTC") + pd.Timedelta(days=1)).timestamp()
                * 1_000_000
            )
            base += f" AND ts < {us}"
        base += " ORDER BY ts"
        df = self._con.execute(base).df()
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"] * 1000, utc=True)
        return df
