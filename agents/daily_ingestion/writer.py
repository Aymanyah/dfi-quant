"""Hive-partitioned parquet writer with atomic completion markers.

Invariants:
- Writes go to `<root>/_tmp/<uuid>/...` first.
- On success, files are moved to the final partition path and a
  `_SUCCESS` marker is written.
- An existing `_SUCCESS` short-circuits the write (idempotency).
"""
from __future__ import annotations
import os, uuid, pathlib
from dataclasses import dataclass
from datetime import date
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class PartitionKey:
    exchange: str
    data_type: str
    symbol: str
    day: date

    def relpath(self) -> str:
        return (
            f"exchange={self.exchange}"
            f"/data_type={self.data_type}"
            f"/symbol={self.symbol}"
            f"/year={self.day.year:04d}"
            f"/month={self.day.month:02d}"
            f"/day={self.day.day:02d}"
        )


class HivePartitionWriter:
    def __init__(self, root: str, fs=None):
        """`root` is the storage root, e.g. s3://dfi-tardis-raw or /local/path.
        `fs` is an optional fsspec filesystem; default is local.
        """
        self.root = root.rstrip('/')
        self.fs = fs

    def is_complete(self, key: PartitionKey) -> bool:
        marker = f"{self.root}/{key.relpath()}/_SUCCESS"
        return self._exists(marker)

    def write(self, key: PartitionKey, table: pa.Table) -> str:
        if self.is_complete(key):
            return f"{self.root}/{key.relpath()}"
        tmp_dir = f"{self.root}/_tmp/{uuid.uuid4().hex}"
        self._mkdir(tmp_dir)
        tmp_file = f"{tmp_dir}/part-00000.parquet"
        pq.write_table(table, tmp_file, compression='zstd',
                       use_dictionary=True, version='2.6')
        final_dir = f"{self.root}/{key.relpath()}"
        self._mkdir(final_dir)
        self._mv(tmp_file, f"{final_dir}/part-00000.parquet")
        self._touch(f"{final_dir}/_SUCCESS")
        self._rmtree(tmp_dir)
        return final_dir

    # --- fs shims (local fallback) ---
    def _exists(self, path):
        return self.fs.exists(path) if self.fs else os.path.exists(path)
    def _mkdir(self, path):
        if self.fs:
            self.fs.makedirs(path, exist_ok=True)
        else:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    def _mv(self, src, dst):
        if self.fs: self.fs.mv(src, dst)
        else: os.replace(src, dst)
    def _touch(self, path):
        if self.fs:
            with self.fs.open(path, 'wb') as f: f.write(b'')
        else:
            pathlib.Path(path).touch()
    def _rmtree(self, path):
        if self.fs:
            try: self.fs.rm(path, recursive=True)
            except Exception: pass
        else:
            import shutil; shutil.rmtree(path, ignore_errors=True)
