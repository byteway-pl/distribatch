"""Checkpoints for the local and multiprocessing backends.

Layout (shared with the Spark backend, which stores results as parquet instead):

    <path>/manifest.json         job fingerprint, task, id column
    <path>/results/*.jsonl       one object per processed row: id, value, error, meta

Results are buffered and flushed every `flush_every` rows (and when the run ends or is
interrupted), each flush an atomically renamed new file. On load, a successful result
wins over a failed one for the same id; otherwise the newest wins.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from distribatch.models.resolve import fuse_path
from distribatch.types import ConfigError, Result, ResultMeta

if TYPE_CHECKING:
    from distribatch.job import BaseJob

MANIFEST = "manifest.json"
RESULTS = "results"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class Stored:
    value: Any
    error: str | None
    meta: dict[str, Any]

    def result(self, row_id: int) -> Result:
        return Result(row_id, self.value, self.error, ResultMeta(**self.meta))


def manifest_for(job: BaseJob, id_col: str) -> dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "fingerprint": job.fingerprint(),
        "task": job.task,
        "id_col": id_col,
    }


def check_manifest(found: Mapping[str, Any], expected: Mapping[str, Any], path: str) -> None:
    for key in ("fingerprint", "id_col"):
        if found.get(key) != expected[key]:
            raise ConfigError(
                f"checkpoint {path} belongs to a different job ({key} differs: "
                f"{found.get(key)!r} != {expected[key]!r}); use another path or delete it"
            )


def validate_ids(records: Sequence[Mapping[str, Any]], id_col: str) -> list[Hashable]:
    ids: list[Hashable] = []
    seen: set[Hashable] = set()
    for i, record in enumerate(records):
        if id_col not in record:
            raise ConfigError(f"row {i} has no id column {id_col!r}")
        value = record[id_col]
        if isinstance(value, bool) or not isinstance(value, str | int):
            raise ConfigError(
                f"id column {id_col!r} must hold strings or integers, got "
                f"{type(value).__name__} in row {i}"
            )
        if value in seen:
            raise ConfigError(f"id column {id_col!r} has duplicate value {value!r}")
        seen.add(value)
        ids.append(value)
    return ids


class LocalCheckpoint:
    def __init__(self, path: str, job: BaseJob, id_col: str, flush_every: int) -> None:
        self.path = path
        self.root = Path(fuse_path(path))
        self.manifest = manifest_for(job, id_col)
        self.flush_every = flush_every
        self._buffer: list[dict[str, Any]] = []

    def open(self) -> dict[Hashable, Stored]:
        """Create or validate the checkpoint; return the stored results by id."""
        (self.root / RESULTS).mkdir(parents=True, exist_ok=True)
        manifest_file = self.root / MANIFEST
        if manifest_file.exists():
            check_manifest(json.loads(manifest_file.read_text()), self.manifest, self.path)
        else:
            _atomic_write(manifest_file, json.dumps(self.manifest, indent=2))
        done: dict[Hashable, Stored] = {}
        for file in sorted((self.root / RESULTS).glob("*.jsonl")):
            for line in file.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                stored = Stored(row["value"], row["error"], row["meta"])
                previous = done.get(row["id"])
                if previous is None or stored.error is None or previous.error is not None:
                    done[row["id"]] = stored
        return done

    def add(self, row_id: Hashable, result: Result) -> None:
        self._buffer.append(
            {
                "id": row_id,
                "value": result.value,
                "error": result.error,
                "meta": result.meta.to_dict(),
            }
        )
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        lines = "".join(json.dumps(row, default=str) + "\n" for row in self._buffer)
        name = f"part-{time.time_ns():020d}-{os.getpid()}.jsonl"
        _atomic_write(self.root / RESULTS / name, lines)
        self._buffer.clear()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
