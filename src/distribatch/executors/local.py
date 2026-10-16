"""Sequential in-process backend: reference semantics, tests, previews."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from distribatch._pipeline import process_records
from distribatch.types import Result

if TYPE_CHECKING:
    from distribatch.job import BaseJob


def execute(
    job: BaseJob,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    on_batch: Callable[[list[Result]], None] | None = None,
) -> list[Result]:
    """Results in input order; `on_batch` sees each batch as soon as it is done."""
    results: list[Result] = []
    for start in range(0, len(records), batch_size):
        stop = min(start + batch_size, len(records))
        batch = process_records(job, records[start:stop], range(start, stop), batch_size)
        if on_batch is not None:
            on_batch(batch)
        results.extend(batch)
    return results
