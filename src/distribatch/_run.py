"""Run orchestration on top of the executors: dedupe, checkpoint/resume, run summary.

`BaseJob.run` validates its arguments and calls `run_records` (local/multiprocessing)
or `run_spark`; the executors themselves stay unaware of checkpoints and dedupe.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from distribatch import _dedupe
from distribatch.checkpoint import LocalCheckpoint, Stored, validate_ids
from distribatch.stats import summarize_rows
from distribatch.types import Result

if TYPE_CHECKING:
    from distribatch.job import BaseJob

log = logging.getLogger("distribatch")

LOCAL_CHECKPOINT_EVERY = 1_000
SPARK_CHECKPOINT_EVERY = 10_000


@dataclass(frozen=True)
class RunOptions:
    output_col: str
    batch_size: int
    workers: int | None
    partitions: int | None
    dedupe: bool
    checkpoint: str | None
    id_col: str | None
    retry_failed: bool
    checkpoint_every: int | None


def run_records(
    job: BaseJob, records: Sequence[Mapping[str, Any]], backend: str, opts: RunOptions
) -> list[Result]:
    """Results for `records` in order (row_id = position)."""
    start = time.monotonic()
    ckpt: LocalCheckpoint | None = None
    ids: list[Hashable] = []
    done: dict[Hashable, Stored] = {}
    if opts.checkpoint is not None and opts.id_col is not None:
        ids = validate_ids(records, opts.id_col)
        ckpt = LocalCheckpoint(
            opts.checkpoint, job, opts.id_col, opts.checkpoint_every or LOCAL_CHECKPOINT_EVERY
        )
        done = ckpt.open()

    def is_done(i: int) -> bool:
        stored = done.get(ids[i]) if ckpt is not None else None
        return stored is not None and (stored.error is None or not opts.retry_failed)

    todo = [i for i in range(len(records)) if not is_done(i)]
    todo_records = [records[i] for i in todo]

    # Positions in `todo` served by each executed record (one each without dedupe).
    if opts.dedupe:
        uniques, mapping = _dedupe.plan(todo_records, job.required_columns())
        exec_records = [todo_records[u] for u in uniques]
        served: dict[int, list[int]] = defaultdict(list)
        for position, u in enumerate(mapping):
            served[u].append(position)
    else:
        exec_records = todo_records
        mapping = list(range(len(todo)))
        served = {i: [i] for i in range(len(todo))}

    def on_batch(batch: list[Result]) -> None:
        if ckpt is None:
            return
        for result in batch:
            for position in served[result.row_id]:
                ckpt.add(ids[todo[position]], result)

    if done:
        log.info(
            "distribatch: checkpoint has %d of %d rows; processing %d",
            len(records) - len(todo),
            len(records),
            len(todo),
        )
    try:
        executed = _execute(job, exec_records, backend, opts, on_batch)
    finally:
        if ckpt is not None:
            ckpt.flush()

    fresh = _dedupe.fan_out(executed, mapping) if opts.dedupe else executed
    results: list[Result] = []
    fresh_iter = iter(fresh)
    todo_set = set(todo)
    for i in range(len(records)):
        if i in todo_set:
            r = next(fresh_iter)
            results.append(Result(i, r.value, r.error, r.meta))
        else:
            results.append(done[ids[i]].result(i))

    stats = summarize_rows((r.error, r.meta.to_dict()) for r in results)
    log.info("distribatch: %s; %.1f s", stats, time.monotonic() - start)
    return results


def _execute(
    job: BaseJob,
    records: Sequence[Mapping[str, Any]],
    backend: str,
    opts: RunOptions,
    on_batch: Any,
) -> list[Result]:
    if backend == "multiprocessing":
        from distribatch.executors import process_pool

        return process_pool.execute(
            job, records, batch_size=opts.batch_size, workers=opts.workers, on_batch=on_batch
        )
    from distribatch.executors import local

    return local.execute(job, records, batch_size=opts.batch_size, on_batch=on_batch)


def run_spark(job: BaseJob, df: Any, opts: RunOptions) -> Any:
    from distribatch.executors import spark

    if opts.checkpoint is None:
        return spark.execute(
            job,
            df,
            output_col=opts.output_col,
            batch_size=opts.batch_size,
            partitions=opts.partitions,
            dedupe=opts.dedupe,
        )
    assert opts.id_col is not None
    return spark.execute_checkpointed(
        job,
        df,
        output_col=opts.output_col,
        batch_size=opts.batch_size,
        partitions=opts.partitions,
        dedupe=opts.dedupe,
        checkpoint=opts.checkpoint,
        id_col=opts.id_col,
        retry_failed=opts.retry_failed,
        wave_rows=opts.checkpoint_every or SPARK_CHECKPOINT_EVERY,
    )
