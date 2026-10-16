"""Spark backend: `DataFrame.mapInArrow`, Spark Connect compatible.

The output schema is computed on the driver from the job's `output_type()` before any
work runs. On the executors each Arrow batch is processed by the shared pipeline:
only the columns the job reads are converted to Python, every other column passes
through as the original Arrow arrays, and the three output columns are appended.
Engines come from the process-level cache, so Python worker reuse (Spark's default)
loads weights once per worker, not per task.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from distribatch import _dedupe, outputs
from distribatch._pipeline import process_records
from distribatch._serialize import dumps_job, loads_job
from distribatch.checkpoint import RESULTS, check_manifest, manifest_for
from distribatch.io import output_columns
from distribatch.models.resolve import stage_uc_model
from distribatch.models.specs import UCModel
from distribatch.outputs import OutputTypeError, OutType
from distribatch.types import ConfigError, format_error

if TYPE_CHECKING:
    from distribatch.job import BaseJob

log = logging.getLogger("distribatch")


def execute(
    job: BaseJob,
    df: Any,
    *,
    output_col: str,
    batch_size: int,
    partitions: int | None,
    dedupe: bool = False,
) -> Any:
    from distribatch._deps import require

    T = require("pyspark.sql.types", "spark")
    if isinstance(job.model, UCModel) and job.model.stage_dir is not None:
        # Download once on the driver (which has registry credentials in notebooks);
        # executors copy from the shared stage directory to their local disk.
        job = job.model_copy(update={"model": stage_uc_model(job.model)})
    value_type = job.output_type()
    names = output_columns(output_col)
    types = (value_type, outputs.STRING, outputs.META)
    schema = T.StructType(
        [
            *df.schema.fields,
            *(
                T.StructField(n, outputs.to_spark(t), True)
                for n, t in zip(names, types, strict=True)
            ),
        ]
    )
    fn = _PartitionFn(
        job_bytes=dumps_job(job),
        columns=tuple(job.required_columns()),
        batch_size=batch_size,
        value_type=value_type,
        names=names,
        types=types,
        dedupe=dedupe,
    )
    if partitions is not None:
        df = df.repartition(partitions)
    return df.mapInArrow(fn, schema)


@dataclass(frozen=True)
class _PartitionFn:
    """Runs on the executors (distribatch must be installed on the cluster). The job
    travels pre-serialized, see `_serialize`."""

    job_bytes: bytes
    columns: tuple[str, ...]
    batch_size: int
    value_type: OutType
    names: tuple[str, str, str]
    types: tuple[OutType, OutType, OutType]
    dedupe: bool = False

    def __call__(self, batches: Iterator[Any]) -> Iterator[Any]:
        import pyarrow as pa

        pin_task_gpus()
        job = loads_job(self.job_bytes)
        arrow_types = [outputs.to_arrow(t) for t in self.types]
        for batch in batches:
            n = batch.num_rows
            if n == 0:
                continue
            records = _records(batch, self.columns)
            if self.dedupe:
                uniques, mapping = _dedupe.plan(records, self.columns)
                unique_records = [records[u] for u in uniques]
                done = process_records(job, unique_records, range(len(uniques)), self.batch_size)
                results = _dedupe.fan_out(done, mapping)
            else:
                results = process_records(job, records, range(n), self.batch_size)
            values: list[Any] = []
            errors: list[str | None] = []
            metas: list[Any] = []
            for result in results:
                value, error = result.value, result.error
                if error is None:
                    try:
                        value = outputs.coerce(self.value_type, value)
                    except OutputTypeError as exc:
                        value, error = None, format_error(exc)
                values.append(value)
                errors.append(error)
                metas.append(outputs.coerce(outputs.META, result.meta.to_dict()))
            new = [
                pa.array(column, type=arrow_type)
                for column, arrow_type in zip((values, errors, metas), arrow_types, strict=True)
            ]
            yield pa.RecordBatch.from_arrays(
                [*batch.columns, *new], names=[*batch.schema.names, *self.names]
            )


_PINNED_ENV = "DISTRIBATCH_GPU_PINNED"


def pin_task_gpus() -> None:
    """Restrict this Python worker to the GPUs Spark assigned to its task.

    Spark hands GPU addresses to tasks (``spark.task.resource.gpu.amount``) but does
    not set ``CUDA_VISIBLE_DEVICES``, so every model on a multi-GPU node would land on
    GPU 0. Done once per worker process, before the engine (and CUDA) loads; the
    cached engine then stays on those GPUs.
    """
    if os.environ.get(_PINNED_ENV):
        return
    os.environ[_PINNED_ENV] = "1"
    try:
        from pyspark import TaskContext

        context = TaskContext.get()
        gpu = context.resources().get("gpu") if context is not None else None
    except Exception:  # no Spark task context (e.g. tests calling the function directly)
        return
    if gpu is not None and gpu.addresses:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu.addresses)
        log.info("distribatch: Python worker pinned to GPU(s) %s", ",".join(gpu.addresses))


def _records(batch: Any, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    """Only the columns the job reads, as Python rows (binary stays bytes)."""
    data = {c: batch.column(batch.schema.get_field_index(c)).to_pylist() for c in columns}
    return [{c: data[c][i] for c in columns} for i in range(batch.num_rows)]


def execute_checkpointed(
    job: BaseJob,
    df: Any,
    *,
    output_col: str,
    batch_size: int,
    partitions: int | None,
    dedupe: bool,
    checkpoint: str,
    id_col: str,
    retry_failed: bool,
    wave_rows: int,
) -> Any:
    """Eager run in waves, each appended to ``<checkpoint>/results`` (parquet); returns
    the input joined with the stored results (re-reading it costs no inference)."""
    from distribatch._deps import require

    F = require("pyspark.sql.functions", "spark")
    spark = df.sparkSession
    root = checkpoint.rstrip("/")
    results_path = f"{root}/{RESULTS}"
    _ensure_manifest(spark, f"{root}/manifest", manifest_for(job, id_col), checkpoint)

    dupes = df.groupBy(F.col(f"`{id_col}`")).count().where(F.col("count") > 1).limit(1)
    if dupes.count():
        raise ConfigError(f"id column {id_col!r} has duplicate values")

    names = output_columns(output_col)
    value_col, error_col, _ = names
    stored = _latest(spark, results_path, id_col, error_col)
    todo = df
    if stored is not None:
        finished = stored.where(F.col(f"`{error_col}`").isNull()) if retry_failed else stored
        todo = df.join(finished.select(f"`{id_col}`"), on=id_col, how="left_anti")
    columns = list(dict.fromkeys([id_col, *job.required_columns()]))
    todo = todo.select(*[F.col(f"`{c}`") for c in columns])
    remaining = todo.count()
    waves = max(1, -(-remaining // wave_rows)) if remaining else 0
    log.info("distribatch: %d rows to process in %d wave(s)", remaining, waves)
    for k in range(waves):
        wave = todo
        if waves > 1:
            wave = todo.where(F.pmod(F.hash(F.col(f"`{id_col}`")), F.lit(waves)) == k)
        out = execute(
            job,
            wave,
            output_col=output_col,
            batch_size=batch_size,
            partitions=partitions,
            dedupe=dedupe,
        )
        (
            out.select(*[F.col(f"`{c}`") for c in (id_col, *names)])
            .withColumn("_written_at", F.current_timestamp())
            .write.mode("append")
            .parquet(results_path)
        )
        log.info("distribatch: wave %d/%d written to %s", k + 1, waves, results_path)

    stored = _latest(spark, results_path, id_col, error_col)
    if stored is None:  # empty input, nothing ever written
        return execute(job, df, output_col=output_col, batch_size=batch_size, partitions=None)
    joined = df.join(stored, on=id_col, how="left")
    return joined.select(*[F.col(f"`{c}`") for c in df.columns], *[F.col(f"`{c}`") for c in names])


def _ensure_manifest(spark: Any, path: str, expected: dict[str, Any], checkpoint: str) -> None:
    try:
        found = spark.read.json(path).first()
    except Exception:  # path does not exist yet
        found = None
    if found is not None:
        check_manifest(found.asDict(), expected, checkpoint)
        return
    spark.createDataFrame([expected]).coalesce(1).write.mode("overwrite").json(path)


def _latest(spark: Any, results_path: str, id_col: str, error_col: str) -> Any:
    """One stored row per id: a success over a failure, then the newest."""
    from distribatch._deps import require

    F = require("pyspark.sql.functions", "spark")
    W = require("pyspark.sql.window", "spark").Window
    try:
        stored = spark.read.parquet(results_path)
        stored.schema  # noqa: B018 — forces the path check
    except Exception:
        return None
    order = W.partitionBy(F.col(f"`{id_col}`")).orderBy(
        F.col(f"`{error_col}`").isNull().desc(), F.col("_written_at").desc()
    )
    return (
        stored.withColumn("_rank", F.row_number().over(order))
        .where(F.col("_rank") == 1)
        .drop("_rank", "_written_at")
    )
