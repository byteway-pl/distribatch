"""Single-node parallel backend: a `spawn` process pool, one engine per worker.

- The job travels to workers via cloudpickle (`_serialize`), so schema classes defined
  in a notebook or a driver-only module work. Records and results use plain pickle.
- Each worker pins its GPU slot (`CUDA_VISIBLE_DEVICES`) before anything imports CUDA
  libraries, then loads the engine once in its initializer.
- At most `workers` chunks are in flight, so a dead worker (OOM, segfault) can only
  have been processing those. Suspect chunks are re-run alone; a chunk that kills its
  worker alone is bisected down to the offending row, which gets a
  `WorkerCrashedError`. Pool restarts are bounded by `MAX_POOL_RESTARTS`.
- Exceptions raised in a worker (engine load failures, e.g. missing credentials) are
  configuration problems: they abort the run, as with the local backend.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue as queue_mod
import shutil
import subprocess
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from distribatch._pipeline import process_records
from distribatch._serialize import dumps_job, loads_job
from distribatch.engines.base import get_engine
from distribatch.types import ConfigError, Result, ResultMeta, WorkerCrashedError, format_error

if TYPE_CHECKING:
    from distribatch.job import BaseJob

log = logging.getLogger("distribatch")

MAX_POOL_RESTARTS = 10
PROGRESS_INTERVAL_S = 10.0
DEFAULT_CPU_WORKERS = 4

# (row_id, value, error, meta) — plain tuples cross the process boundary.
_Wire = tuple[int, Any, str | None, dict[str, Any]]


@dataclass(frozen=True)
class _Chunk:
    row_ids: tuple[int, ...]
    isolated: bool = False  # ran (or will run) alone: a crash proves it is the culprit


def execute(
    job: BaseJob,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    workers: int | None,
    on_batch: Callable[[list[Result]], None] | None = None,
) -> list[Result]:
    """Results in input order; `on_batch` sees each finished chunk (any order)."""
    if not records:
        return []

    def deliver(batch: list[Result]) -> None:
        for result in batch:
            results[result.row_id] = result
        if on_batch is not None:
            on_batch(batch)

    n_workers, slots = plan_workers(job, workers)
    pending = deque(
        _Chunk(tuple(range(start, min(start + batch_size, len(records)))))
        for start in range(0, len(records), batch_size)
    )
    n_workers = min(n_workers, len(pending))
    job_bytes = dumps_job(job)
    results: dict[int, Result] = {}
    isolate: deque[_Chunk] = deque()
    in_flight: dict[Future[list[_Wire]], _Chunk] = {}
    progress = _Progress(len(records))
    restarts = 0
    log.info("distribatch: %d rows on %d worker process(es)", len(records), n_workers)
    pool = _new_pool(job_bytes, n_workers, slots)
    try:
        while pending or isolate or in_flight:
            crashed: list[_Chunk] = []
            try:
                if isolate:  # suspects run strictly one at a time
                    if not in_flight:
                        chunk = isolate.popleft()
                        in_flight[_submit(pool, records, chunk, batch_size)] = chunk
                else:
                    while pending and len(in_flight) < n_workers:
                        chunk = pending.popleft()
                        in_flight[_submit(pool, records, chunk, batch_size)] = chunk
            except BrokenProcessPool:
                crashed.append(chunk)

            if in_flight and not crashed:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    chunk = in_flight.pop(future)
                    try:
                        wire = future.result()
                    except BrokenProcessPool:
                        crashed.append(chunk)
                        continue
                    deliver([Result(r[0], r[1], r[2], ResultMeta(**r[3])) for r in wire])
                    progress.update(wire)

            if not crashed:
                continue
            # The pool is dead: everything still in flight is a suspect too.
            suspects = crashed + list(in_flight.values())
            in_flight.clear()
            _kill(pool)
            restarts += 1
            for chunk in suspects:
                if not chunk.isolated:
                    isolate.append(_Chunk(chunk.row_ids, isolated=True))
                elif len(chunk.row_ids) > 1:
                    mid = len(chunk.row_ids) // 2
                    isolate.appendleft(_Chunk(chunk.row_ids[mid:], isolated=True))
                    isolate.appendleft(_Chunk(chunk.row_ids[:mid], isolated=True))
                else:
                    row_id = chunk.row_ids[0]
                    deliver([_crash_result(row_id, "worker process died on this row")])
                    progress.failed(1)
                    log.warning("distribatch: row %d killed its worker process", row_id)
            if restarts > MAX_POOL_RESTARTS:
                reason = f"worker pool restart limit ({MAX_POOL_RESTARTS}) reached"
                for chunk in (*isolate, *pending):
                    deliver([_crash_result(row_id, reason) for row_id in chunk.row_ids])
                    progress.failed(len(chunk.row_ids))
                log.error("distribatch: %s; remaining rows marked as failed", reason)
                isolate.clear()
                pending.clear()
                break
            log.warning(
                "distribatch: a worker process died; restarting the pool (%d/%d)",
                restarts,
                MAX_POOL_RESTARTS,
            )
            pool = _new_pool(job_bytes, n_workers, slots)
    except BaseException:
        _kill(pool)
        raise
    pool.shutdown(wait=True)
    progress.finish()
    return [results[row_id] for row_id in range(len(records))]


def plan_workers(job: BaseJob, workers: int | None) -> tuple[int, list[str]]:
    """(worker count, GPU slots). A slot is a CUDA_VISIBLE_DEVICES value; workers take
    slots round-robin. Local models get one worker per slot of `tensor_parallel_size`
    GPUs by default; API/fake models default to a few CPU workers."""
    if workers is not None and workers < 1:
        raise ConfigError("workers must be >= 1")
    if job.model.kind not in ("uc", "path"):
        return workers or min(DEFAULT_CPU_WORKERS, os.cpu_count() or 1), []
    gpus = visible_gpus()
    if not gpus:
        return workers or 1, []
    tp = int(job.engine_options.get("tensor_parallel_size", 1))
    if tp < 1 or tp > len(gpus):
        raise ConfigError(f"tensor_parallel_size={tp} does not fit {len(gpus)} visible GPU(s)")
    slots = [",".join(gpus[i : i + tp]) for i in range(0, len(gpus) - tp + 1, tp)]
    return workers or len(slots), slots


def visible_gpus() -> list[str]:
    """GPU ids visible to this process, without importing torch."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g.strip() for g in env.split(",") if g.strip() and g.strip() != "-1"]
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return []
    try:
        out = subprocess.run([nvidia_smi, "-L"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    count = sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
    return [str(i) for i in range(count)]


# --- driver helpers -------------------------------------------------------------------


def _new_pool(job_bytes: bytes, n_workers: int, slots: list[str]) -> ProcessPoolExecutor:
    ctx = multiprocessing.get_context("spawn")
    slot_queue: Any = ctx.Queue()
    for i in range(n_workers):
        slot_queue.put(slots[i % len(slots)] if slots else None)
    return ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(job_bytes, slot_queue),
    )


def _submit(
    pool: ProcessPoolExecutor,
    records: Sequence[Mapping[str, Any]],
    chunk: _Chunk,
    batch_size: int,
) -> Future[list[_Wire]]:
    rows = [records[i] for i in chunk.row_ids]
    return pool.submit(_run_chunk, rows, list(chunk.row_ids), batch_size)


def _kill(pool: ProcessPoolExecutor) -> None:
    """Shut the pool down without waiting for running chunks (crash or interrupt)."""
    # `_processes` is private but stable since 3.x; without it workers would keep
    # running their current chunk (minutes for big GPU batches) after an interrupt.
    processes = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)


def _crash_result(row_id: int, message: str) -> Result:
    error = format_error(WorkerCrashedError(message))
    return Result(row_id, None, error, ResultMeta(attempts=0))


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.errors = 0
        self.start = self.last = time.monotonic()

    def update(self, wire: list[_Wire]) -> None:
        self.done += len(wire)
        self.errors += sum(1 for row in wire if row[2] is not None)
        now = time.monotonic()
        if now - self.last >= PROGRESS_INTERVAL_S:
            self.last = now
            self._log(now)

    def failed(self, n: int) -> None:
        self.done += n
        self.errors += n

    def finish(self) -> None:
        self._log(time.monotonic())

    def _log(self, now: float) -> None:
        elapsed = max(now - self.start, 1e-9)
        log.info(
            "distribatch: %d/%d rows (%d errors), %.1f rows/s",
            self.done,
            self.total,
            self.errors,
            self.done / elapsed,
        )


# --- worker side ----------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(job_bytes: bytes, slot_queue: Any) -> None:
    try:
        slot = slot_queue.get_nowait()
    except queue_mod.Empty:
        slot = None
    if slot is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = slot  # before any CUDA library is imported
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        job = loads_job(job_bytes)
        _WORKER["job"] = job
        get_engine(job.model, job.engine, job.engine_options, job.task)  # load once, now
    except Exception as exc:
        # Raised from the first task instead: an initializer exception would only
        # surface as an anonymous BrokenProcessPool.
        _WORKER["error"] = exc


def _run_chunk(
    records: list[Mapping[str, Any]], row_ids: list[int], batch_size: int
) -> list[_Wire]:
    if "error" in _WORKER:
        raise _WORKER["error"]
    results = process_records(_WORKER["job"], records, row_ids, batch_size)
    return [(r.row_id, r.value, r.error, r.meta.to_dict()) for r in results]
