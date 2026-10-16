"""Row batch -> results. The one code path every executor runs on its workers.

Task-agnostic: the job builds requests (`build_request`) and post-processes
completions (`finish`); the engine dispatches on the job's task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from distribatch.engines.base import Engine, get_engine
from distribatch.types import (
    Completion,
    ConfigError,
    EngineError,
    MissingDependencyError,
    Request,
    Result,
    ResultMeta,
    format_error,
)

if TYPE_CHECKING:
    from distribatch.job import BaseJob


def process_records(
    job: BaseJob,
    records: Sequence[Mapping[str, Any]],
    row_ids: Sequence[int],
    batch_size: int,
) -> list[Result]:
    """Results for `records` (same order). Never raises for row-level problems."""
    engine = get_engine(job.model, job.engine, job.engine_options, job.task)
    results: list[Result] = []
    for start in range(0, len(records), batch_size):
        stop = start + batch_size
        results.extend(_process_batch(job, engine, records[start:stop], row_ids[start:stop]))
    return results


def _process_batch(
    job: BaseJob,
    engine: Engine,
    records: Sequence[Mapping[str, Any]],
    row_ids: Sequence[int],
) -> list[Result]:
    done: dict[int, Result] = {}
    requests: list[Request] = []
    for row_id, record in zip(row_ids, records, strict=True):
        try:
            requests.append(job.build_request(row_id, record, engine.capabilities))
        except Exception as exc:
            done[row_id] = Result(row_id, None, format_error(exc), ResultMeta(attempts=0))

    if requests:
        try:
            completions = {c.row_id: c for c in engine.run(job.task, requests)}
        except (ConfigError, MissingDependencyError):
            raise  # e.g. a task component failing to load: not a row problem
        except Exception as exc:
            error = format_error(EngineError(f"batch failed: {format_error(exc)}"))
            completions = {r.row_id: Completion(r.row_id, None, error) for r in requests}
        for request in requests:
            completion = completions.get(request.row_id) or Completion(
                request.row_id, None, format_error(EngineError("engine returned no completion"))
            )
            done[request.row_id] = _finish(job, completion)

    return [done[row_id] for row_id in row_ids]


def _finish(job: BaseJob, completion: Completion) -> Result:
    if completion.error is not None or completion.output is None:
        error = completion.error or format_error(EngineError("empty completion"))
        return Result(completion.row_id, None, error, completion.meta)
    try:
        value = job.finish(completion.output)
    except Exception as exc:
        return Result(completion.row_id, None, format_error(exc), completion.meta)
    return Result(completion.row_id, value, None, completion.meta)
