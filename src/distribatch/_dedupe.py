"""Send identical inputs to the model once and fan the result out.

Two rows are identical when the job's input columns hold equal values. Copies get the
same value/error and a meta with ``attempts=0`` and no tokens or latency, so run
statistics and costs count the model call once.
"""

from __future__ import annotations

import json
import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from distribatch.types import Result, ResultMeta


def input_key(record: Mapping[str, Any], columns: Sequence[str]) -> Hashable:
    return tuple(_hashable(record.get(c)) for c in columns)


def plan(
    records: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> tuple[list[int], list[int]]:
    """(indices of the first occurrence of each distinct input, position in that list for
    every record)."""
    first: dict[Hashable, int] = {}
    uniques: list[int] = []
    mapping: list[int] = []
    for i, record in enumerate(records):
        key = input_key(record, columns)
        if key not in first:
            first[key] = len(uniques)
            uniques.append(i)
        mapping.append(first[key])
    return uniques, mapping


def fan_out(unique_results: Sequence[Result], mapping: Sequence[int]) -> list[Result]:
    """One result per original record (row_id = record position)."""
    seen: set[int] = set()
    out: list[Result] = []
    for row_id, u in enumerate(mapping):
        source = unique_results[u]
        if u in seen:
            out.append(reused(source, row_id))
        else:
            seen.add(u)
            out.append(Result(row_id, source.value, source.error, source.meta))
    return out


def reused(result: Result, row_id: int) -> Result:
    meta = ResultMeta(
        attempts=0, model_id=result.meta.model_id, finish_reason=result.meta.finish_reason
    )
    return Result(row_id, result.value, result.error, meta)


def _hashable(value: Any) -> Hashable:
    if value is None or isinstance(value, str | bytes | int | bool):
        return value
    if isinstance(value, float):
        return ("nan",) if math.isnan(value) else value
    if isinstance(value, bytearray | memoryview):
        return bytes(value)
    return json.dumps(value, sort_keys=True, default=str)
