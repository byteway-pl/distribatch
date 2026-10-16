"""Input/output adapters for record-based backends (list of dicts, pandas).

Detection never imports pandas/pyspark: an object can only be a pandas/Spark
DataFrame if the user already imported that library.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from distribatch.types import ConfigError, Result


@dataclass(frozen=True)
class TabularInput:
    records: list[Mapping[str, Any]]
    columns: list[str]
    kind: Literal["records", "pandas"]
    original: Any


def output_columns(output_col: str) -> tuple[str, str, str]:
    """(value, error, meta) column names."""
    return output_col, f"{output_col}_error", f"{output_col}_meta"


def is_pandas_dataframe(data: Any) -> bool:
    pd = sys.modules.get("pandas")
    return pd is not None and isinstance(data, pd.DataFrame)


def is_spark_dataframe(data: Any) -> bool:
    # Covers classic (pyspark.sql.dataframe) and Spark Connect (pyspark.sql.connect...).
    return any(
        cls.__name__ == "DataFrame" and cls.__module__.startswith("pyspark.sql")
        for cls in type(data).__mro__
    )


def read_input(data: Any) -> TabularInput:
    if is_pandas_dataframe(data):
        return TabularInput(
            records=data.to_dict(orient="records"),
            columns=[str(c) for c in data.columns],
            kind="pandas",
            original=data,
        )
    if isinstance(data, Sequence) and not isinstance(data, str | bytes):
        if not all(isinstance(row, Mapping) for row in data):
            raise ConfigError("list input must contain only mappings (dicts), one per row")
        columns: dict[str, None] = {}
        for row in data:
            columns.update(dict.fromkeys(row))
        return TabularInput(
            records=list(data), columns=list(columns), kind="records", original=data
        )
    raise ConfigError(
        f"unsupported input type {type(data).__name__}; "
        "pass a pandas DataFrame, a list of dicts, or a Spark DataFrame with backend='spark'"
    )


def head(data: Any, n: int) -> Any:
    """First `n` rows, keeping the input type (Spark -> pandas, collected to the driver)."""
    if is_spark_dataframe(data):
        return data.limit(n).toPandas()
    if is_pandas_dataframe(data):
        return data.head(n)
    if isinstance(data, Sequence) and not isinstance(data, str | bytes):
        return list(data[:n])
    return data


def write_output(inp: TabularInput, results: Sequence[Result], output_col: str) -> Any:
    if len(results) != len(inp.records):
        raise AssertionError(f"{len(results)} results for {len(inp.records)} rows")
    value_col, error_col, meta_col = output_columns(output_col)
    values = [r.value for r in results]
    errors = [r.error for r in results]
    metas = [r.meta.to_dict() for r in results]
    if inp.kind == "pandas":
        out = inp.original.copy()
        out[value_col] = values
        out[error_col] = errors
        out[meta_col] = metas
        return out
    return [
        {**record, value_col: value, error_col: error, meta_col: meta}
        for record, value, error, meta in zip(inp.records, values, errors, metas, strict=True)
    ]
