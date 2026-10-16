"""Run statistics computed from a job's output (any backend).

`summarize(output)` reads the ``<out>_error`` and ``<out>_meta`` columns: on Spark it
aggregates in the cluster (no rows are collected), for pandas/list it iterates. Costs
use prices you pass in; no price tables are shipped (they go stale).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from distribatch import io
from distribatch._deps import require
from distribatch.types import ConfigError

if TYPE_CHECKING:
    from distribatch.job import BaseJob

_ERROR_CLASS = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")


@dataclass(frozen=True)
class RunStats:
    rows: int
    succeeded: int
    failed: int
    reused: int  # served from an identical input (dedupe) or a checkpoint copy
    retries: int  # extra API attempts beyond the first
    prompt_tokens: int
    completion_tokens: int
    audio_seconds: float
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    errors: dict[str, int] = field(default_factory=dict)  # by error class

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.rows if self.rows else 1.0

    def cost(
        self,
        *,
        input_per_million: float = 0.0,
        output_per_million: float = 0.0,
        per_audio_minute: float = 0.0,
    ) -> float:
        """Estimated cost with your prices (e.g. USD per 1M tokens / per audio minute)."""
        return (
            self.prompt_tokens * input_per_million / 1e6
            + self.completion_tokens * output_per_million / 1e6
            + self.audio_seconds / 60 * per_audio_minute
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "success_rate": self.success_rate}

    def __str__(self) -> str:
        parts = [
            f"{self.rows} rows: {self.succeeded} ok, {self.failed} failed "
            f"({self.success_rate:.1%} success)",
        ]
        if self.reused:
            parts.append(f"{self.reused} reused")
        if self.prompt_tokens or self.completion_tokens:
            parts.append(f"tokens {self.prompt_tokens} in / {self.completion_tokens} out")
        if self.audio_seconds:
            parts.append(f"audio {self.audio_seconds / 3600:.2f} h")
        if self.retries:
            parts.append(f"{self.retries} retries")
        if self.latency_ms_p50 is not None:
            parts.append(
                f"latency p50 {self.latency_ms_p50:.0f} ms, p95 {self.latency_ms_p95:.0f} ms"
            )
        if self.errors:
            top = ", ".join(
                f"{k} x{v}" for k, v in sorted(self.errors.items(), key=lambda kv: -kv[1])
            )
            parts.append(f"errors: {top}")
        return "; ".join(parts)

    def log_to_mlflow(self, job: BaseJob | None = None, prefix: str = "distribatch.") -> None:
        """Log metrics (and the job's identity as params) to the active MLflow run
        (one is started if none is active)."""
        mlflow = require("mlflow", "mlflow")
        metrics = {
            f"{prefix}{k}": float(v)
            for k, v in self.to_dict().items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        }
        metrics.update({f"{prefix}errors.{k}": float(v) for k, v in self.errors.items()})
        mlflow.log_metrics(metrics)
        if job is not None:
            mlflow.log_params(
                {
                    f"{prefix}task": job.task,
                    f"{prefix}model": job.model.model_dump_json(),
                    f"{prefix}engine": job.engine,
                    f"{prefix}fingerprint": job.fingerprint(),
                }
            )


def summarize(output: Any, output_col: str = "output") -> RunStats:
    """Statistics of a `run()` result (Spark DataFrame, pandas DataFrame or list)."""
    _, error_col, meta_col = io.output_columns(output_col)
    if io.is_spark_dataframe(output):
        return _summarize_spark(output, error_col, meta_col)
    if io.is_pandas_dataframe(output):
        _require_columns(list(output.columns), error_col, meta_col)
        rows = zip(output[error_col].tolist(), output[meta_col].tolist(), strict=True)
        return summarize_rows(rows)
    if isinstance(output, list):
        if output:
            _require_columns(list(output[0]), error_col, meta_col)
        return summarize_rows((row[error_col], row[meta_col]) for row in output)
    raise ConfigError(f"cannot summarize a {type(output).__name__}")


def summarize_rows(rows: Iterable[tuple[Any, Mapping[str, Any] | None]]) -> RunStats:
    """Statistics from (error, meta) pairs."""
    n = ok = reused = retries = prompt = completion = 0
    audio = 0.0
    latencies: list[float] = []
    errors: Counter[str] = Counter()
    for error, meta in rows:
        meta = meta or {}
        n += 1
        if _is_missing(error):
            ok += 1
            if meta.get("attempts") == 0:
                reused += 1
        else:
            errors[_error_class(str(error))] += 1
        retries += max(int(meta.get("attempts") or 0) - 1, 0)
        prompt += int(meta.get("prompt_tokens") or 0)
        completion += int(meta.get("completion_tokens") or 0)
        audio += float(meta.get("audio_seconds") or 0.0)
        if meta.get("latency_ms") is not None:
            latencies.append(float(meta["latency_ms"]))
    return RunStats(
        rows=n,
        succeeded=ok,
        failed=n - ok,
        reused=reused,
        retries=retries,
        prompt_tokens=prompt,
        completion_tokens=completion,
        audio_seconds=audio,
        latency_ms_p50=_percentile(latencies, 0.5),
        latency_ms_p95=_percentile(latencies, 0.95),
        errors=dict(errors),
    )


def _summarize_spark(df: Any, error_col: str, meta_col: str) -> RunStats:
    F = require("pyspark.sql.functions", "spark")
    _require_columns(list(df.columns), error_col, meta_col)
    err, meta = F.col(f"`{error_col}`"), F.col(f"`{meta_col}`")
    ok = err.isNull()
    row = df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(ok, 1).otherwise(0)).alias("ok"),
        F.sum(F.when(ok & (meta["attempts"] == 0), 1).otherwise(0)).alias("reused"),
        F.sum(F.greatest(meta["attempts"] - 1, F.lit(0))).alias("retries"),
        F.sum(meta["prompt_tokens"]).alias("prompt"),
        F.sum(meta["completion_tokens"]).alias("completion"),
        F.sum(meta["audio_seconds"]).alias("audio"),
        F.percentile_approx(meta["latency_ms"], [0.5, 0.95]).alias("latency"),
    ).first()
    classes = (
        df.where(err.isNotNull())
        .groupBy(F.regexp_extract(err, _ERROR_CLASS.pattern, 1).alias("cls"))
        .count()
        .collect()
    )
    latency = row["latency"] or [None, None]
    n = int(row["rows"] or 0)
    succeeded = int(row["ok"] or 0)
    return RunStats(
        rows=n,
        succeeded=succeeded,
        failed=n - succeeded,
        reused=int(row["reused"] or 0),
        retries=int(row["retries"] or 0),
        prompt_tokens=int(row["prompt"] or 0),
        completion_tokens=int(row["completion"] or 0),
        audio_seconds=float(row["audio"] or 0.0),
        latency_ms_p50=latency[0],
        latency_ms_p95=latency[1],
        errors={(r["cls"] or "Error"): int(r["count"]) for r in classes},
    )


def _require_columns(columns: list[str], *needed: str) -> None:
    missing = [c for c in needed if c not in columns]
    if missing:
        raise ConfigError(
            f"output has no column(s) {', '.join(missing)}; pass the output_col used in run()"
        )


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _error_class(error: str) -> str:
    m = _ERROR_CLASS.match(error)
    return m[1] if m else "Error"


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]
