"""Job classes: the declarative definition of a batch inference run, one per task.

A job knows how to turn one input row into an engine request (`build_request`) and how
to post-process the engine output (`finish`); executors and the pipeline stay
task-agnostic.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from distribatch import io, outputs
from distribatch.config import Backend, EngineName, Sampling
from distribatch.engines.base import (
    Capabilities,
    check_engine_available,
    check_feature,
    validate_engine_options,
)
from distribatch.models import ModelSpec, parse
from distribatch.models.resolve import fuse_path
from distribatch.outputs import OutType
from distribatch.prompt import CompiledPrompt, Template, compile_prompt
from distribatch.types import (
    AudioInput,
    ChatRequest,
    ConfigError,
    EmbedRequest,
    EngineError,
    InvalidInputError,
    Request,
    Task,
    TranscribeRequest,
)

DEFAULT_BATCH_SIZE = 32


class BaseJob(BaseModel, ABC):
    """Shared job fields and the `run`/`preview` entry points.

    Jobs are immutable and picklable; `run` decides where (backend) and on what data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    task: ClassVar[Task]

    model: ModelSpec
    engine: EngineName = "auto"
    engine_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model", mode="before")
    @classmethod
    def _parse_model_uri(cls, v: Any) -> Any:
        return parse(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def _check_engine(self) -> BaseJob:
        validate_engine_options(self.model, self.engine, self.task, self.engine_options)
        return self

    # --- task hooks -------------------------------------------------------------------

    @abstractmethod
    def required_columns(self) -> tuple[str, ...]:
        """Input columns the job reads (validated up front)."""

    @abstractmethod
    def build_request(
        self, row_id: int, record: Mapping[str, Any], capabilities: Capabilities
    ) -> Request:
        """Engine request for one row. Raising makes it a row-level error."""

    @abstractmethod
    def finish(self, output: Any) -> Any:
        """Final value from the engine output. Raising makes it a row-level error."""

    @abstractmethod
    def output_type(self) -> OutType:
        """Type of the values `finish` returns (drives the Spark output schema)."""

    # --- running ----------------------------------------------------------------------

    def run(
        self,
        data: Any,
        *,
        backend: Backend = "local",
        output_col: str = "output",
        batch_size: int = DEFAULT_BATCH_SIZE,
        workers: int | None = None,
        partitions: int | None = None,
        dedupe: bool = False,
        checkpoint: str | None = None,
        id_col: str | None = None,
        retry_failed: bool = True,
        checkpoint_every: int | None = None,
    ) -> Any:
        """Run the job over `data` and return it with the output columns added:
        ``<output_col>``, ``<output_col>_error`` (None on success), ``<output_col>_meta``.

        - ``backend="local"`` / ``"multiprocessing"``: pandas DataFrame or list of dicts
          in, same type out. `workers`: worker processes (default: one per GPU slot for
          local models, a few for API models).
        - ``backend="spark"``: Spark DataFrame in, Spark DataFrame out. Without a
          checkpoint the result is **lazy**: inference runs when an action executes it,
          and again on every further action unless you write or cache it. `partitions`:
          repartition the input first (parallelism = tasks x engine concurrency).
        - `dedupe`: send identical inputs (same values in the job's input columns) to
          the model once; copies get ``attempts=0`` in their meta. On Spark per batch.
        - `checkpoint` (a directory; local path, ``/Volumes/...`` or ``dbfs:/``) with
          `id_col` (unique ids): results are persisted as they complete, and a rerun
          skips ids already done (failed ones too unless `retry_failed`). The
          directory is bound to this job's fingerprint. On Spark the run then executes
          eagerly in waves of `checkpoint_every` rows (default 10 000; local backends
          flush every 1 000) and returns a DataFrame reading the checkpoint.
        """
        if batch_size < 1:
            raise ConfigError("batch_size must be >= 1")
        if not output_col:
            raise ConfigError("output_col must be a non-empty string")
        if backend not in ("local", "multiprocessing", "spark"):
            raise ConfigError(f"unknown backend {backend!r}")
        if workers is not None and backend != "multiprocessing":
            raise ConfigError("workers= only applies to backend='multiprocessing'")
        if partitions is not None and backend != "spark":
            raise ConfigError("partitions= only applies to backend='spark'")
        if partitions is not None and partitions < 1:
            raise ConfigError("partitions must be >= 1")
        if (checkpoint is None) != (id_col is None):
            raise ConfigError("checkpoint= and id_col= must be given together")
        if checkpoint_every is not None and checkpoint_every < 1:
            raise ConfigError("checkpoint_every must be >= 1")

        from distribatch import _run

        options = _run.RunOptions(
            output_col=output_col,
            batch_size=batch_size,
            workers=workers,
            partitions=partitions,
            dedupe=dedupe,
            checkpoint=checkpoint,
            id_col=id_col,
            retry_failed=retry_failed,
            checkpoint_every=checkpoint_every,
        )
        if backend == "spark":
            if not io.is_spark_dataframe(data):
                raise ConfigError("backend='spark' needs a Spark DataFrame")
            columns = list(data.columns)
            self._check_columns(columns, output_col)
            if id_col is not None and id_col not in columns:
                raise ConfigError(f"id column {id_col!r} is not in the input")
            check_engine_available(self.model, self.engine, self.task)
            return _run.run_spark(self, data, options)

        if io.is_spark_dataframe(data):
            raise ConfigError("Spark DataFrames need backend='spark'")
        inp = io.read_input(data)
        # An empty list of dicts has no known columns — nothing to validate against.
        known = inp.kind == "pandas" or bool(inp.records)
        self._check_columns(inp.columns, output_col, check_missing=known)
        check_engine_available(self.model, self.engine, self.task)
        results = _run.run_records(self, inp.records, backend, options)
        return io.write_output(inp, results, output_col)

    def fingerprint(self) -> str:
        """Identity of what this job computes (for checkpoints). Ignores options that
        don't change results: engine_options, model staging/copy, credential source."""
        data = self.model_dump(
            mode="json",
            exclude={
                "engine_options": True,
                "output_schema": True,
                "model": {"copy_to_local", "stage_dir", "api_key_env"},
            },
        )
        compiled = getattr(self, "_compiled", None)
        data["output_schema"] = compiled.json_schema if compiled is not None else None
        data["task"] = self.task
        payload = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def preview(self, data: Any, n: int = 5, *, output_col: str = "output") -> Any:
        """Run the first `n` rows locally — a quick check of inputs and outputs.

        A Spark DataFrame is sampled to the driver and returned as pandas.
        """
        return self.run(io.head(data, n), backend="local", output_col=output_col)

    def _check_columns(
        self, columns: list[str], output_col: str, *, check_missing: bool = True
    ) -> None:
        missing = [c for c in self.required_columns() if check_missing and c not in columns]
        if missing:
            raise ConfigError(
                f"job references column(s) missing from the input: {', '.join(missing)}; "
                f"available: {', '.join(columns) or '(none)'}"
            )
        clashing = [c for c in io.output_columns(output_col) if c in columns]
        if clashing:
            raise ConfigError(
                f"output column(s) already exist in the input: {', '.join(clashing)}; "
                "choose another output_col"
            )


class GenerateJob(BaseJob):
    """Chat/LLM generation; text output, or a validated dict with `output_schema`."""

    task: ClassVar[Task] = "generate"

    prompt: str
    system: str | None = None
    output_schema: type[BaseModel] | dict[str, Any] | None = None
    sampling: Sampling = Field(default_factory=Sampling)

    _compiled: CompiledPrompt = PrivateAttr()

    @model_validator(mode="after")
    def _compile(self) -> GenerateJob:
        self._compiled = compile_prompt(self.prompt, self.system, self.output_schema)
        return self

    @property
    def compiled_prompt(self) -> CompiledPrompt:
        return self._compiled

    def required_columns(self) -> tuple[str, ...]:
        return self._compiled.fields

    def build_request(
        self, row_id: int, record: Mapping[str, Any], capabilities: Capabilities
    ) -> ChatRequest:
        inline_schema = not capabilities.structured_output
        messages = self._compiled.build_messages(record, inline_schema=inline_schema)
        return ChatRequest(row_id, messages, self.sampling, self._compiled.json_schema)

    def finish(self, output: Any) -> Any:
        if not isinstance(output, str):
            raise EngineError(f"expected text output, got {type(output).__name__}")
        return self._compiled.parse_output(output)

    def output_type(self) -> OutType:
        schema = self._compiled.json_schema
        return outputs.STRING if schema is None else outputs.from_json_schema(schema)


class EmbedJob(BaseJob):
    """Text embeddings: one ``list[float]`` per row."""

    task: ClassVar[Task] = "embed"

    text: str
    dimensions: int | None = Field(default=None, gt=0)
    normalize: bool = False

    _template: Template = PrivateAttr()

    @model_validator(mode="after")
    def _compile(self) -> EmbedJob:
        self._template = Template.compile(self.text)
        if self.dimensions is not None:
            check_feature(self.model, self.engine, self.task, "embed_dimensions", "dimensions=")
        return self

    def required_columns(self) -> tuple[str, ...]:
        return self._template.fields

    def build_request(
        self, row_id: int, record: Mapping[str, Any], capabilities: Capabilities
    ) -> EmbedRequest:
        text = self._template.render(record)
        if not text.strip():
            raise InvalidInputError("text to embed is empty")
        return EmbedRequest(row_id, text, self.dimensions)

    def finish(self, output: Any) -> list[float]:
        if hasattr(output, "tolist"):  # numpy / torch vectors
            output = output.tolist()
        if not isinstance(output, list | tuple) or not output:
            raise EngineError(f"expected a non-empty vector, got {type(output).__name__}")
        try:
            vector = [float(v) for v in output]
        except (TypeError, ValueError) as exc:
            raise EngineError(f"embedding contains non-numeric values: {exc}") from exc
        if not all(math.isfinite(v) for v in vector):
            raise EngineError("embedding contains NaN or infinite values")
        if self.dimensions is not None and len(vector) != self.dimensions:
            raise EngineError(f"expected {self.dimensions} dimensions, got {len(vector)}")
        if self.normalize:
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]
        return vector

    def output_type(self) -> OutType:
        return outputs.array(outputs.FLOAT)


class TranscribeJob(BaseJob):
    """Speech-to-text. The audio column holds a path (str) or the file's bytes."""

    task: ClassVar[Task] = "transcribe"

    audio_col: str = Field(min_length=1)
    language: str | None = None
    prompt: str | None = None
    timestamps: bool = False
    translate: bool = False

    def required_columns(self) -> tuple[str, ...]:
        return (self.audio_col,)

    def build_request(
        self, row_id: int, record: Mapping[str, Any], capabilities: Capabilities
    ) -> TranscribeRequest:
        if self.audio_col not in record:
            raise InvalidInputError(f"row has no column {self.audio_col!r}")
        return TranscribeRequest(
            row_id,
            audio_input(record[self.audio_col]),
            language=self.language,
            prompt=self.prompt,
            timestamps=self.timestamps,
            translate=self.translate,
        )

    def finish(self, output: Any) -> Any:
        if not self.timestamps:
            if isinstance(output, Mapping):
                output = output.get("text")
            if not isinstance(output, str):
                raise EngineError(f"expected transcript text, got {type(output).__name__}")
            return output
        if not isinstance(output, Mapping) or not isinstance(output.get("text"), str):
            raise EngineError("expected a transcript with 'text' and 'segments'")
        try:
            segments = [
                {"start": float(s["start"]), "end": float(s["end"]), "text": str(s["text"])}
                for s in output.get("segments") or []
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise EngineError(f"malformed transcript segments: {exc}") from exc
        return {"text": output["text"], "language": output.get("language"), "segments": segments}

    def output_type(self) -> OutType:
        return outputs.TRANSCRIPT if self.timestamps else outputs.STRING


def audio_input(value: Any) -> AudioInput:
    """A cell of the audio column as an AudioInput (Spark paths mapped to FUSE paths)."""
    if isinstance(value, bytes | bytearray | memoryview):
        if not value:
            raise InvalidInputError("audio bytes are empty")
        return AudioInput(data=bytes(value))
    if isinstance(value, str):
        path = value.strip()
        if not path:
            raise InvalidInputError("audio path is empty")
        return AudioInput(path=fuse_path(path))
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise InvalidInputError("audio is null")
    raise InvalidInputError(
        f"audio column must hold a path (str) or bytes, got {type(value).__name__}"
    )
