"""Shared contract types and the error hierarchy.

Requests (`ChatRequest`, `EmbedRequest`, `TranscribeRequest`) and `Completion` are the
only interface between jobs/executors and engines; `Result` is what the pipeline hands
back to executors for assembling the output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from distribatch.config import Sampling


class DistribatchError(Exception):
    """Base class for all errors raised by distribatch."""


class ConfigError(DistribatchError):
    """Invalid job configuration. Always raised before any work is scheduled.

    Deliberately not a ValueError: pydantic would wrap it into a ValidationError
    when raised from a validator, hiding the type from callers.
    """


class MissingDependencyError(DistribatchError, ImportError):
    """An optional dependency needed by the selected engine/executor is not installed."""

    def __init__(self, module: str, extra: str) -> None:
        self.module = module
        self.extra = extra
        super().__init__(
            f"'{module}' is required for this feature. "
            f"Install it with: pip install 'distribatch[{extra}]'"
        )

    def __reduce__(self) -> tuple[type[MissingDependencyError], tuple[str, str]]:
        # Picklable with its custom __init__ (crosses process boundaries).
        return type(self), (self.module, self.extra)


class InvalidInputError(DistribatchError):
    """A row's input cannot be used, e.g. a null audio cell (row-level)."""


class PromptRenderError(InvalidInputError):
    """A row could not be rendered into a prompt/text template (row-level)."""


class EngineError(DistribatchError):
    """The engine failed to produce a completion (row-level)."""


class WorkerCrashedError(DistribatchError):
    """The worker process died (OOM, segfault, ...) while processing the row (row-level)."""


class SchemaValidationError(DistribatchError):
    """The model output did not match the requested output schema (row-level)."""


def format_error(exc: BaseException) -> str:
    """Row-level error string: ``"<ErrorClass>: <message>"``."""
    return f"{type(exc).__name__}: {exc}"


Task = Literal["generate", "embed", "transcribe"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    row_id: int
    messages: tuple[ChatMessage, ...]
    sampling: Sampling
    json_schema: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EmbedRequest:
    row_id: int
    text: str
    dimensions: int | None = None


@dataclass(frozen=True, slots=True)
class AudioInput:
    """Exactly one of `path` (read by the worker) or `data` (raw file bytes)."""

    path: str | None = None
    data: bytes | None = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.data is None):
            raise ValueError("AudioInput needs exactly one of `path` or `data`")

    @property
    def label(self) -> str:
        """Short human-readable description (for errors/logs; never the bytes)."""
        return self.path if self.path is not None else f"<{len(self.data or b'')} bytes>"


@dataclass(frozen=True, slots=True)
class TranscribeRequest:
    row_id: int
    audio: AudioInput
    language: str | None = None
    prompt: str | None = None
    timestamps: bool = False
    translate: bool = False


Request = ChatRequest | EmbedRequest | TranscribeRequest


@dataclass(slots=True)
class ResultMeta:
    """Same fields for every task (inapplicable ones stay None) — one Spark struct."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    audio_seconds: float | None = None
    latency_ms: float | None = None
    attempts: int = 1
    model_id: str | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Completion:
    """Raw engine output for one request: `output` on success, `error` otherwise.

    `output` is a str (chat, plain transcript), list[float] (embedding) or dict
    (transcript with segments).
    """

    row_id: int
    output: Any
    error: str | None = None
    meta: ResultMeta = field(default_factory=ResultMeta)


@dataclass(frozen=True, slots=True)
class Result:
    """Final per-row outcome after the job's post-processing."""

    row_id: int
    value: Any
    error: str | None
    meta: ResultMeta
