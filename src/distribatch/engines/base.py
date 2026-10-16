"""Engine protocol, engine selection and the process-level engine cache."""

from __future__ import annotations

import importlib
import importlib.util
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from distribatch.config import EngineName
from distribatch.models.specs import ModelSpec
from distribatch.types import (
    ChatRequest,
    Completion,
    ConfigError,
    DistribatchError,
    EmbedRequest,
    Request,
    Task,
    TranscribeRequest,
)

_R = TypeVar("_R", ChatRequest, EmbedRequest, TranscribeRequest)


@dataclass(frozen=True)
class Capabilities:
    tasks: frozenset[Task] = field(default_factory=frozenset)
    structured_output: bool = False  # honours ChatRequest.json_schema natively
    needs_gpu: bool = False


class Engine:
    """Turns a batch of requests into completions.

    Engines never see DataFrames. Each task method returns exactly one completion per
    request (any order); per-request failures go into `Completion.error`. An exception
    escaping a task method fails the whole batch, so engines catch what they can.
    Subclasses override the methods of the tasks they support.
    """

    name: ClassVar[str]
    capabilities: Capabilities = Capabilities()

    def __init__(self, spec: ModelSpec, options: Mapping[str, Any]) -> None:
        self.spec = spec
        self.options = dict(options)

    @classmethod
    def validate_options(cls, options: Mapping[str, Any]) -> None:
        """Check `engine_options` on the driver (raise ConfigError). Engine modules must
        stay importable without their heavy dependencies for this to work."""

    def load(self) -> None:
        """Load weights / open clients. Called once per process by `get_engine`."""

    def generate(self, requests: Sequence[ChatRequest]) -> list[Completion]:
        raise self._unsupported("generate")

    def embed(self, requests: Sequence[EmbedRequest]) -> list[Completion]:
        raise self._unsupported("embed")

    def transcribe(self, requests: Sequence[TranscribeRequest]) -> list[Completion]:
        raise self._unsupported("transcribe")

    def close(self) -> None:
        """Release resources."""

    def run(self, task: Task, requests: Sequence[Request]) -> list[Completion]:
        """Dispatch a homogeneous batch of requests to the task method."""
        if task == "generate":
            return self.generate([_expect(r, ChatRequest) for r in requests])
        if task == "embed":
            return self.embed([_expect(r, EmbedRequest) for r in requests])
        return self.transcribe([_expect(r, TranscribeRequest) for r in requests])

    def _unsupported(self, task: Task) -> DistribatchError:
        return DistribatchError(f"engine {self.name!r} does not support task {task!r}")


def _expect(request: Request, kind: type[_R]) -> _R:
    if not isinstance(request, kind):
        raise TypeError(f"expected {kind.__name__}, got {type(request).__name__}")
    return request


@dataclass(frozen=True)
class _EngineInfo:
    """Static facts about an engine, known without importing it."""

    model_kinds: tuple[str, ...]
    tasks: frozenset[Task]
    impl: str | None = None  # "module:Class"; None = planned, not implemented yet
    features: frozenset[str] = frozenset()  # optional abilities, e.g. "embed_dimensions"


_ALL_TASKS: frozenset[Task] = frozenset({"generate", "embed", "transcribe"})

_ENGINES: dict[str, _EngineInfo] = {
    "fake": _EngineInfo(
        ("fake",),
        _ALL_TASKS,
        "distribatch.engines.fake:FakeEngine",
        frozenset({"embed_dimensions"}),
    ),
    "openai": _EngineInfo(
        ("openai",),
        _ALL_TASKS,
        "distribatch.engines.openai_api:OpenAIEngine",
        frozenset({"embed_dimensions"}),
    ),
    "vllm": _EngineInfo(
        ("uc", "path"),
        frozenset({"generate", "embed"}),
        "distribatch.engines.vllm_engine:VLLMEngine",
        frozenset({"embed_dimensions"}),
    ),
    "transformers": _EngineInfo(
        ("uc", "path"), _ALL_TASKS, "distribatch.engines.hf_transformers:TransformersEngine"
    ),
    "faster_whisper": _EngineInfo(
        ("uc", "path"),
        frozenset({"transcribe"}),
        "distribatch.engines.faster_whisper_engine:FasterWhisperEngine",
    ),
}

# Preference order for engine="auto", per model kind (filtered by task support).
_AUTO_ORDER: dict[str, tuple[str, ...]] = {
    "fake": ("fake",),
    "openai": ("openai",),
    "uc": ("vllm", "faster_whisper", "transformers"),
    "path": ("vllm", "faster_whisper", "transformers"),
}

# Engines picked by "auto" only when usable on this machine (checked on the worker).
_NEEDS_PROBE = {"vllm", "faster_whisper"}


def candidate_engines(spec: ModelSpec, requested: EngineName | str, task: Task) -> tuple[str, ...]:
    """Engines that could run `task` on `spec`, in preference order (config check)."""
    if requested == "auto":
        candidates = tuple(n for n in _AUTO_ORDER[spec.kind] if task in _ENGINES[n].tasks)
        if not candidates:
            raise ConfigError(f"no engine supports task {task!r} for a {spec.kind!r} model")
        return candidates
    info = _ENGINES.get(requested)
    if info is None:
        raise ConfigError(f"unknown engine {requested!r}; choose from {', '.join(_ENGINES)}")
    if spec.kind not in info.model_kinds:
        raise ConfigError(
            f"engine {requested!r} cannot run a {spec.kind!r} model; "
            f"it supports: {', '.join(info.model_kinds)}"
        )
    if task not in info.tasks:
        raise ConfigError(f"engine {requested!r} does not support task {task!r}")
    return (requested,)


def check_engine_available(spec: ModelSpec, requested: EngineName | str, task: Task) -> None:
    """Driver-side check that some implemented engine fits (fail fast)."""
    candidates = candidate_engines(spec, requested, task)
    if not any(_ENGINES[n].impl for n in candidates):
        raise ConfigError(f"engine {' / '.join(candidates)} is not implemented yet")


def validate_engine_options(
    spec: ModelSpec, requested: EngineName | str, task: Task, options: Mapping[str, Any]
) -> None:
    """Driver-side `engine_options` check when the engine is already determined."""
    candidates = [n for n in candidate_engines(spec, requested, task) if _ENGINES[n].impl]
    if len(candidates) == 1:
        _engine_class(candidates[0]).validate_options(options)


def check_feature(
    spec: ModelSpec, requested: EngineName | str, task: Task, feature: str, what: str
) -> None:
    """Driver-side: every engine that might run the job must support `feature`."""
    candidates = [n for n in candidate_engines(spec, requested, task) if _ENGINES[n].impl]
    lacking = [n for n in candidates if feature not in _ENGINES[n].features]
    if lacking:
        able = [n for n in _ENGINES if feature in _ENGINES[n].features and _ENGINES[n].impl]
        raise ConfigError(
            f"{what} is not supported by engine {' / '.join(lacking)}; "
            f"choose engine= one of: {', '.join(able)}"
        )


def resolve_engine(spec: ModelSpec, requested: EngineName | str, task: Task) -> str:
    """Pick the concrete engine. Runs on the worker: GPU visibility differs per node."""
    candidates = [n for n in candidate_engines(spec, requested, task) if _ENGINES[n].impl]
    if not candidates:
        raise ConfigError(f"engine {requested!r} is not implemented yet")
    if requested == "auto":
        usable = [n for n in candidates if n not in _NEEDS_PROBE or _usable(n)]
        candidates = usable or candidates[-1:]
    return candidates[0]


def _usable(name: str) -> bool:
    if name == "vllm":
        if importlib.util.find_spec("vllm") is None or importlib.util.find_spec("torch") is None:
            return False
        import torch

        return bool(torch.cuda.is_available())
    if name == "faster_whisper":
        return importlib.util.find_spec("faster_whisper") is not None
    return True


_CACHE: dict[str, Engine] = {}
_CACHE_LOCK = threading.Lock()


def get_engine(
    spec: ModelSpec, requested: EngineName | str, options: Mapping[str, Any], task: Task
) -> Engine:
    """The loaded engine for (spec, engine, options), created once per process."""
    name = resolve_engine(spec, requested, task)
    key = json.dumps(
        [name, spec.model_dump(mode="json"), dict(options)], sort_keys=True, default=repr
    )
    with _CACHE_LOCK:
        engine = _CACHE.get(key)
        if engine is None:
            engine = _engine_class(name)(spec, options)
            engine.load()
            _CACHE[key] = engine
    return engine


def clear_engine_cache() -> None:
    with _CACHE_LOCK:
        for engine in _CACHE.values():
            engine.close()
        _CACHE.clear()


def _engine_class(name: str) -> type[Engine]:
    impl = _ENGINES[name].impl
    if impl is None:
        raise ConfigError(f"engine {name!r} is not implemented yet")
    module_name, _, class_name = impl.partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, Engine)):
        raise DistribatchError(f"{impl} is not an Engine")
    return cls
