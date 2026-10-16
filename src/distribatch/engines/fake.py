"""Deterministic fake engine for tests and dry runs (every task, no weights/network)."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import struct
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, TypeVar

from distribatch.engines.base import Capabilities, Engine
from distribatch.models.specs import FakeModel, ModelSpec
from distribatch.types import (
    ChatRequest,
    Completion,
    ConfigError,
    EmbedRequest,
    ResultMeta,
    TranscribeRequest,
)

DEFAULT_EMBEDDING_DIM = 8
FAKE_AUDIO_SECONDS = 1.0

_Req = TypeVar("_Req", ChatRequest, EmbedRequest, TranscribeRequest)


class FakeEngine(Engine):
    name = "fake"
    # Class-wide counter so tests can assert that weights are "loaded" once per process.
    load_count: ClassVar[int] = 0

    def __init__(self, spec: ModelSpec, options: Mapping[str, Any]) -> None:
        if not isinstance(spec, FakeModel):
            raise ConfigError(f"fake engine needs a FakeModel, got {spec.kind!r}")
        super().__init__(spec, options)
        self.fake = spec
        self.capabilities = Capabilities(
            tasks=frozenset({"generate", "embed", "transcribe"}),
            structured_output=spec.structured_output,
        )

    def load(self) -> None:
        type(self).load_count += 1

    def generate(self, requests: Sequence[ChatRequest]) -> list[Completion]:
        def one(request: ChatRequest, prompt: str, meta: ResultMeta) -> Any:
            text = self.fake.response if self.fake.response is not None else prompt
            meta.prompt_tokens = sum(len(m.content.split()) for m in request.messages)
            meta.completion_tokens = len(text.split())
            meta.finish_reason = "stop"
            return text

        return self._each(requests, _user_text, one)

    def embed(self, requests: Sequence[EmbedRequest]) -> list[Completion]:
        def one(request: EmbedRequest, text: str, meta: ResultMeta) -> Any:
            meta.prompt_tokens = len(text.split())
            return _hash_vector(text, request.dimensions or DEFAULT_EMBEDDING_DIM)

        return self._each(requests, lambda r: r.text, one)

    def transcribe(self, requests: Sequence[TranscribeRequest]) -> list[Completion]:
        def one(request: TranscribeRequest, label: str, meta: ResultMeta) -> Any:
            if self.fake.response is not None:
                text = self.fake.response
            elif request.audio.data is not None:
                text = request.audio.data.decode("utf-8", errors="replace")
            else:
                text = f"transcript of {label}"
            meta.audio_seconds = FAKE_AUDIO_SECONDS
            if not request.timestamps:
                return text
            segment = {"start": 0.0, "end": FAKE_AUDIO_SECONDS, "text": text}
            return {"text": text, "language": request.language or "en", "segments": [segment]}

        return self._each(requests, lambda r: r.audio.label, one)

    def _each(
        self,
        requests: Sequence[_Req],
        probe: Callable[[_Req], str],
        produce: Callable[[_Req, str, ResultMeta], Any],
    ) -> list[Completion]:
        """Shared fake behaviour: crash/fail markers matched against `probe(request)`."""
        probes = [probe(r) for r in requests]
        if self.fake.crash_marker and any(self.fake.crash_marker in p for p in probes):
            raise RuntimeError("fake engine crash")
        if self.fake.exit_marker and any(self.fake.exit_marker in p for p in probes):
            if multiprocessing.parent_process() is None:
                raise RuntimeError("fake engine process exit (simulated in main process)")
            os._exit(13)
        out: list[Completion] = []
        for request, probed in zip(requests, probes, strict=True):
            start = time.perf_counter()
            if self.fake.delay_ms:
                time.sleep(self.fake.delay_ms / 1000)
            meta = ResultMeta(model_id="fake")
            if self.fake.fail_marker and self.fake.fail_marker in probed:
                output, error = None, "EngineError: fake failure"
            else:
                output, error = produce(request, probed, meta), None
            meta.latency_ms = (time.perf_counter() - start) * 1000
            out.append(Completion(request.row_id, output, error, meta))
        return out


def _user_text(request: ChatRequest) -> str:
    return next((m.content for m in reversed(request.messages) if m.role == "user"), "")


def _hash_vector(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding in [-1, 1) derived from the text."""
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        values.extend(v / 2**31 for v in struct.unpack("<8i", digest))
        counter += 1
    return values[:dim]
