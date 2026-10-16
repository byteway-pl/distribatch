"""OpenAI-compatible engine: chat completions, embeddings, audio transcription.

Works with the official OpenAI API, Databricks Model Serving (``base_url="databricks"``)
and any compatible server (vLLM, TGI, LiteLLM, ...). The `openai` package is imported
only in `load()`, so this module is importable on the driver for option validation.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from distribatch._deps import require
from distribatch.engines._aio import LoopThread
from distribatch.engines.base import Capabilities, Engine
from distribatch.models.specs import DATABRICKS, ModelSpec, OpenAIModel
from distribatch.prompt import with_inline_schema
from distribatch.types import (
    AudioInput,
    ChatMessage,
    ChatRequest,
    Completion,
    ConfigError,
    EmbedRequest,
    EngineError,
    InvalidInputError,
    ResultMeta,
    TranscribeRequest,
    format_error,
)

_T = TypeVar("_T")
_X = TypeVar("_X")

_RETRYABLE_STATUS = {408, 409, 429}


class OpenAISettings(BaseModel):
    """`engine_options` accepted by the openai engine (validated on the driver)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_concurrency: int = Field(
        default=16, ge=1, description="Concurrent requests per worker process."
    )
    requests_per_minute: float | None = Field(
        default=None, gt=0, description="Request-start rate limit per worker process."
    )
    timeout_s: float = Field(default=120.0, gt=0, description="Per-request timeout in seconds.")
    max_retries: int = Field(
        default=5, ge=0, description="Retries on 408/409/429/5xx, timeouts and connection errors."
    )
    backoff_base_s: float = Field(
        default=1.0, ge=0, description="First retry delay; doubles per retry, with jitter."
    )
    backoff_max_s: float = Field(
        default=60.0, ge=0, description="Upper bound for a retry delay (also caps Retry-After)."
    )
    structured_output: bool = Field(
        default=True,
        description="Send output_schema as a json_schema response_format (falls back to "
        "prompting automatically if the server rejects it).",
    )
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = Field(
        default="max_tokens",
        description="Request field for Sampling.max_tokens (reasoning models need "
        "max_completion_tokens).",
    )
    embed_max_inputs: int = Field(default=256, ge=1, description="Texts per embeddings request.")
    max_upload_mb: float = Field(
        default=25.0, gt=0, description="Audio files above this size are row errors."
    )


class OpenAIEngine(Engine):
    name = "openai"

    @classmethod
    def validate_options(cls, options: Mapping[str, Any]) -> None:
        _settings(options)

    def __init__(self, spec: ModelSpec, options: Mapping[str, Any]) -> None:
        if not isinstance(spec, OpenAIModel):
            raise ConfigError(f"openai engine needs an OpenAIModel, got {spec.kind!r}")
        super().__init__(spec, options)
        self.api = spec
        self.settings = _settings(options)
        self.capabilities = Capabilities(
            tasks=frozenset({"generate", "embed", "transcribe"}),
            structured_output=self.settings.structured_output,
        )
        # Flipped off (for this process) if the server rejects `response_format`.
        self._native_schema = self.settings.structured_output
        self._loop: LoopThread | None = None
        self._client: Any = None
        self._openai: Any = None
        self._limiter: _RateLimiter | None = None

    # --- lifecycle --------------------------------------------------------------------

    def load(self) -> None:
        self._openai = require("openai")
        base_url, api_key = resolve_credentials(self.api)
        self._loop = LoopThread(name="distribatch-openai")
        self._client = self._loop.run(self._make_client(base_url, api_key))
        if self.settings.requests_per_minute:
            self._limiter = _RateLimiter(self.settings.requests_per_minute)

    async def _make_client(self, base_url: str | None, api_key: str) -> Any:
        # Created on the engine's loop: the underlying httpx client binds to it.
        return self._openai.AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=self.settings.timeout_s, max_retries=0
        )

    def close(self) -> None:
        if self._loop is None:
            return
        if self._client is not None:
            with contextlib.suppress(Exception):  # best effort on shutdown
                self._loop.run(self._client.close())
        self._loop.stop()
        self._loop = None

    # --- tasks ------------------------------------------------------------------------

    def generate(self, requests: Sequence[ChatRequest]) -> list[Completion]:
        return self._run(self._gather(requests, self._chat_one))

    def embed(self, requests: Sequence[EmbedRequest]) -> list[Completion]:
        n = self.settings.embed_max_inputs
        chunks = [list(requests[i : i + n]) for i in range(0, len(requests), n)]
        nested = self._run(self._gather(chunks, self._embed_chunk))
        return [c for chunk in nested for c in chunk]

    def transcribe(self, requests: Sequence[TranscribeRequest]) -> list[Completion]:
        return self._run(self._gather(requests, self._transcribe_one))

    def _run(self, coro: Awaitable[_T]) -> _T:
        if self._loop is None:
            raise EngineError("openai engine is not loaded")
        return self._loop.run(_as_coroutine(coro))

    async def _gather(self, items: Sequence[_X], fn: Callable[[_X], Awaitable[_T]]) -> list[_T]:
        semaphore = asyncio.Semaphore(self.settings.max_concurrency)

        async def guarded(item: _X) -> _T:
            async with semaphore:
                return await fn(item)

        return list(await asyncio.gather(*(guarded(item) for item in items)))

    # --- chat -------------------------------------------------------------------------

    async def _chat_one(self, request: ChatRequest) -> Completion:
        meta = ResultMeta(attempts=0, model_id=self.api.model)
        start = time.perf_counter()
        try:
            messages, schema = request.messages, request.json_schema
            if schema is not None and not self._native_schema:
                messages, schema = with_inline_schema(messages, schema), None
            try:
                response = await self._call(partial(self._chat, request, messages, schema), meta)
            except self._openai.BadRequestError as exc:
                if schema is None or not _mentions_response_format(exc):
                    raise
                self._native_schema = False  # server can't do it: inline from now on
                messages = with_inline_schema(messages, schema)
                response = await self._call(partial(self._chat, request, messages, None), meta)
            choice = response.choices[0] if response.choices else None
            text = choice.message.content if choice is not None else None
            if response.usage is not None:
                meta.prompt_tokens = response.usage.prompt_tokens
                meta.completion_tokens = response.usage.completion_tokens
            meta.model_id = response.model or meta.model_id
            meta.finish_reason = choice.finish_reason if choice is not None else None
            if text is None:
                raise EngineError("response has no message content")
            return _done(request.row_id, text, meta, start)
        except Exception as exc:
            return _failed(request.row_id, exc, meta, start)

    async def _chat(
        self,
        request: ChatRequest,
        messages: tuple[ChatMessage, ...],
        json_schema: dict[str, Any] | None,
    ) -> Any:
        sampling = request.sampling
        kwargs: dict[str, Any] = {
            "model": self.api.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": sampling.temperature,
            self.settings.max_tokens_param: sampling.max_tokens,
        }
        if sampling.top_p is not None:
            kwargs["top_p"] = sampling.top_p
        if sampling.stop:
            kwargs["stop"] = list(sampling.stop)
        if sampling.seed is not None:
            kwargs["seed"] = sampling.seed
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema},
            }
        return await self._client.chat.completions.create(**kwargs)

    # --- embeddings -------------------------------------------------------------------

    async def _embed_chunk(self, chunk: list[EmbedRequest]) -> list[Completion]:
        meta = ResultMeta(attempts=0, model_id=self.api.model)
        start = time.perf_counter()
        try:
            response = await self._call(partial(self._embeddings, chunk), meta)
        except Exception as exc:
            if len(chunk) > 1 and _is_client_error(exc):
                # One bad input fails the whole request: retry row by row to isolate it.
                singles = await asyncio.gather(*(self._embed_chunk([r]) for r in chunk))
                return [c for single in singles for c in single]
            return [_failed(r.row_id, exc, _copy(meta), start) for r in chunk]

        data = sorted(response.data, key=lambda d: d.index)
        if len(data) != len(chunk):
            error = EngineError(f"{len(data)} embeddings returned for {len(chunk)} inputs")
            return [_failed(r.row_id, error, _copy(meta), start) for r in chunk]
        meta.model_id = response.model or meta.model_id
        if len(chunk) == 1 and response.usage is not None:
            # Usage is per request; only attributable to a row when the batch is one row.
            meta.prompt_tokens = response.usage.prompt_tokens
        return [
            _done(r.row_id, list(d.embedding), _copy(meta), start)
            for r, d in zip(chunk, data, strict=True)
        ]

    async def _embeddings(self, chunk: list[EmbedRequest]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.api.model,
            "input": [r.text for r in chunk],
            "encoding_format": "float",
        }
        if chunk[0].dimensions is not None:
            kwargs["dimensions"] = chunk[0].dimensions
        return await self._client.embeddings.create(**kwargs)

    # --- audio ------------------------------------------------------------------------

    async def _transcribe_one(self, request: TranscribeRequest) -> Completion:
        meta = ResultMeta(attempts=0, model_id=self.api.model)
        start = time.perf_counter()
        try:
            filename, data = await asyncio.to_thread(_read_audio, request.audio)
            size_mb = len(data) / 1024 / 1024
            if size_mb > self.settings.max_upload_mb:
                raise InvalidInputError(
                    f"audio is {size_mb:.1f} MB, over the {self.settings.max_upload_mb:g} MB "
                    "upload limit"
                )
            response = await self._call(partial(self._audio, request, filename, data), meta)
            text = _field(response, "text")
            if not isinstance(text, str):
                raise EngineError("transcription response has no text")
            duration = _field(response, "duration")
            meta.audio_seconds = float(duration) if duration is not None else None
            if not request.timestamps:
                return _done(request.row_id, text, meta, start)
            segments = [
                {"start": _field(s, "start"), "end": _field(s, "end"), "text": _field(s, "text")}
                for s in _field(response, "segments") or []
            ]
            language = _field(response, "language") or request.language
            output = {"text": text, "language": language, "segments": segments}
            return _done(request.row_id, output, meta, start)
        except Exception as exc:
            return _failed(request.row_id, exc, meta, start)

    async def _audio(self, request: TranscribeRequest, filename: str, data: bytes) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.api.model,
            "file": (filename, data),
            "response_format": "verbose_json" if request.timestamps else "json",
        }
        if request.prompt:
            kwargs["prompt"] = request.prompt
        if request.translate:
            return await self._client.audio.translations.create(**kwargs)
        if request.language:
            kwargs["language"] = request.language
        if request.timestamps:
            kwargs["timestamp_granularities"] = ["segment"]
        return await self._client.audio.transcriptions.create(**kwargs)

    # --- retries ----------------------------------------------------------------------

    async def _call(self, fn: Callable[[], Awaitable[_T]], meta: ResultMeta) -> _T:
        """Call with rate limiting and retries; `meta.attempts` counts every try."""
        retries = 0
        while True:
            if self._limiter is not None:
                await self._limiter.acquire()
            meta.attempts += 1
            try:
                return await fn()
            except Exception as exc:
                if retries >= self.settings.max_retries or not self._retryable(exc):
                    raise
                retries += 1
                await asyncio.sleep(self._backoff(retries, exc))

    def _retryable(self, exc: Exception) -> bool:
        oa = self._openai
        if isinstance(exc, oa.APITimeoutError | oa.APIConnectionError):
            return True
        if isinstance(exc, oa.APIStatusError):
            return exc.status_code in _RETRYABLE_STATUS or exc.status_code >= 500
        return False

    def _backoff(self, retry: int, exc: Exception) -> float:
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return min(retry_after, self.settings.backoff_max_s)
        cap = min(self.settings.backoff_max_s, self.settings.backoff_base_s * 2.0 ** (retry - 1))
        return cap / 2 + random.uniform(0, cap / 2)


def resolve_credentials(spec: OpenAIModel) -> tuple[str | None, str]:
    """(base_url, api_key) on the worker. Raises ConfigError with a hint when missing."""
    if spec.base_url == DATABRICKS:
        host, token = _databricks_host_and_token(need_token=spec.api_key_env is None)
        if spec.api_key_env is not None:
            token = _env_key(spec.api_key_env)
        return f"{host}/serving-endpoints", token
    if spec.api_key_env is not None:
        return spec.base_url, _env_key(spec.api_key_env)
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return spec.base_url, key
    if spec.base_url is not None:
        # Self-hosted OpenAI-compatible servers often need no key at all.
        return spec.base_url, "unused"
    raise ConfigError(
        "no API key: set OPENAI_API_KEY or pass api_key_env= naming the variable that "
        "holds it (on Spark it must be set on the executors, e.g. via the cluster's "
        "environment variables)"
    )


def _env_key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"environment variable {name} is not set on this worker (on Spark, set it in "
            "the cluster's environment variables, e.g. {{secrets/<scope>/<key>}})"
        )
    return value


def _databricks_host_and_token(*, need_token: bool) -> tuple[str, str]:
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if host and (token or not need_token):
        return _normalize_host(host), token
    try:
        core = require("databricks.sdk.core", "databricks")
        config = core.Config()
        headers: dict[str, str] = config.authenticate()
    except Exception as exc:
        raise ConfigError(
            "cannot resolve Databricks host/token: set DATABRICKS_HOST and DATABRICKS_TOKEN "
            f"on the workers or configure databricks-sdk auth ({format_error(exc)})"
        ) from exc
    token = token or headers.get("Authorization", "").removeprefix("Bearer ")
    return _normalize_host(host or config.host), token


def _normalize_host(host: str) -> str:
    host = host.strip().rstrip("/")
    return host if host.startswith(("http://", "https://")) else f"https://{host}"


def _settings(options: Mapping[str, Any]) -> OpenAISettings:
    try:
        return OpenAISettings.model_validate(dict(options))
    except ValidationError as exc:
        raise ConfigError(f"invalid engine_options for the openai engine: {exc}") from exc


class _RateLimiter:
    """Spaces request starts evenly: at most `per_minute` per minute per process."""

    def __init__(self, per_minute: float) -> None:
        self._interval = 60.0 / per_minute
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)


_AUDIO_SIGNATURES: tuple[tuple[int, bytes, str], ...] = (
    (8, b"WAVE", "wav"),
    (0, b"fLaC", "flac"),
    (0, b"OggS", "ogg"),
    (0, b"ID3", "mp3"),
    (0, b"\xff\xfb", "mp3"),
    (0, b"\xff\xf3", "mp3"),
    (4, b"ftyp", "m4a"),
    (0, b"\x1a\x45\xdf\xa3", "webm"),
)


def _read_audio(audio: AudioInput) -> tuple[str, bytes]:
    """(filename, bytes) to upload. APIs detect the format from the file extension."""
    if audio.data is not None:
        return f"audio.{_sniff_extension(audio.data)}", audio.data
    assert audio.path is not None
    try:
        return Path(audio.path).name, Path(audio.path).read_bytes()
    except OSError as exc:
        raise InvalidInputError(f"cannot read audio file {audio.path!r}: {exc}") from exc


def _sniff_extension(data: bytes) -> str:
    for offset, magic, ext in _AUDIO_SIGNATURES:
        if data[offset : offset + len(magic)] == magic:
            return ext
    return "wav"


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def _mentions_response_format(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "json_schema" in text


def _is_client_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in _RETRYABLE_STATUS


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(str(value)) if value is not None else None
    except ValueError:
        return None


def _copy(meta: ResultMeta) -> ResultMeta:
    return ResultMeta(**meta.to_dict())


def _done(row_id: int, output: Any, meta: ResultMeta, start: float) -> Completion:
    meta.latency_ms = (time.perf_counter() - start) * 1000
    return Completion(row_id, output, None, meta)


def _failed(row_id: int, exc: BaseException, meta: ResultMeta, start: float) -> Completion:
    meta.latency_ms = (time.perf_counter() - start) * 1000
    return Completion(row_id, None, format_error(exc), meta)


async def _as_coroutine(awaitable: Awaitable[_T]) -> _T:
    return await awaitable
