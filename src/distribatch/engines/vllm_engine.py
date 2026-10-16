"""vLLM engine: offline batched generation (with guided JSON) and embeddings on GPUs.

One `vllm.LLM` per task, created on first use (generation and pooling models need
different engine modes). vLLM batches internally, so each call gets the whole batch.
`engine_options` are passed to `vllm.LLM(...)` as is (tensor_parallel_size,
gpu_memory_utilization, max_model_len, dtype, quantization, ...); the common ones are
type-checked on the driver.

API differences across vLLM versions are bridged here: structured outputs
(`StructuredOutputsParams` in newer releases, `GuidedDecodingParams` before) and the
pooling-mode switch (`runner="pooling"` vs. the older `task="embed"`).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from distribatch._deps import require
from distribatch.engines.base import Capabilities, Engine
from distribatch.models.resolve import LocalModel, resolve_local
from distribatch.models.specs import ModelSpec, PathModel, UCModel
from distribatch.types import (
    ChatRequest,
    Completion,
    ConfigError,
    EmbedRequest,
    ResultMeta,
    format_error,
)

_R = TypeVar("_R", ChatRequest, EmbedRequest)


class VLLMSettings(BaseModel):
    """Known `engine_options` (validated); unknown keys pass through to `vllm.LLM`."""

    model_config = ConfigDict(frozen=True, extra="allow")

    tensor_parallel_size: int = Field(
        default=1, ge=1, description="GPUs per model replica (also sizes multiprocessing slots)."
    )
    gpu_memory_utilization: float | None = Field(
        default=None, gt=0, le=1, description="Share of GPU memory vLLM may use."
    )
    max_model_len: int | None = Field(
        default=None, ge=1, description="Context length; longer prompts become row errors."
    )
    dtype: str | None = Field(default=None, description='e.g. "bfloat16", "float16".')
    quantization: str | None = Field(default=None, description='e.g. "awq", "gptq", "fp8".')
    trust_remote_code: bool = Field(
        default=False, description="Allow custom model code from the model repository."
    )


class VLLMEngine(Engine):
    name = "vllm"

    @classmethod
    def validate_options(cls, options: Mapping[str, Any]) -> None:
        _settings(options)

    def __init__(self, spec: ModelSpec, options: Mapping[str, Any]) -> None:
        if not isinstance(spec, PathModel | UCModel):
            raise ConfigError(f"vllm engine needs a local model, got {spec.kind!r}")
        super().__init__(spec, options)
        _settings(options)
        self.capabilities = Capabilities(
            tasks=frozenset({"generate", "embed"}), structured_output=True, needs_gpu=True
        )
        self._vllm: Any = None
        self.local: LocalModel | None = None
        self._llms: dict[str, Any] = {}

    def load(self) -> None:
        self._vllm = require("vllm", "vllm")
        self.local = resolve_local(self.spec)

    def close(self) -> None:
        self._llms.clear()

    def _llm(self, mode: str) -> Any:
        if mode not in self._llms:
            assert self.local is not None
            kwargs: dict[str, Any] = {
                "model": self.local.path,
                "tokenizer": self.local.tokenizer,
                **{k: v for k, v in self.options.items() if v is not None},
            }
            if self.local.revision:
                kwargs["revision"] = self.local.revision
            if mode == "generate":
                self._llms[mode] = self._vllm.LLM(**kwargs)
            else:
                try:
                    self._llms[mode] = self._vllm.LLM(runner="pooling", **kwargs)
                except TypeError:  # older vLLM
                    self._llms[mode] = self._vllm.LLM(task="embed", **kwargs)
        return self._llms[mode]

    def _meta(self) -> ResultMeta:
        return ResultMeta(model_id=self.local.source if self.local else None)

    # --- generate -----------------------------------------------------------------------

    def generate(self, requests: Sequence[ChatRequest]) -> list[Completion]:
        llm = self._llm("generate")
        return self._isolating(list(requests), lambda batch: self._chat(llm, batch))

    def _chat(self, llm: Any, requests: list[ChatRequest]) -> list[Completion]:
        start = time.perf_counter()
        conversations = [
            [{"role": m.role, "content": m.content} for m in r.messages] for r in requests
        ]
        params = [self._sampling_params(r) for r in requests]
        outputs = llm.chat(conversations, sampling_params=params, use_tqdm=False)
        latency = (time.perf_counter() - start) * 1000 / len(requests)
        out = []
        for request, result in zip(requests, outputs, strict=True):
            choice = result.outputs[0]
            meta = self._meta()
            meta.prompt_tokens = len(result.prompt_token_ids or [])
            meta.completion_tokens = len(choice.token_ids or [])
            meta.finish_reason = choice.finish_reason
            meta.latency_ms = latency
            out.append(Completion(request.row_id, choice.text, None, meta))
        return out

    def _sampling_params(self, request: ChatRequest) -> Any:
        s = request.sampling
        kwargs: dict[str, Any] = {"temperature": s.temperature, "max_tokens": s.max_tokens}
        if s.top_p is not None:
            kwargs["top_p"] = s.top_p
        if s.stop:
            kwargs["stop"] = list(s.stop)
        if s.seed is not None:
            kwargs["seed"] = s.seed
        if request.json_schema is not None:
            kwargs.update(self._structured(request.json_schema))
        return self._vllm.SamplingParams(**kwargs)

    def _structured(self, schema: dict[str, Any]) -> dict[str, Any]:
        params = self._vllm.sampling_params
        if hasattr(params, "StructuredOutputsParams"):
            return {"structured_outputs": params.StructuredOutputsParams(json=schema)}
        return {"guided_decoding": params.GuidedDecodingParams(json=schema)}

    # --- embed --------------------------------------------------------------------------

    def embed(self, requests: Sequence[EmbedRequest]) -> list[Completion]:
        llm = self._llm("embed")
        return self._isolating(list(requests), lambda batch: self._embed(llm, batch))

    def _embed(self, llm: Any, requests: list[EmbedRequest]) -> list[Completion]:
        start = time.perf_counter()
        kwargs: dict[str, Any] = {"use_tqdm": False}
        if requests[0].dimensions is not None:
            kwargs["pooling_params"] = self._vllm.PoolingParams(dimensions=requests[0].dimensions)
        outputs = llm.embed([r.text for r in requests], **kwargs)
        latency = (time.perf_counter() - start) * 1000 / len(requests)
        out = []
        for request, result in zip(requests, outputs, strict=True):
            meta = self._meta()
            meta.prompt_tokens = len(getattr(result, "prompt_token_ids", None) or [])
            meta.latency_ms = latency
            out.append(Completion(request.row_id, list(result.outputs.embedding), None, meta))
        return out

    # --- errors -------------------------------------------------------------------------

    def _isolating(
        self, requests: list[_R], fn: Callable[[list[_R]], list[Completion]]
    ) -> list[Completion]:
        """vLLM rejects a whole call for one bad prompt (e.g. longer than max_model_len):
        retry such a batch row by row so only the offending rows fail."""
        try:
            return fn(requests)
        except Exception as exc:
            if len(requests) == 1:
                return [Completion(requests[0].row_id, None, format_error(exc), self._meta())]
            return [c for r in requests for c in self._isolating([r], fn)]


def _settings(options: Mapping[str, Any]) -> VLLMSettings:
    try:
        return VLLMSettings.model_validate(dict(options))
    except ValidationError as exc:
        raise ConfigError(f"invalid engine_options for the vllm engine: {exc}") from exc
