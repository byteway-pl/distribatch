"""Hugging Face transformers engine: generation, embeddings and speech recognition.

The CPU-capable fallback for local models (vLLM is preferred on GPUs for generation
and embeddings). Model files come from `models.resolve`; each task's components are
loaded on first use. Heavy imports happen in `load()` only.

- generate: causal LMs with the tokenizer's chat template (plain "User:/Assistant:"
  fallback when there is none). Structured output is prompted (no native support).
- embed: AutoModel + pooling. sentence-transformers exports are honoured
  (``modules.json``: Transformer, Pooling cls/mean/max/lasttoken, Normalize;
  ``sentence_bert_config.json`` max_seq_length) without needing that package.
- transcribe: the ``automatic-speech-recognition`` pipeline (Whisper and others),
  chunked long-form audio, optional segment timestamps; audio decoded by `audio.py`.

Micro-batches that fail with out-of-memory are split in half and retried; other
micro-batch failures become row errors for that micro-batch only.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from distribatch import audio as audio_mod
from distribatch._deps import require
from distribatch.engines.base import Capabilities, Engine
from distribatch.models.resolve import LocalModel, resolve_local
from distribatch.models.specs import ModelSpec, PathModel, UCModel
from distribatch.types import (
    ChatMessage,
    ChatRequest,
    Completion,
    ConfigError,
    EmbedRequest,
    ResultMeta,
    TranscribeRequest,
    format_error,
)

_R = TypeVar("_R", ChatRequest, EmbedRequest, TranscribeRequest)

_ST_MODULES = {
    "sentence_transformers.models.Transformer",
    "sentence_transformers.models.Pooling",
    "sentence_transformers.models.Normalize",
}


class TransformersSettings(BaseModel):
    """`engine_options` accepted by the transformers engine (validated on the driver)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str = Field(
        default="auto", description='"auto" (cuda if available, else cpu), "cpu", "cuda:1", "mps".'
    )
    device_map: str | None = Field(
        default=None, description='e.g. "auto" to shard a big model over GPUs (needs accelerate).'
    )
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = Field(
        default="auto", description="auto = bfloat16/float16 on CUDA, float32 otherwise."
    )
    trust_remote_code: bool = Field(
        default=False, description="Allow custom model code from the model repository."
    )
    micro_batch_size: int = Field(
        default=8, ge=1, description="Rows per forward pass (halved automatically on OOM)."
    )
    pooling: Literal["auto", "mean", "cls", "max", "last"] = Field(
        default="auto",
        description="Embedding pooling; auto = from the sentence-transformers export, else mean.",
    )
    max_input_tokens: int | None = Field(
        default=None, ge=1, description="Embedding input truncation (default: model limit)."
    )
    chunk_length_s: float = Field(
        default=30.0, gt=0, description="Speech recognition: chunk audio longer than this."
    )
    attn_implementation: str | None = Field(
        default=None, description='e.g. "sdpa" or "flash_attention_2".'
    )


class TransformersEngine(Engine):
    name = "transformers"

    @classmethod
    def validate_options(cls, options: Mapping[str, Any]) -> None:
        _settings(options)

    def __init__(self, spec: ModelSpec, options: Mapping[str, Any]) -> None:
        if not isinstance(spec, PathModel | UCModel):
            raise ConfigError(f"transformers engine needs a local model, got {spec.kind!r}")
        super().__init__(spec, options)
        self.settings = _settings(options)
        self.capabilities = Capabilities(
            tasks=frozenset({"generate", "embed", "transcribe"}), structured_output=False
        )
        self._torch: Any = None
        self._tf: Any = None
        self.local: LocalModel | None = None
        self._device: str = "cpu"
        self._lm: tuple[Any, Any] | None = None
        self._embedder: _Embedder | None = None
        self._asr: Any = None

    def load(self) -> None:
        self._torch = require("torch", "transformers")
        self._tf = require("transformers", "transformers")
        self.local = resolve_local(self.spec)
        device = self.settings.device
        if device == "auto":
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
        self._device = device

    def close(self) -> None:
        self._lm = self._embedder = self._asr = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # --- shared loading helpers ---------------------------------------------------------

    def _model_kwargs(self) -> dict[str, Any]:
        assert self.local is not None
        kwargs: dict[str, Any] = {"trust_remote_code": self.settings.trust_remote_code}
        if self.local.revision:
            kwargs["revision"] = self.local.revision
        dtype = self._dtype()
        if dtype is not None:
            # transformers >= 4.56 renamed `torch_dtype` to `dtype` (5.x dropped the old).
            kwargs["dtype" if _version_at_least(self._tf, 4, 56) else "torch_dtype"] = dtype
        if self.settings.device_map:
            kwargs["device_map"] = self.settings.device_map
        if self.settings.attn_implementation:
            kwargs["attn_implementation"] = self.settings.attn_implementation
        return kwargs

    def _dtype(self) -> Any:
        torch = self._torch
        name = self.settings.dtype
        if name == "auto":
            if not self._device.startswith("cuda"):
                return torch.float32
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return getattr(torch, name)

    def _place(self, model: Any) -> Any:
        if not self.settings.device_map:
            model = model.to(self._device)
        return model.eval()

    def _tokenizer(self, padding_side: str) -> Any:
        assert self.local is not None
        tok = self._tf.AutoTokenizer.from_pretrained(
            self.local.tokenizer,
            trust_remote_code=self.settings.trust_remote_code,
            revision=self.local.revision,
        )
        tok.padding_side = padding_side
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        return tok

    def _meta(self) -> ResultMeta:
        return ResultMeta(model_id=self.local.source if self.local else None)

    # --- generate -----------------------------------------------------------------------

    def generate(self, requests: Sequence[ChatRequest]) -> list[Completion]:
        if self._lm is None:
            assert self.local is not None
            tok = self._tokenizer("left")  # decoder-only generation pads on the left
            model = self._tf.AutoModelForCausalLM.from_pretrained(
                self.local.path, **self._model_kwargs()
            )
            self._lm = (tok, self._place(model))
        groups: dict[Any, list[ChatRequest]] = defaultdict(list)
        for request in requests:
            groups[request.sampling].append(request)
        out: list[Completion] = []
        for group in groups.values():
            out.extend(self._micro_batches(group, self._generate_batch))
        return out

    def _generate_batch(self, requests: list[ChatRequest]) -> list[Completion]:
        assert self._lm is not None
        torch = self._torch
        tok, model = self._lm
        start = time.perf_counter()
        templated = getattr(tok, "chat_template", None) is not None
        prompts = [_render_chat(tok, r.messages, templated) for r in requests]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=not templated).to(
            model.device
        )
        sampling = requests[0].sampling
        kwargs: dict[str, Any] = {
            "max_new_tokens": sampling.max_tokens,
            "do_sample": sampling.temperature > 0,
            "pad_token_id": tok.pad_token_id,
        }
        if sampling.temperature > 0:
            kwargs["temperature"] = sampling.temperature
            if sampling.top_p is not None:
                kwargs["top_p"] = sampling.top_p
        if sampling.seed is not None:
            torch.manual_seed(sampling.seed)
        with torch.inference_mode():
            generated = model.generate(**enc, **kwargs)
        new_tokens = generated[:, enc["input_ids"].shape[1] :].tolist()
        latency = (time.perf_counter() - start) * 1000 / len(requests)
        stop_ids = {tok.eos_token_id, tok.pad_token_id} - {None}
        out: list[Completion] = []
        for i, request in enumerate(requests):
            ids = new_tokens[i]
            cut = next((j for j, t in enumerate(ids) if t in stop_ids), None)
            finish = "stop" if cut is not None else "length"
            ids = ids if cut is None else ids[:cut]
            text = tok.decode(ids, skip_special_tokens=True)
            for stop in sampling.stop or ():
                if stop in text:
                    text, finish = text[: text.index(stop)], "stop"
            meta = self._meta()
            meta.prompt_tokens = int(enc["attention_mask"][i].sum())
            meta.completion_tokens = len(ids)
            meta.finish_reason = finish
            meta.latency_ms = latency
            out.append(Completion(request.row_id, text, None, meta))
        return out

    # --- embed --------------------------------------------------------------------------

    def embed(self, requests: Sequence[EmbedRequest]) -> list[Completion]:
        if self._embedder is None:
            self._embedder = self._load_embedder()
        return self._micro_batches(requests, self._embed_batch)

    def _load_embedder(self) -> _Embedder:
        assert self.local is not None
        root = Path(self.local.path)
        pooling, normalize, max_len = _sentence_transformers_config(root)
        if self.settings.pooling != "auto":
            pooling = self.settings.pooling
        tok = self._tokenizer("right")
        model = self._place(self._tf.AutoModel.from_pretrained(str(root), **self._model_kwargs()))
        limit = self.settings.max_input_tokens or max_len
        if limit is None:
            positions = getattr(model.config, "max_position_embeddings", None) or 512
            limit = min(int(tok.model_max_length), int(positions))
        return _Embedder(tok, model, pooling or "mean", normalize, limit)

    def _embed_batch(self, requests: list[EmbedRequest]) -> list[Completion]:
        assert self._embedder is not None
        torch = self._torch
        e = self._embedder
        start = time.perf_counter()
        enc = e.tokenizer(
            [r.text for r in requests],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=e.max_length,
        ).to(e.model.device)
        with torch.inference_mode():
            hidden = e.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if e.pooling == "cls":
            pooled = hidden[:, 0]
        elif e.pooling == "max":
            pooled = hidden.masked_fill(mask == 0, float("-inf")).max(dim=1).values
        elif e.pooling == "last":
            last = enc["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        else:
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if e.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors = pooled.float().cpu().tolist()
        latency = (time.perf_counter() - start) * 1000 / len(requests)
        out = []
        for i, request in enumerate(requests):
            meta = self._meta()
            meta.prompt_tokens = int(enc["attention_mask"][i].sum())
            meta.latency_ms = latency
            out.append(Completion(request.row_id, vectors[i], None, meta))
        return out

    # --- transcribe ---------------------------------------------------------------------

    def transcribe(self, requests: Sequence[TranscribeRequest]) -> list[Completion]:
        if self._asr is None:
            assert self.local is not None
            kwargs = self._model_kwargs()
            if not self.settings.device_map:
                kwargs["device"] = self._device
            self._asr = self._tf.pipeline(
                "automatic-speech-recognition",
                model=self.local.path,
                tokenizer=self.local.tokenizer,
                feature_extractor=self.local.processor,
                **kwargs,
            )
        rate = int(getattr(self._asr.feature_extractor, "sampling_rate", 16_000))
        out: list[Completion] = []
        decoded: dict[int, tuple[Any, float]] = {}
        for request in requests:
            try:
                decoded[request.row_id] = audio_mod.decode(request.audio, rate)
            except Exception as exc:
                out.append(Completion(request.row_id, None, format_error(exc), self._meta()))
        groups: dict[tuple[Any, ...], list[TranscribeRequest]] = defaultdict(list)
        for request in requests:
            if request.row_id in decoded:
                key = (request.language, request.prompt, request.timestamps, request.translate)
                groups[key].append(request)
        for group in groups.values():
            out.extend(
                self._micro_batches(group, lambda batch: self._asr_batch(batch, decoded, rate))
            )
        return out

    def _asr_batch(
        self,
        requests: list[TranscribeRequest],
        decoded: Mapping[int, tuple[Any, float]],
        rate: int,
    ) -> list[Completion]:
        first = requests[0]
        generate_kwargs: dict[str, Any] = {}
        if first.language:
            generate_kwargs["language"] = first.language
        if first.translate:
            generate_kwargs["task"] = "translate"
        if first.prompt:
            ids = self._asr.tokenizer.get_prompt_ids(first.prompt, return_tensors="pt")
            generate_kwargs["prompt_ids"] = ids.to(self._device)
        start = time.perf_counter()
        inputs = [{"raw": decoded[r.row_id][0], "sampling_rate": rate} for r in requests]
        call: dict[str, Any] = {
            "batch_size": len(inputs),
            "return_timestamps": first.timestamps,
            "generate_kwargs": generate_kwargs,
        }
        if any(decoded[r.row_id][1] > self.settings.chunk_length_s for r in requests):
            # Long-form only: audio within one window needs no chunking (and chunking
            # forces timestamp decoding in recent transformers).
            call["chunk_length_s"] = self.settings.chunk_length_s
        results = self._asr(inputs, **call)
        latency = (time.perf_counter() - start) * 1000 / len(requests)
        out = []
        for request, result in zip(requests, results, strict=True):
            duration = decoded[request.row_id][1]
            text = str(result.get("text", "")).strip()
            output: Any = text
            if request.timestamps:
                segments = []
                for chunk in result.get("chunks") or []:
                    begin, end = chunk.get("timestamp") or (None, None)
                    segments.append(
                        {
                            "start": float(begin or 0.0),
                            "end": float(end if end is not None else duration),
                            "text": str(chunk.get("text", "")).strip(),
                        }
                    )
                output = {"text": text, "language": request.language, "segments": segments}
            meta = self._meta()
            meta.audio_seconds = duration
            meta.latency_ms = latency
            out.append(Completion(request.row_id, output, None, meta))
        return out

    # --- micro-batching -----------------------------------------------------------------

    def _micro_batches(
        self, requests: Sequence[_R], fn: Callable[[list[_R]], list[Completion]]
    ) -> list[Completion]:
        size = self.settings.micro_batch_size
        out: list[Completion] = []
        for i in range(0, len(requests), size):
            out.extend(self._guarded(list(requests[i : i + size]), fn))
        return out

    def _guarded(
        self, requests: list[_R], fn: Callable[[list[_R]], list[Completion]]
    ) -> list[Completion]:
        try:
            return fn(requests)
        except Exception as exc:
            if len(requests) > 1 and _is_oom(exc):
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
                mid = len(requests) // 2
                return self._guarded(requests[:mid], fn) + self._guarded(requests[mid:], fn)
            error = format_error(exc)
            return [Completion(r.row_id, None, error, self._meta()) for r in requests]


class _Embedder:
    def __init__(
        self, tokenizer: Any, model: Any, pooling: str, normalize: bool, max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.pooling = pooling
        self.normalize = normalize
        self.max_length = max_length


def _render_chat(tok: Any, messages: tuple[ChatMessage, ...], templated: bool) -> str:
    chat = [{"role": m.role, "content": m.content} for m in messages]
    if templated:
        return str(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
    parts = [
        m["content"] if m["role"] == "system" else f"{m['role'].capitalize()}: {m['content']}"
        for m in chat
    ]
    return "\n\n".join(parts) + "\n\nAssistant:"


def _sentence_transformers_config(root: Path) -> tuple[str | None, bool, int | None]:
    """(pooling, normalize, max_seq_length) from a sentence-transformers export."""
    modules_file = root / "modules.json"
    if not modules_file.is_file():
        return None, False, None
    modules = json.loads(modules_file.read_text())
    unsupported = [m["type"] for m in modules if m.get("type") not in _ST_MODULES]
    if unsupported:
        raise ConfigError(
            f"sentence-transformers modules not supported by the transformers engine: "
            f"{', '.join(unsupported)}"
        )
    pooling: str | None = None
    for module in modules:
        if module["type"].endswith(".Pooling"):
            cfg = json.loads((root / module["path"] / "config.json").read_text())
            pooling = _pooling_mode(cfg)
    normalize = any(m["type"].endswith(".Normalize") for m in modules)
    max_len = None
    bert_cfg = root / "sentence_bert_config.json"
    if bert_cfg.is_file():
        max_len = json.loads(bert_cfg.read_text()).get("max_seq_length")
    return pooling, normalize, int(max_len) if max_len else None


def _pooling_mode(cfg: Mapping[str, Any]) -> str:
    modes = {
        "pooling_mode_cls_token": "cls",
        "pooling_mode_mean_tokens": "mean",
        "pooling_mode_max_tokens": "max",
        "pooling_mode_lasttoken": "last",
    }
    for key, mode in modes.items():
        if cfg.get(key):
            return mode
    enabled = [k for k, v in cfg.items() if k.startswith("pooling_mode") and v]
    raise ConfigError(f"unsupported sentence-transformers pooling: {', '.join(enabled)}")


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def _version_at_least(module: Any, major: int, minor: int) -> bool:
    parts = str(getattr(module, "__version__", "0.0")).split(".")
    try:
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (ValueError, IndexError):
        return True


def _settings(options: Mapping[str, Any]) -> TransformersSettings:
    try:
        return TransformersSettings.model_validate(dict(options))
    except ValidationError as exc:
        raise ConfigError(f"invalid engine_options for the transformers engine: {exc}") from exc
