"""faster-whisper engine (CTranslate2): fast Whisper transcription on GPU and CPU.

Models in CTranslate2 format (``model.bin``) are used as they are. Hugging Face Whisper
models (e.g. registered in Unity Catalog with the MLflow transformers flavor) are
converted once per node into the model cache (``ctranslate2`` converter; needs
`distribatch[transformers]` for the conversion only).

Audio is decoded by `audio.py` (same formats and errors as the transformers engine).
Files are transcribed independently; with ``num_workers > 1`` several run in parallel
(CTranslate2 releases the GIL). ``batched=True`` uses faster-whisper's VAD-based
batched pipeline, which speeds up long recordings on GPUs.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from distribatch import audio as audio_mod
from distribatch._deps import require
from distribatch.engines.base import Capabilities, Engine
from distribatch.models._cache import entry_name, materialize
from distribatch.models.resolve import LocalModel, resolve_local
from distribatch.models.specs import ModelSpec, PathModel, UCModel
from distribatch.types import (
    Completion,
    ConfigError,
    ResultMeta,
    TranscribeRequest,
    format_error,
)

_TOKENIZER_FILES = ("tokenizer.json", "preprocessor_config.json", "vocabulary.json")


class FasterWhisperSettings(BaseModel):
    """`engine_options` accepted by the faster_whisper engine (validated on the driver)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str = Field(default="auto", description='"auto", "cpu" or "cuda".')
    device_index: int | list[int] = Field(
        default=0, description="GPU index, or several for one model on multiple GPUs."
    )
    compute_type: str = Field(
        default="default", description='e.g. "float16", "int8_float16", "int8".'
    )
    cpu_threads: int = Field(default=0, ge=0, description="CPU threads; 0 = CTranslate2 default.")
    num_workers: int = Field(
        default=1, ge=1, description="Files transcribed in parallel (threads) per process."
    )
    beam_size: int = Field(default=5, ge=1, description="Beam search width; 1 = greedy.")
    vad_filter: bool = Field(default=False, description="Skip silence with Silero VAD.")
    batched: bool = Field(
        default=False, description="VAD-based batched pipeline (faster long audio on GPU)."
    )
    batch_size: int = Field(default=8, ge=1, description="Segments per batch with batched=True.")
    convert_quantization: str | None = Field(
        default="float16", description="Weight type when converting HF Whisper to CTranslate2."
    )


class FasterWhisperEngine(Engine):
    name = "faster_whisper"

    @classmethod
    def validate_options(cls, options: Mapping[str, Any]) -> None:
        _settings(options)

    def __init__(self, spec: ModelSpec, options: Mapping[str, Any]) -> None:
        if not isinstance(spec, PathModel | UCModel):
            raise ConfigError(f"faster_whisper engine needs a local model, got {spec.kind!r}")
        super().__init__(spec, options)
        self.settings = _settings(options)
        self.capabilities = Capabilities(tasks=frozenset({"transcribe"}))
        self.local: LocalModel | None = None
        self._model: Any = None
        self._batched: Any = None

    def load(self) -> None:
        fw = require("faster_whisper", "whisper")
        self.local = resolve_local(self.spec)
        model_dir = ct2_model_dir(self.local, self.settings.convert_quantization)
        s = self.settings
        self._model = fw.WhisperModel(
            model_dir,
            device=s.device,
            device_index=s.device_index,
            compute_type=s.compute_type,
            cpu_threads=s.cpu_threads,
            num_workers=s.num_workers,
        )
        if s.batched:
            self._batched = fw.BatchedInferencePipeline(model=self._model)

    def close(self) -> None:
        self._model = self._batched = None

    def transcribe(self, requests: Sequence[TranscribeRequest]) -> list[Completion]:
        if self.settings.num_workers == 1:
            return [self._one(r) for r in requests]
        with ThreadPoolExecutor(self.settings.num_workers) as pool:
            return list(pool.map(self._one, requests))

    def _one(self, request: TranscribeRequest) -> Completion:
        meta = ResultMeta(model_id=self.local.source if self.local else None)
        start = time.perf_counter()
        try:
            samples, duration = audio_mod.decode(request.audio)
            kwargs: dict[str, Any] = {
                "language": request.language,
                "task": "translate" if request.translate else "transcribe",
                "beam_size": self.settings.beam_size,
                "initial_prompt": request.prompt,
                "vad_filter": self.settings.vad_filter,
            }
            if self._batched is not None:
                segments, info = self._batched.transcribe(
                    samples, batch_size=self.settings.batch_size, **kwargs
                )
            else:
                segments, info = self._model.transcribe(samples, **kwargs)
            parts = [  # `segments` is lazy: decoding happens while iterating
                {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
                for s in segments
            ]
            text = " ".join(p["text"] for p in parts if p["text"])
            meta.audio_seconds = float(getattr(info, "duration", None) or duration)
            meta.latency_ms = (time.perf_counter() - start) * 1000
            output: Any = text
            if request.timestamps:
                language = request.language or getattr(info, "language", None)
                output = {"text": text, "language": language, "segments": parts}
            return Completion(request.row_id, output, None, meta)
        except Exception as exc:
            meta.latency_ms = (time.perf_counter() - start) * 1000
            return Completion(request.row_id, None, format_error(exc), meta)


def ct2_model_dir(local: LocalModel, quantization: str | None) -> str:
    """A CTranslate2 Whisper directory for `local`, converting HF weights if needed."""
    path = Path(local.path)
    if (path / "model.bin").is_file():
        return str(path)
    if path.is_dir():
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "whisper":
            raise ConfigError(
                f"faster_whisper needs a Whisper model, got model_type={config.get('model_type')!r}"
            )
    ct2 = require("ctranslate2", "whisper")

    def convert(tmp: Path) -> None:
        try:
            converter = ct2.converters.TransformersConverter(local.path, revision=local.revision)
            converter.convert(str(tmp), quantization=quantization, force=True)
        except ImportError as exc:
            raise ConfigError(
                "converting a Hugging Face Whisper model for faster_whisper needs "
                f"torch and transformers: pip install 'distribatch[transformers]' ({exc})"
            ) from exc
        # faster-whisper reads the tokenizer/feature-extractor config from the model dir.
        for source in (local.path, local.tokenizer, local.processor):
            for name in _TOKENIZER_FILES:
                file = Path(source) / name
                if file.is_file() and not (tmp / name).exists():
                    shutil.copy(file, tmp / name)

    return str(materialize(entry_name("ct2", local.path, str(quantization)), convert))


def _settings(options: Mapping[str, Any]) -> FasterWhisperSettings:
    try:
        return FasterWhisperSettings.model_validate(dict(options))
    except ValidationError as exc:
        raise ConfigError(f"invalid engine_options for the faster_whisper engine: {exc}") from exc
