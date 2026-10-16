"""Generation and engine configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EngineName = Literal["auto", "vllm", "transformers", "faster_whisper", "openai", "fake"]
Backend = Literal["local", "multiprocessing", "spark"]


class Sampling(BaseModel):
    """Generation parameters, shared by every row of a job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float = Field(
        default=0.0, ge=0.0, description="Sampling temperature; 0 = greedy (deterministic)."
    )
    max_tokens: int = Field(
        default=512, gt=0, description="Maximum number of generated tokens per row."
    )
    top_p: float | None = Field(
        default=None, gt=0.0, le=1.0, description="Nucleus sampling; None = engine default."
    )
    stop: tuple[str, ...] | None = Field(
        default=None, description="Stop sequences; generation ends before the first match."
    )
    seed: int | None = Field(default=None, description="Random seed, where the engine supports it.")
