"""Distribatch — batch LLM generation, embeddings and transcription over DataFrames.

Importing this package must stay cheap: heavy dependencies (torch, vllm, transformers,
pyspark, openai, mlflow, databricks-sdk) are imported lazily by the engine or executor
that needs them.
"""

from distribatch import models
from distribatch.config import Sampling
from distribatch.job import BaseJob, EmbedJob, GenerateJob, TranscribeJob
from distribatch.outputs import OutputTypeError
from distribatch.stats import RunStats, summarize
from distribatch.types import (
    ConfigError,
    DistribatchError,
    EngineError,
    InvalidInputError,
    MissingDependencyError,
    PromptRenderError,
    SchemaValidationError,
    WorkerCrashedError,
)

__version__ = "0.1.0a1"

__all__ = [
    "BaseJob",
    "ConfigError",
    "DistribatchError",
    "EmbedJob",
    "EngineError",
    "GenerateJob",
    "InvalidInputError",
    "MissingDependencyError",
    "OutputTypeError",
    "PromptRenderError",
    "RunStats",
    "Sampling",
    "SchemaValidationError",
    "TranscribeJob",
    "WorkerCrashedError",
    "__version__",
    "models",
    "summarize",
]
