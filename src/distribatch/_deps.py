"""Lazy import of optional (heavy) dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType

from distribatch.types import MissingDependencyError

# Modules that must never be imported by `import distribatch`; enforced by
# tests/test_lazy_imports.py. Maps top-level module -> extra that provides it.
OPTIONAL_MODULES: dict[str, str] = {
    "torch": "transformers",
    "transformers": "transformers",
    "accelerate": "transformers",
    "soundfile": "audio",
    "scipy": "audio",
    "faster_whisper": "whisper",
    "ctranslate2": "whisper",
    "vllm": "vllm",
    "pyspark": "spark",
    "openai": "openai",
    "mlflow": "mlflow",
    "databricks.sdk": "databricks",
}


def require(module: str, extra: str | None = None) -> ModuleType:
    """Import `module` or raise MissingDependencyError naming the extra to install."""
    if extra is None:
        extra = OPTIONAL_MODULES.get(module, OPTIONAL_MODULES.get(module.split(".")[0], "all"))
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(module, extra) from exc
