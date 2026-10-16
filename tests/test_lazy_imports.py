"""`import distribatch` must not pull in any heavy optional dependency."""

import json
import subprocess
import sys

import pytest

from distribatch import MissingDependencyError
from distribatch._deps import OPTIONAL_MODULES, require


def test_import_does_not_load_heavy_modules() -> None:
    # Fresh interpreter: the test process itself may already have pyspark/openai loaded.
    code = (
        "import json, sys, distribatch, distribatch.models, distribatch.engines, "
        "distribatch.executors, distribatch.engines.openai_api, distribatch.engines.fake, "
        "distribatch.engines.hf_transformers, distribatch.engines.vllm_engine, "
        "distribatch.engines.faster_whisper_engine, "
        "distribatch.executors.spark, distribatch.executors.process_pool, "
        "distribatch.models.resolve, distribatch.audio, distribatch.outputs\n"
        f"heavy = {sorted(OPTIONAL_MODULES)!r}\n"
        "print(json.dumps([m for m in heavy if m in sys.modules]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert json.loads(out) == []


def test_require_missing_module_names_the_extra() -> None:
    with pytest.raises(MissingDependencyError, match=r"distribatch\[vllm\]"):
        require("definitely_not_installed_mod", extra="vllm")


def test_require_returns_module() -> None:
    assert require("json").__name__ == "json"
