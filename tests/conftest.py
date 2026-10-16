import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from openai_stub import OpenAIStub

REPO_ROOT = Path(__file__).resolve().parent.parent
# Test models and caches stay inside the repo (git-ignored .cache/, on whatever disk the
# repo lives on), never in ~/.cache. Remove them with scripts/clean_caches.sh.
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("DISTRIBATCH_CACHE_DIR", str(REPO_ROOT / ".cache" / "distribatch"))

# Small Hugging Face models for the local engine tests (~200 MB in total).
TINY_MODELS = {
    "llm": (
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        ["*.json", "*.safetensors", "tokenizer.model"],
    ),
    "embedder": (
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        ["*.json", "*.safetensors", "vocab.txt", "1_Pooling/*"],
    ),
    # Real (smallest) Whisper: the tiny-random one lacks the generation config that
    # transformers 5 needs. ~150 MB.
    "whisper": ("openai/whisper-tiny", ["*.json", "*.txt", "model.safetensors"]),
}

# Read by workers of every backend. Spark's JVM (and its Python workers) inherit the
# environment at session start, so it must be set before any SparkSession exists.
os.environ.setdefault("STUB_KEY", "sk-test")

# JDK homes tried when JAVA_HOME is unset and `java` on PATH does not work
# (macOS ships a /usr/bin/java stub that fails without a registered JDK).
_JDK_CANDIDATES = [
    "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
    "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home",
    "/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home",
]


def _works(java: str) -> bool:
    try:
        return subprocess.run([java, "-version"], capture_output=True).returncode == 0
    except OSError:
        return False


def _ensure_java() -> bool:
    home = os.environ.get("JAVA_HOME")
    if home and _works(str(Path(home) / "bin" / "java")):
        return True
    java = shutil.which("java")
    if java is not None and _works(java):
        return True
    for candidate in _JDK_CANDIDATES:
        if _works(str(Path(candidate) / "bin" / "java")):
            os.environ["JAVA_HOME"] = candidate
            return True
    return False


JAVA_AVAILABLE = _ensure_java()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if JAVA_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="Java runtime not available for local pyspark")
    for item in items:
        if "spark" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def spark() -> Iterator[Any]:
    from pyspark.sql import SparkSession

    # Python workers must run this interpreter (the system python may be older).
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    session = (
        SparkSession.builder.master("local[2]")
        .appName("distribatch-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.task.maxFailures", "1")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def tiny_models() -> dict[str, Path]:
    """Local directories of the tiny test models (downloaded once into HF_HOME)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from huggingface_hub import snapshot_download

    paths = {}
    for key, (repo, patterns) in TINY_MODELS.items():
        try:
            paths[key] = Path(snapshot_download(repo, allow_patterns=patterns))
        except Exception as exc:  # offline
            pytest.skip(f"cannot download {repo}: {exc}")
    return paths


@pytest.fixture(autouse=True)
def _fresh_engine_cache() -> Iterator[None]:
    from distribatch.engines.base import clear_engine_cache
    from distribatch.engines.fake import FakeEngine

    clear_engine_cache()
    FakeEngine.load_count = 0
    yield
    clear_engine_cache()


@pytest.fixture
def openai_stub() -> Iterator["OpenAIStub"]:
    from openai_stub import OpenAIStub

    stub = OpenAIStub()
    yield stub
    stub.close()
