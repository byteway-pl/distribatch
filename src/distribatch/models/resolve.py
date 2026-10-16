"""Turn a PathModel / UCModel into local Hugging Face directories (on the worker).

Layouts understood:
- a plain HF directory (``config.json`` + weights, tokenizer files alongside);
- an MLflow ``transformers`` flavor artifact (``model/`` with weights,
  ``components/tokenizer`` / ``components/feature_extractor``);
- an MLflow artifact saved without weights (``save_pretrained=False``): the
  ``source_model_name`` recorded in ``MLmodel`` is used as a Hub repo id;
- any directory tree containing exactly one HF model directory (searched shallowest
  first), e.g. sentence-transformers exports.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distribatch._deps import require
from distribatch.models._cache import entry_name, materialize
from distribatch.models.specs import ModelSpec, PathModel, UCModel
from distribatch.types import ConfigError, format_error

_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf", "*.index.json")
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.txt")


@dataclass(frozen=True)
class LocalModel:
    path: str  # HF model directory (or Hub repo id for reference-only MLflow models)
    tokenizer: str  # directory/repo with the tokenizer
    processor: str  # directory/repo with the feature extractor (audio models)
    source: str  # human-readable origin, reported as `model_id`
    revision: str | None = None


def resolve_local(spec: ModelSpec) -> LocalModel:
    if isinstance(spec, PathModel):
        path = Path(fuse_path(spec.path))
        if not path.is_dir():
            raise ConfigError(f"model directory {spec.path!r} does not exist on this worker")
        if spec.copy_to_local:
            path = materialize(
                entry_name("path", str(path), _fingerprint(path)),
                lambda tmp: shutil.copytree(path, tmp, dirs_exist_ok=True),
            )
        return locate(path, source=spec.path)
    if isinstance(spec, UCModel):
        root, version = download_uc(spec)
        return locate(root, source=f"{spec.name}/{version}")
    raise ConfigError(f"{spec.kind!r} models have no local weights")


def fuse_path(path: str) -> str:
    """Local (FUSE) path for a Databricks/Spark path: ``dbfs:/Volumes/x`` ->
    ``/Volumes/x`` (UC Volumes, as `binaryFile` reports them), ``dbfs:/x`` -> ``/dbfs/x``,
    ``file:/x`` -> ``/x``."""
    if path.startswith("dbfs:/"):
        rest = path.removeprefix("dbfs:").lstrip("/")
        return "/" + rest if rest.startswith("Volumes/") else "/dbfs/" + rest
    if path.startswith("file:"):
        return "/" + path.removeprefix("file:").lstrip("/")
    return os.path.expanduser(path)


def download_uc(spec: UCModel, root: Path | None = None) -> tuple[Path, int]:
    """Download a UC model version into the node cache (or `root`): (dir, version)."""
    mlflow = require("mlflow", "mlflow")
    try:
        mlflow.set_registry_uri("databricks-uc")
        client = mlflow.MlflowClient()
        version = spec.version
        if version is None:
            version = int(client.get_model_version_by_alias(spec.name, spec.alias).version)
    except Exception as exc:
        raise ConfigError(
            f"cannot resolve Unity Catalog model {spec.mlflow_uri}: {format_error(exc)}. "
            "Spark executors need DATABRICKS_HOST/DATABRICKS_TOKEN in the cluster "
            "environment, or set stage_dir= so the driver downloads it once."
        ) from exc
    uri = f"models:/{spec.name}/{version}"

    def fill(tmp: Path) -> None:
        mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(tmp))

    return materialize(entry_name("uc", spec.name, str(version)), fill, root=root), version


def stage_uc_model(spec: UCModel) -> PathModel:
    """Driver side: download a UC model into `spec.stage_dir` (e.g. a UC Volume) once,
    so executors copy it from there instead of talking to the registry."""
    assert spec.stage_dir is not None
    root, _ = download_uc(spec, root=Path(fuse_path(spec.stage_dir)))
    return PathModel(path=str(root), copy_to_local=True)


def locate(root: Path, *, source: str) -> LocalModel:
    weights = _find_model_dir(root)
    if weights is None:
        reference = _mlflow_source_model(root)
        if reference is None:
            raise ConfigError(f"no Hugging Face model (config.json + weights) under {root}")
        repo, revision = reference
        return LocalModel(repo, repo, repo, source=source, revision=revision)
    components = root / "components"
    tokenizer = weights
    if not _has_any(weights, _TOKENIZER_FILES) and (components / "tokenizer").is_dir():
        tokenizer = components / "tokenizer"
    processor = tokenizer
    if (weights / "preprocessor_config.json").exists():
        processor = weights
    else:
        for name in ("feature_extractor", "processor"):
            if (components / name / "preprocessor_config.json").exists():
                processor = components / name
                break
    return LocalModel(str(weights), str(tokenizer), str(processor), source=source)


def _find_model_dir(root: Path) -> Path | None:
    if _is_model_dir(root):
        return root
    candidates = sorted(
        (p.parent for p in root.rglob("config.json") if _is_model_dir(p.parent)),
        key=lambda p: (len(p.relative_to(root).parts), str(p)),
    )
    return candidates[0] if candidates else None


def _is_model_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and any(
        next(path.glob(pattern), None) is not None for pattern in _WEIGHT_PATTERNS
    )


def _has_any(path: Path, names: tuple[str, ...]) -> bool:
    return any((path / name).exists() for name in names)


def _mlflow_source_model(root: Path) -> tuple[str, str | None] | None:
    mlmodel = root / "MLmodel"
    if not mlmodel.is_file():
        return None
    text = mlmodel.read_text()
    name = re.search(r"^\s*source_model_name:\s*['\"]?([^'\"\s]+)", text, re.MULTILINE)
    revision = re.search(r"^\s*source_model_revision:\s*['\"]?([^'\"\s]+)", text, re.MULTILINE)
    if name is None:
        return None
    return name[1], (revision[1] if revision else None)


def _fingerprint(path: Path) -> str:
    """Changes when the top-level files change (re-copy on updated weights)."""
    parts: list[Any] = []
    for child in sorted(path.iterdir()):
        stat = child.stat()
        parts.append(f"{child.name}:{stat.st_size}:{stat.st_mtime_ns}")
    # sha256, not hash(): str hashes are randomized per process, and every worker
    # must compute the same cache key.
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
